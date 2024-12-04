import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
from pathlib import Path
import time
import io
import base64
import warnings
import easyocr
import torchvision
import json
import difflib
import gc
import signal
import asyncio
import sys
from functools import wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# Import Grounding DINO and SAM
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor

# Timeout decorator for async functions
def async_timeout(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request timeout")
        return wrapper
    return decorator

# Middleware for request timeout
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=300)  # 5 minutes timeout
        except asyncio.TimeoutError:
            return Response("Request timeout", status_code=504)

# Resource cleanup context manager
@asynccontextmanager
async def managed_resource():
    try:
        yield
    finally:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        # Close all matplotlib figures
        plt.close('all')

# Models and Config
class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"

class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class TextAlignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

@dataclass
class AppConfig:
    """Application configuration with type hints"""
    CONFIG_FILE: Path = Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDED_CHECKPOINT: Path = Path("groundingdino_swint_ogc.pth")
    SAM_CHECKPOINT: Path = Path("sam_vit_h_4b8939.pth")
    LOG_DIR: Path = Path("logs")
    ALLOWED_EXTENSIONS: set = field(default_factory=lambda: {ImageFormat.PNG.value, ImageFormat.JPEG.value, ImageFormat.JPG.value})
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    BOX_THRESHOLD: float = 0.3
    TEXT_THRESHOLD: float = 0.25
    MASK_PADDING: int = 5
    IOU_THRESHOLD: float = 0.5
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    MAX_CONCURRENT_REQUESTS: int = 10

class BoundingBox(BaseModel):
    x: float = Field(..., description="X-coordinate of top-left corner")
    y: float = Field(..., description="Y-coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")

class DetectedObject(BaseModel):
    object: str = Field(..., description="Detected object phrase")
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_text: str = Field(default="")
    text_alignment: Optional[TextAlignment] = Field(default=None)
    line_count: Optional[int] = Field(default=None)

class ProcessingResponse(BaseModel):
    status: ProcessingStatus
    message: str
    visualization: Optional[str] = None
    masked_output: Optional[str] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)

# Graceful shutdown handler
def handle_shutdown(signum, frame):
    logging.info("Received shutdown signal, cleaning up...")
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Force garbage collection
    gc.collect()
    # Close all matplotlib figures
    plt.close('all')
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Utility Functions
def determine_text_alignment(bbox: BoundingBox, boxes_in_group: List[BoundingBox] = None) -> TextAlignment:
    """
    Determine text alignment based on box positions.
    For text blocks, it checks variance in positions to determine alignment.
    """
    if not boxes_in_group:
        boxes_in_group = [bbox]
        return TextAlignment.LEFT  # Single line defaults to LEFT

    # Calculate left and right positions for all boxes
    left_positions = [box.x for box in boxes_in_group]
    right_positions = [box.x + box.width for box in boxes_in_group]
    
    # Calculate variances
    left_variance = max(left_positions) - min(left_positions)
    right_variance = max(right_positions) - min(right_positions)
    
    # Define thresholds
    threshold = 10  # Pixels threshold for considering positions aligned
    
    # If all lines start at almost same position, it's left aligned
    if left_variance <= threshold:
        return TextAlignment.LEFT
    
    # If all lines end at almost same position, it's right aligned
    if right_variance <= threshold:
        return TextAlignment.RIGHT
    
    # If neither left nor right aligned (high variance in both), it's center aligned
    return TextAlignment.CENTER

def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def are_boxes_nearby(box1: BoundingBox, box2: BoundingBox, distance_threshold: float = 50) -> bool:
    """Check if two bounding boxes are within a certain distance of each other"""
    center1_x = box1.x + box1.width / 2
    center1_y = box1.y + box1.height / 2
    center2_x = box2.x + box2.width / 2
    center2_y = box2.y + box2.height / 2

    horizontal_distance = abs(center1_x - center2_x)
    vertical_distance = abs(center1_y - center2_y)
    similar_height = abs(box1.height - box2.height) < min(box1.height, box2.height) * 0.5
    max_vertical_spacing = min(box1.height, box2.height) * 1.2
    y_aligned = vertical_distance < min(box1.height, box2.height) * 0.5
    x_overlap = (min(box1.x + box1.width, box2.x + box2.width) > max(box1.x, box2.x))
    proper_spacing = vertical_distance < max_vertical_spacing

    return (y_aligned or (x_overlap and proper_spacing)) and similar_height

def merge_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    """Merge multiple bounding boxes into one encompassing box"""
    x_min = min(box.x for box in boxes)
    y_min = min(box.y for box in boxes)
    x_max = max(box.x + box.width for box in boxes)
    y_max = max(box.y + box.height for box in boxes)

    return BoundingBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min
    )

def group_text_objects(objects: List[DetectedObject], distance_threshold: float = 50) -> List[DetectedObject]:
    """Group nearby text objects together"""
    if not objects:
        return []

    groups = []
    used_indices = set()

    for i, obj1 in enumerate(objects):
        if i in used_indices:
            continue

        current_group = {i}
        used_indices.add(i)

        changed = True
        while changed:
            changed = False
            for j, obj2 in enumerate(objects):
                if j in used_indices:
                    continue

                for idx in current_group:
                    if are_boxes_nearby(objects[idx].bbox, obj2.bbox, distance_threshold):
                        current_group.add(j)
                        used_indices.add(j)
                        changed = True
                        break

        groups.append(current_group)

    merged_objects = []
    for group in groups:
        group_objects = [objects[i] for i in group]
        
        # Sort objects by vertical position
        sorted_objects = sorted(group_objects, key=lambda obj: obj.bbox.y)
        merged_bbox = merge_boxes([obj.bbox for obj in group_objects])
        
        # Count lines based on number of original detections
        line_count = len(group_objects)
        
        # Get all bboxes in the group for alignment check
        group_boxes = [obj.bbox for obj in group_objects]
        
        # Determine text alignment from all boxes in group
        text_alignment = determine_text_alignment(merged_bbox, group_boxes)
        
        # Join texts with newline to preserve line breaks
        merged_text = '\n'.join(obj.detected_text for obj in sorted_objects)
        avg_confidence = sum(obj.confidence for obj in group_objects) / len(group_objects)
        
        merged_objects.append(DetectedObject(
            object="text",
            bbox=merged_bbox,
            confidence=avg_confidence,
            detected_text=merged_text,
            text_alignment=text_alignment,
            line_count=line_count
        ))

    return merged_objects

def load_image(image_path):
    """Load and transform image for model input"""
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    """Load and initialize the model"""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounded_output(model, image, caption, box_threshold, text_threshold, iou_threshold=0.5, device="cpu"):
    """Get model predictions with proper error handling and resource cleanup"""
    try:
        with torch.no_grad():
            caption = caption.lower().strip()
            if not caption.endswith("."):
                caption += "."
            model = model.to(device)
            image = image.to(device)
            
            outputs = model(image[None], captions=[caption])
            
            logits = outputs["pred_logits"].cpu().sigmoid()[0]
            boxes = outputs["pred_boxes"].cpu()[0]
            
            logits_filt = logits.clone()
            boxes_filt = boxes.clone()
            filt_mask = logits.max(dim=1)[0] > box_threshold
            logits_filt = logits[filt_mask]
            boxes_filt = boxes[filt_mask]
            
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            
            pred_phrases = []
            scores = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                pred_phrases.append(pred_phrase)
                scores.append(logit.max().item())

            if len(boxes_filt) > 0:
                scores_tensor = torch.tensor(scores)
                nms_idx = torchvision.ops.nms(boxes_filt, scores_tensor, iou_threshold).numpy().tolist()
                boxes_filt = boxes_filt[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                logits_filt = logits_filt[nms_idx]
            
            return boxes_filt, pred_phrases, logits_filt
    finally:
        # Cleanup CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def show_mask(mask, ax, random_color=False):
    """Display the segmentation mask on the given axis"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    """Display bounding box with label on the given axis"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_visualization(image, masks, boxes, phrases):
    """Save visualization of masks and boxes"""
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes, phrases):
            show_box(box.numpy() if isinstance(box, torch.Tensor) else box, plt.gca(), label)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf
    finally:
        plt.close('all')

def save_masked_output(image, masks, boxes, padding=5):
    """Save masked output with transparency"""
    height, width = image.shape[:2]
    transparent_mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    combined_mask = np.zeros((height, width), dtype=bool)
    for mask in masks:
        mask_np = mask.cpu().numpy()[0]
        kernel = np.ones((padding*2, padding*2), np.uint8)
        padded_mask = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1)
        combined_mask = combined_mask | padded_mask.astype(bool)
    
    transparent_mask[combined_mask] = np.concatenate([image[combined_mask], np.full((combined_mask.sum(), 1), 255)], axis=1)
    
    masked_image = Image.fromarray(transparent_mask)
    buf = io.BytesIO()
    masked_image.save(buf, format='PNG')
    buf.seek(0)
    return buf

class ImageProcessor:
    def __init__(self, config: AppConfig):
        """Initialize the image processor with configuration"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration"""
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_DIR / 'app.log'),
                logging.StreamHandler()
            ]
        )

    def _initialize_models(self):
        """Initialize all required models with proper error handling"""
        try:
            self.model = load_model(
                str(self.config.CONFIG_FILE),
                str(self.config.GROUNDED_CHECKPOINT),
                self.device
            )
            self.predictor = SamPredictor(
                build_sam(checkpoint=str(self.config.SAM_CHECKPOINT)).to(self.device)
            )
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logging.info("Models initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise RuntimeError("Failed to initialize models") from e

    def _cleanup_resources(self):
        """Clean up resources and free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        plt.close('all')

    def process_image(self, image_content: bytes, prompt: str, auto_detect_text: bool = False) -> ProcessingResponse:
        """Process image content with given prompt and options"""
        start_time = time.time()
        image_path = "temp_image.jpg"
        
        try:
            # Save temporary image
            with open(image_path, 'wb') as f:
                f.write(image_content)

            # Load and process image
            image_pil, image_tensor = load_image(image_path)
            
            # Get model predictions
            boxes_filt, pred_phrases, logits_filt = get_grounded_output(
                self.model, image_tensor, prompt,
                self.config.BOX_THRESHOLD,
                self.config.TEXT_THRESHOLD,
                self.config.IOU_THRESHOLD,
                device=self.device
            )

            # Process image with OpenCV
            image_cv2 = cv2.imread(image_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_cv2)

            objects = []
            masks = []
            boxes = []

            if len(boxes_filt) > 0:
                # Process detected objects
                size = image_pil.size
                H, W = size[1], size[0]
                boxes_filt = process_boxes(boxes_filt, W, H)
                
                transformed_boxes = self.predictor.transform.apply_boxes_torch(
                    boxes_filt, image_cv2.shape[:2]
                ).to(self.device)

                masks_output = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                
                prompt_masks, _, _ = masks_output

                masks.extend([m.cpu() for m in prompt_masks])
                boxes.extend([b for b in boxes_filt])

                # Process each detected object
                for box, phrase, logit in zip(boxes_filt, pred_phrases, logits_filt):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    roi = image_cv2[y1:y2, x1:x2]
                    
                    detected_text = ''
                    if roi.size > 0 and auto_detect_text:
                        try:
                            text_results = self.reader.readtext(roi)
                            detected_text = ' '.join([text[1] for text in text_results]) if text_results else ''
                        except Exception as e:
                            logging.error(f"Error in OCR: {str(e)}")

                    bbox_obj = BoundingBox(
                        x=float(box[0]),
                        y=float(box[1]),
                        width=float(box[2] - box[0]),
                        height=float(box[3] - box[1])
                    )

                    objects.append(DetectedObject(
                        object=phrase,
                        bbox=bbox_obj,
                        confidence=float(logit.max()),
                        detected_text=detected_text,
                        text_alignment=determine_text_alignment(bbox_obj) if detected_text else None,
                        line_count=1
                    ))

            # Perform text detection if requested
            if auto_detect_text:
                text_objects = self._detect_text(image_cv2)
                if text_objects:
                    text_boxes = torch.tensor([
                        [obj.bbox.x, obj.bbox.y, 
                         obj.bbox.x + obj.bbox.width, 
                         obj.bbox.y + obj.bbox.height] 
                        for obj in text_objects
                    ]).to(self.device)

                    transformed_boxes = self.predictor.transform.apply_boxes_torch(
                        text_boxes, image_cv2.shape[:2]
                    ).to(self.device)

                    text_masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                    masks.extend([m.cpu() for m in text_masks])
                    boxes.extend([b.cpu() for b in text_boxes])
                    objects.extend(text_objects)

            # Generate response
            if not objects:
                return ProcessingResponse(
                    status=ProcessingStatus.ERROR,
                    message="No objects or text detected",
                    processing_time=time.time() - start_time
                )

            # Create visualizations
            vis_output = save_visualization(image_cv2, masks, boxes, [obj.object for obj in objects])
            masked_output = save_masked_output(image_cv2, masks, boxes, padding=self.config.MASK_PADDING)

            return ProcessingResponse(
                status=ProcessingStatus.SUCCESS,
                message="Image processed successfully",
                visualization=base64.b64encode(vis_output.getvalue()).decode('utf-8'),
                masked_output=base64.b64encode(masked_output.getvalue()).decode('utf-8'),
                objects=objects,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message=str(e),
                processing_time=time.time() - start_time
            )
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
            self._cleanup_resources()

    def _detect_text(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect text in the image using OCR"""
        try:
            ocr_results = self.reader.readtext(image)
            text_objects = []
            
            for result in ocr_results:
                bbox, detected_text, conf = result
                if not detected_text.strip():
                    continue
                    
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                x_max = max(point[0] for point in bbox)
                y_max = max(point[1] for point in bbox)
                
                bbox_obj = BoundingBox(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                )

                text_objects.append(DetectedObject(
                    object="text",
                    bbox=bbox_obj,
                    confidence=float(conf),
                    detected_text=detected_text,
                    text_alignment=determine_text_alignment(bbox_obj),
                    line_count=1
                ))

            return group_text_objects(text_objects)
        except Exception as e:
            logging.error(f"Error in text detection: {str(e)}")
            return []
        
# FastAPI application setup with improved error handling and resource management
app = FastAPI(
    title="Image Processing API", 
    version="2.0.0",
    description="API for image processing with object detection and text recognition"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TimeoutMiddleware)

# Initialize configuration and processor
config = AppConfig()
processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup"""
    global processor
    try:
        processor = ImageProcessor(config)
        logging.info("Image processor initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize image processor: {str(e)}")
        raise RuntimeError("Failed to initialize image processor") from e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global processor
    if processor:
        processor._cleanup_resources()
    logging.info("Cleanup completed")

def validate_file_size(content_length: int):
    """Validate file size before processing"""
    if content_length > config.MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {config.MAX_CONTENT_LENGTH/1024/1024}MB"
        )

def validate_file_type(filename: str):
    """Validate file extension"""
    file_extension = filename.split(".")[-1].lower() if filename else ""
    if not file_extension or file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

@app.post("/api/v2/process_image", response_model=ProcessingResponse)
@async_timeout(300)  # 5 minutes timeout
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    auto_detect_text: bool = Form(False)
) -> ProcessingResponse:
    """
    Process an image with object detection and optional text recognition
    
    Parameters:
    - file: Image file to process
    - prompt: Text prompt for object detection
    - auto_detect_text: Whether to perform OCR on the image
    
    Returns:
    - ProcessingResponse object containing detection results and visualizations
    """
    try:
        # Validate request
        validate_file_type(file.filename)
        content = await file.read()
        validate_file_size(len(content))
        
        # Process image with resource management
        async with managed_resource():
            result = processor.process_image(content, prompt, auto_detect_text)
            
            if result.status == ProcessingStatus.ERROR:
                raise HTTPException(status_code=500, detail=result.message)
                
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during image processing")
    finally:
        await file.close()

@app.get("/api/v2/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "version": app.version,
        "device": str(processor.device if processor else "not initialized")
    }

@app.post("/api/v2/cleanup")
async def force_cleanup():
    """Force cleanup of system resources"""
    try:
        if processor:
            processor._cleanup_resources()
        gc.collect()
        return {"status": "success", "message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with improved settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=10,  # Limit concurrent requests
        timeout_notify=30,
        timeout_graceful_shutdown=30,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
            },
        },
    )
    
    # Add signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle_shutdown)
    
    # Start server
    server = uvicorn.Server(config)
    server.run()