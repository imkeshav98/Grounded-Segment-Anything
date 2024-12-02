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
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Models and Config
class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"

class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

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

class ProcessingResponse(BaseModel):
    status: ProcessingStatus
    message: str
    visualization: Optional[str] = None
    masked_output: Optional[str] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)

# Image Processing Class
class ImageProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
        self._setup_logging()

    def _setup_logging(self):
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
        """Initialize ML models"""
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

    def process_image(self, image_content: bytes, prompt: str) -> ProcessingResponse:
        """Process single image"""
        start_time = time.time()
        try:
            # Save image temporarily
            image_path = "temp_image.jpg"
            with open(image_path, 'wb') as f:
                f.write(image_content)

            # Process image
            image_pil, image_tensor = load_image(image_path)
            boxes_filt, pred_phrases, logits_filt = get_grounded_output(
                self.model, image_tensor, prompt,
                self.config.BOX_THRESHOLD,
                self.config.TEXT_THRESHOLD,
                self.config.IOU_THRESHOLD,
                device=self.device
            )

            if len(boxes_filt) == 0:
                return ProcessingResponse(
                    status=ProcessingStatus.ERROR,
                    message="No objects detected",
                    processing_time=time.time() - start_time
                )

            # Process detections
            image_cv2 = cv2.imread(image_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_cv2)

            size = image_pil.size
            H, W = size[1], size[0]
            boxes_filt = process_boxes(boxes_filt, W, H)

            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                boxes_filt, image_cv2.shape[:2]
            ).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Generate outputs
            vis_output = save_visualization(image_cv2, masks, boxes_filt, pred_phrases)
            masked_output = save_masked_output(
                image_cv2, masks, boxes_filt,
                padding=self.config.MASK_PADDING
            )

            # Process detected objects
            objects = []
            for box, phrase, logit in zip(boxes_filt, pred_phrases, logits_filt):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                roi = image_cv2[y1:y2, x1:x2]
                
                detected_text = ''
                if roi.size > 0:
                    try:
                        text_results = self.reader.readtext(roi)
                        detected_text = ' '.join([text[1] for text in text_results]) if text_results else ''
                    except Exception as e:
                        logging.error(f"Error in OCR: {str(e)}")

                objects.append(DetectedObject(
                    object=phrase,
                    bbox=BoundingBox(
                        x=float(box[0]),
                        y=float(box[1]),
                        width=float(box[2] - box[0]),
                        height=float(box[3] - box[1])
                    ),
                    confidence=float(logit.max()),
                    detected_text=detected_text
                ))

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

    # Modify the process_image_with_text_search method in ImageProcessor class
    def process_image_with_text_search(
        self, 
        image_content: bytes,
        distance_threshold: float = 50
    ) -> ProcessingResponse:
        """Process image to detect all text regions"""
        start_time = time.time()
        try:
            # Save image temporarily
            image_path = "temp_image.jpg"
            with open(image_path, 'wb') as f:
                f.write(image_content)

            # Load and process image
            image_cv2 = cv2.imread(image_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            
            # Perform OCR on the entire image
            ocr_results = self.reader.readtext(image_cv2)
            
            # Create objects for all detected text
            objects = []
            for result in ocr_results:
                bbox, detected_text, conf = result
                if not detected_text.strip():
                    continue
                    
                # Convert OCR bbox format to our format
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                x_max = max(point[0] for point in bbox)
                y_max = max(point[1] for point in bbox)
                
                objects.append(DetectedObject(
                    object="text",  # Generic label for text regions
                    bbox=BoundingBox(
                        x=float(x_min),
                        y=float(y_min),
                        width=float(x_max - x_min),
                        height=float(y_max - y_min)
                    ),
                    confidence=float(conf),
                    detected_text=detected_text
                ))

            # Group nearby text regions
            objects = group_text_objects(objects, distance_threshold)
            
            if not objects:
                return ProcessingResponse(
                    status=ProcessingStatus.ERROR,
                    message="No text found in image",
                    processing_time=time.time() - start_time
                )

            # Initialize SAM predictor
            self.predictor.set_image(image_cv2)
            
            # Generate boxes for SAM
            boxes = torch.tensor([
                [obj.bbox.x, obj.bbox.y, 
                obj.bbox.x + obj.bbox.width, 
                obj.bbox.y + obj.bbox.height] 
                for obj in objects
            ]).to(self.device)

            # Get SAM predictions
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                boxes, image_cv2.shape[:2]
            ).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Move tensors to CPU before visualization
            masks = masks.cpu()
            boxes = boxes.cpu()

            # Generate visualization and masked output
            vis_output = save_visualization(
                image_cv2, 
                masks, 
                boxes,
                [f"{obj.detected_text} ({obj.confidence:.2f})" for obj in objects]
            )
            
            masked_output = save_masked_output(
                image_cv2, 
                masks, 
                boxes,
                padding=self.config.MASK_PADDING
            )

            return ProcessingResponse(
                status=ProcessingStatus.SUCCESS,
                message="Text detection completed successfully",
                visualization=base64.b64encode(vis_output.getvalue()).decode('utf-8'),
                masked_output=base64.b64encode(masked_output.getvalue()).decode('utf-8'),
                objects=objects,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logging.error(f"Error detecting text: {str(e)}")
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message=str(e),
                processing_time=time.time() - start_time
            )
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

# Utility functions
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounded_output(model, image, caption, box_threshold, text_threshold, iou_threshold=0.5, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
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

    # Apply NMS
    if len(boxes_filt) > 0:
        scores_tensor = torch.tensor(scores)
        nms_idx = torchvision.ops.nms(boxes_filt, scores_tensor, iou_threshold).numpy().tolist()
        
        # Filter boxes, phrases, and scores based on NMS
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        logits_filt = logits_filt[nms_idx]
        
        logging.info(f"NMS applied: Reduced from {len(scores)} to {len(nms_idx)} boxes")
    
    return boxes_filt, pred_phrases, logits_filt

def process_boxes(boxes_filt, W, H):
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt.cpu()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_visualization(image, masks, boxes, phrases):
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

def save_masked_output(image, masks, boxes, padding=5):
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
    # Calculate centers
    center1_x = box1.x + box1.width / 2
    center1_y = box1.y + box1.height / 2
    center2_x = box2.x + box2.width / 2
    center2_y = box2.y + box2.height / 2

    # Calculate horizontal and vertical distances
    horizontal_distance = abs(center1_x - center2_x)
    vertical_distance = abs(center1_y - center2_y)

    # Check for similar heights (to differentiate titles from descriptions)
    similar_height = abs(box1.height - box2.height) < min(box1.height, box2.height) * 0.5

    # More strict vertical spacing check
    max_vertical_spacing = min(box1.height, box2.height) * 1.2

    # For horizontal alignment (same line)
    y_aligned = vertical_distance < min(box1.height, box2.height) * 0.5
    
    # For vertical stacking (different lines)
    x_overlap = (
        min(box1.x + box1.width, box2.x + box2.width) >
        max(box1.x, box2.x)
    )
    
    proper_spacing = vertical_distance < max_vertical_spacing

    # Group boxes if they're on the same line or properly spaced description lines
    # But only if they have similar heights (to avoid grouping titles with descriptions)
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

    # Create groups of indices
    groups = []
    used_indices = set()

    for i, obj1 in enumerate(objects):
        if i in used_indices:
            continue

        current_group = {i}
        used_indices.add(i)

        # Keep checking for nearby boxes until no more are found
        changed = True
        while changed:
            changed = False
            for j, obj2 in enumerate(objects):
                if j in used_indices:
                    continue

                # Check if obj2 is near any object in the current group
                for idx in current_group:
                    if are_boxes_nearby(objects[idx].bbox, obj2.bbox, distance_threshold):
                        current_group.add(j)
                        used_indices.add(j)
                        changed = True
                        break

        groups.append(current_group)

    # Merge objects in each group
    merged_objects = []
    for group in groups:
        group_objects = [objects[i] for i in group]
        
        # Merge bounding boxes
        merged_bbox = merge_boxes([obj.bbox for obj in group_objects])
        
        # Combine text and sort by y-coordinate for natural reading order
        sorted_objects = sorted(group_objects, key=lambda obj: obj.bbox.y)
        merged_text = ' '.join(obj.detected_text for obj in sorted_objects)
        
        # Average confidence scores
        avg_confidence = sum(obj.confidence for obj in group_objects) / len(group_objects)
        
        merged_objects.append(DetectedObject(
            object="text",
            bbox=merged_bbox,
            confidence=avg_confidence,
            detected_text=merged_text
        ))

    return merged_objects

# FastAPI application setup
app = FastAPI(title="Image Processing API", version="2.0.0")
config = AppConfig()
processor = ImageProcessor(config)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v2/process_image")
async def process_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
) -> ProcessingResponse:
    """Process image endpoint with validation"""
    # Validate file type
    file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
    if not file_extension or file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read and validate file content
    try:
        content = await file.read()
        if len(content) > config.MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=400, detail="File too large")

        return processor.process_image(content, prompt)
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing upload")
    finally:
        await file.close()

@app.post("/api/v2/search_text")
async def search_text(
    file: UploadFile = File(...)
) -> ProcessingResponse:
    """Detect and segment all text regions in the image"""
    # Validate file type
    file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
    if not file_extension or file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Read and validate file content
        content = await file.read()
        if len(content) > config.MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=400, detail="File too large")

        return processor.process_image_with_text_search(content)
    except Exception as e:
        logging.error(f"Error processing text detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing text detection")
    finally:
        await file.close()

@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": app.version,
        "device": str(processor.device)
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Startup logging messages
    logging.info("Starting Grounded-SAM API Server...")
    logging.info(f"Running on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Start server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )