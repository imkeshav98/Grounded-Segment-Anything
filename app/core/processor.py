import os
import torch
import numpy as np
import logging
import easyocr
from typing import List, Tuple, Optional
import gc
import time
import base64

from GroundingDINO.groundingdino.models import build_model
from segment_anything import build_sam, SamPredictor

from ..config import AppConfig
from ..api.models import (
    DetectedObject,
    BoundingBox,
    ProcessingResponse,
    ProcessingStatus,
    TextAlignment
)
from .utils import (
    load_image,
    save_visualization,
    save_masked_output,
    determine_text_alignment,
    calculate_iou,
    are_boxes_nearby,
    merge_boxes
)
from .model_utils import (
    load_model,
    get_grounded_output,
    process_boxes
)

class ImageProcessor:
    """Main image processing class handling object detection and text recognition"""
    
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
        """Initialize all required models"""
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

            return self._group_text_objects(text_objects)
        except Exception as e:
            logging.error(f"Error in text detection: {str(e)}")
            return []

    def _group_text_objects(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Group related text objects together"""
        if not objects:
            return []

        n = len(objects)
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1

        # Build groups
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j) and are_boxes_nearby(objects[i].bbox, objects[j].bbox):
                    union(i, j)

        # Collect groups
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # Process groups
        merged_objects = []
        for group_indices in groups.values():
            group_objects = [objects[i] for i in group_indices]
            
            y_centers = [(obj.bbox.y + obj.bbox.height/2, i) 
                        for i, obj in enumerate(group_objects)]
            sorted_indices = [i for _, i in sorted(y_centers)]
            sorted_objects = [group_objects[i] for i in sorted_indices]
            
            group_boxes = [obj.bbox for obj in sorted_objects]
            merged_bbox = merge_boxes(group_boxes)
            merged_text = '\n'.join(obj.detected_text for obj in sorted_objects)
            avg_confidence = sum(obj.confidence for obj in sorted_objects) / len(sorted_objects)
            
            y_positions = [y for y, _ in sorted(y_centers)]
            min_height = min(obj.bbox.height for obj in sorted_objects)
            line_threshold = min_height * 0.5
            
            line_count = 1
            prev_y = y_positions[0]
            for y in y_positions[1:]:
                if abs(y - prev_y) > line_threshold:
                    line_count += 1
                    prev_y = y

            merged_objects.append(DetectedObject(
                object="text",
                bbox=merged_bbox,
                confidence=avg_confidence,
                detected_text=merged_text,
                text_alignment=determine_text_alignment(merged_bbox, group_boxes),
                line_count=line_count
            ))

        return merged_objects

    def process_image(self, image_content: bytes, prompt: str, 
                     auto_detect_text: bool = False) -> ProcessingResponse:
        """Process image content with given prompt and options"""
        start_time = time.time()
        
        try:
            # Load and process image
            image_pil, image_tensor = load_image(image_content)
            
            # Get model predictions
            boxes_filt, pred_phrases, logits_filt = get_grounded_output(
                self.model, image_tensor, prompt,
                self.config.BOX_THRESHOLD,
                self.config.TEXT_THRESHOLD,
                self.config.IOU_THRESHOLD,
                device=self.device
            )

            # Process image with OpenCV
            image_cv2 = np.array(image_pil)
            self.predictor.set_image(image_cv2)

            objects = []
            masks = []
            boxes = []

            if len(boxes_filt) > 0:
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
            vis_output = save_visualization(image_cv2, masks, boxes, 
                                         [obj.object for obj in objects])
            masked_output = save_masked_output(image_cv2, masks, boxes, 
                                            padding=self.config.MASK_PADDING)

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
            self._cleanup_resources()