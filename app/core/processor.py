# File: app/core/processor.py

import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import warnings
import torchvision
import time
import gc
import uuid
from typing import List

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap

from app.config import AppConfig
from app.models.schemas import ProcessingResponse, ProcessingStatus, DetectedObject, BoundingBox, LayerType
from app.utils.helpers import determine_text_alignment, group_text_objects, calculate_zindexes
from app.utils.firebase import firebase
from app.core.model_manager import model_manager

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def process_boxes(boxes_filt, W, H):
    """Process and scale bounding boxes"""
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt.cpu()

def get_grounded_output(model, image, caption, box_threshold, text_threshold, iou_threshold=0.5, device="cpu"):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def show_box(box, ax, label, object_id):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0-5, f"{label} (ID: {object_id})", fontsize=7, 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

def save_visualization(image, boxes, objects):
    try:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        for box, obj in zip(boxes, objects):
            show_box(
                box.numpy() if isinstance(box, torch.Tensor) else box,
                plt.gca(),
                obj.object,
                obj.object_id
            )
        
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches="tight", dpi=150)
        plt.close()
        buf.seek(0)
        return buf
    finally:
        plt.close('all')

def save_visualization_with_segmentation(image, boxes, masks, objects, folder_id):
    """Save visualization with both bounding boxes and segmentation masks"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    if len(masks) == len(objects):
        # Generate distinct colors for each mask
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
        
        # Plot segmentation masks first
        for mask, color in zip(masks, colors):
            mask_np = mask.cpu().numpy()[0]  # Get mask data
            # Show mask with semi-transparency
            colored_mask = np.zeros((*mask_np.shape, 4))
            colored_mask[mask_np] = (*color[:3], 0.3)  # RGB + alpha
            plt.imshow(colored_mask)
    
        # Then plot bounding boxes and labels
        for box, obj, color in zip(boxes, objects, colors):
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            # Use same color as mask for consistency
            plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, 
                                            edgecolor=color, 
                                            facecolor=(0,0,0,0), 
                                            lw=2))
            plt.gca().text(x0, y0-5, f"{obj.object} (ID: {obj.object_id})", 
                          fontsize=7,
                          color='black',
                          bbox=dict(facecolor='white', 
                                  alpha=0.7, 
                                  edgecolor='none',
                                  pad=1))
    
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    
    # Upload visualization to Firebase Storage
    vis_url = firebase.upload_image(buf.getvalue(), folder_id, f"visualization.png")

    return vis_url

def save_masked_output(image, masks, boxes, folder_id, padding=5):
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

    # Upload masked image to Firebase Storage
    masked_url = firebase.upload_image(buf.getvalue(), folder_id, f"masked.png")

    return masked_url

def save_individual_mask(image, mask, folder_id, object_id, bbox, padding=5):
    """
    Save individual mask for an object cropped to its bounding box dimensions
    
    Args:
        image: Original image
        mask: Object mask
        folder_id: Storage folder ID
        object_id: Object identifier
        bbox: List/array of [x1, y1, x2, y2] coordinates
        padding: Padding around the segmentation mask
    """
    mask_np = mask.cpu().numpy()[0]
    
    # Get bbox coordinates
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # First crop to bbox
    height, width = image.shape[:2]
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask_np[y1:y2, x1:x2]
    
    # Create transparent mask of cropped size
    h, w = cropped_image.shape[:2]
    transparent_mask = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Apply padding to the mask through dilation
    kernel = np.ones((padding*2, padding*2), np.uint8)
    dilated_mask = cv2.dilate(cropped_mask.astype(np.uint8), kernel, iterations=1)
    
    # Get mask indices where True
    mask_indices = np.where(dilated_mask)
    
    # Copy RGB values for masked pixels
    for i in range(3):
        transparent_mask[mask_indices[0], mask_indices[1], i] = cropped_image[mask_indices[0], mask_indices[1], i]
    
    # Set alpha channel for masked pixels
    transparent_mask[mask_indices[0], mask_indices[1], 3] = 255
    
    # Convert to PIL Image with high quality
    masked_image = Image.fromarray(transparent_mask)
    
    # Save with high quality settings
    buf = io.BytesIO()
    masked_image.save(buf, format='PNG', optimize=False, quality=100)
    buf.seek(0)
    
    # Upload mask to Firebase Storage
    mask_url = firebase.upload_image(
        buf.getvalue(),
        folder_id,
        f"mask_{object_id}.png"
    )

    return mask_url

class ImageProcessor:
    def __init__(self, config: AppConfig):
        """Initialize the ImageProcessor with configuration"""
        try:
            self.config = config
            self._initialize_from_manager()
            self._reset_instance_state()
        except Exception as e:
            print(f"Error initializing ImageProcessor: {str(e)}")
            raise

    def _initialize_from_manager(self):
        """Initialize processor with shared models from ModelManager"""
        try:
            models = model_manager.models
            self.model = models['grounding_model']
            self.predictor = models['sam_predictor']
            self.reader = models['reader']
            self.device = models['device']
        except Exception as e:
            raise RuntimeError(f"Failed to initialize from ModelManager: {str(e)}")

    def _reset_instance_state(self):
        """Reset instance-specific state variables"""
        self.object_id_counter = 1
        self.folder_id = str(uuid.uuid4())

    def cleanup_resources(self):
        """Cleanup instance-specific resources"""
        try:
            gc.collect()
            plt.close('all')
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def _detect_text(self, image: np.ndarray) -> List[DetectedObject]:
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
                    object_id=self.object_id_counter,  # Use class counter
                    object="text",
                    bbox=bbox_obj,
                    confidence=float(conf),
                    detected_text=detected_text,
                    text_alignment=determine_text_alignment(bbox_obj),
                    line_count=1
                ))
                self.object_id_counter += 1  # Increment class counter

            return group_text_objects(text_objects)
        except Exception:
            return []

    def process_image(self, image_content: bytes, prompt: str, auto_detect_text: bool = False) -> ProcessingResponse:
        """Process an image"""
        start_time = time.time()
        temp_path = f"temp_image_{uuid.uuid4()}.jpg"  # Make temp file unique
        
        # Reset state for new processing
        self._reset_instance_state()
        
        try:
            # Ensure prompt is string
            if not isinstance(prompt, str):
                prompt = str(prompt)

            # Save bytes to temporary file
            with open(temp_path, 'wb') as f:
                f.write(image_content)

            # Load and process image
            image_pil, image_tensor = load_image(temp_path)
            
            boxes_filt, pred_phrases, logits_filt = get_grounded_output(
                self.model, image_tensor, prompt,
                self.config.BOX_THRESHOLD,
                self.config.TEXT_THRESHOLD,
                self.config.IOU_THRESHOLD,
                device=self.device
            )

            image_cv2 = cv2.imread(temp_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
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

                for box, phrase, logit in zip(boxes_filt, pred_phrases, logits_filt):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    roi = image_cv2[y1:y2, x1:x2]
                    
                    detected_text = ''
                    if roi.size > 0 and auto_detect_text:
                        try:
                            text_results = self.reader.readtext(roi)
                            detected_text = ' '.join([text[1] for text in text_results]) if text_results else ''
                        except Exception:
                            detected_text = ''

                    bbox_obj = BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1)
                    )

                    objects.append(DetectedObject(
                        object_id=self.object_id_counter,  # Use class counter
                        object=phrase,
                        bbox=bbox_obj,
                        confidence=float(logit.max()),
                        detected_text=detected_text,
                        text_alignment=determine_text_alignment(bbox_obj) if detected_text else None,
                        line_count=1
                    ))
                    self.object_id_counter += 1  # Increment class counter

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

            if not objects:
                return ProcessingResponse(
                    status=ProcessingStatus.ERROR,
                    message="No objects or text detected",
                    processing_time=time.time() - start_time
                )

            vis_output = save_visualization(image_cv2, boxes, objects)

            return ProcessingResponse(
                status=ProcessingStatus.SUCCESS,
                message="Image processed successfully",
                visualization=base64.b64encode(vis_output.getvalue()).decode('utf-8'),
                objects=objects,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            print(f"Error in process_image: {str(e)}")

            self.cleanup_resources();
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message=str(e),
                processing_time=time.time() - start_time
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def regenerate_outputs(self, image_content: bytes, validated_objects: List[DetectedObject]) -> ProcessingResponse:
        """Regenerate visualization and masks for validated objects"""
        start_time = time.time()
        temp_path = f"temp_image_{uuid.uuid4()}.jpg"  # Make temp file unique
        
        try:
            # Write image to temp file
            with open(temp_path, 'wb') as f:
                f.write(image_content)

            # Load and prepare image
            image_cv2 = cv2.imread(temp_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_cv2)

            # Calculate z-indexes for all objects
            validated_objects = calculate_zindexes(validated_objects)
            
            # Sort objects by z-index for proper layering
            validated_objects = sorted(validated_objects, key=lambda x: x.z_index)

            # Convert validated objects to boxes (maintain z-index order)
            boxes = []
            for obj in validated_objects:
                box = [
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.x + obj.bbox.width,
                    obj.bbox.y + obj.bbox.height
                ]
                boxes.append(box)

            boxes = torch.tensor(boxes).to(self.device)
            
            # Generate masks
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                boxes, image_cv2.shape[:2]
            ).to(self.device)

            masks_output = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            masks = [m.cpu() for m in masks_output[0]]
            
            # Add individual masks for image objects (following z-index order)
            for i, obj in enumerate(validated_objects):
                if obj.layer_type == LayerType.IMAGE:
                    bbox = [
                        obj.bbox.x,
                        obj.bbox.y,
                        obj.bbox.x + obj.bbox.width,
                        obj.bbox.y + obj.bbox.height
                    ]
                    obj.mask = save_individual_mask(
                        image_cv2,
                        masks[i],
                        self.folder_id,
                        object_id=obj.object_id,
                        bbox=bbox,
                        padding=1,
                    )

            original_image = firebase.upload_image(image_content, self.folder_id, "original.png")

            # Generate outputs with segmentation
            vis_output = save_visualization_with_segmentation(
                image_cv2, 
                boxes.cpu(), 
                masks,
                validated_objects,
                self.folder_id
            )
            
            masked_output = save_masked_output(
                image_cv2, 
                masks, 
                boxes.cpu(), 
                self.folder_id,
                padding=self.config.MASK_PADDING
            )

            return ProcessingResponse(
                status=ProcessingStatus.SUCCESS,
                message="Image processed successfully",
                original_image=original_image,
                visualization=vis_output,
                masked_output=masked_output,
                objects=validated_objects,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            print(f"Error in regenerate_outputs: {str(e)}")
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message=str(e),
                processing_time=time.time() - start_time
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.cleanup_resources()