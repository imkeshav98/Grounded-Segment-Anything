import asyncio
import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import base64
from PIL import Image
import torchvision
from functools import wraps
from contextlib import asynccontextmanager
from fastapi import HTTPException
from typing import List, Tuple

from ..config import config
from ..api.models import BoundingBox, TextAlignment

def async_timeout(seconds):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request timeout")
        return wrapper
    return decorator

@asynccontextmanager
async def managed_resource():
    """Context manager for resource cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        plt.close('all')

def validate_file_size(content_length: int):
    """Validate uploaded file size"""
    if content_length > config.MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {config.MAX_CONTENT_LENGTH/1024/1024}MB"
        )

def validate_file_type(filename: str):
    """Validate uploaded file type"""
    file_extension = filename.split(".")[-1].lower() if filename else ""
    if not file_extension or file_extension not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

def determine_text_alignment(bbox: BoundingBox, boxes_in_group: List[BoundingBox] = None) -> TextAlignment:
    """Determine text alignment based on bounding box positions"""
    if not boxes_in_group:
        boxes_in_group = [bbox]
        return TextAlignment.LEFT

    left_positions = [box.x for box in boxes_in_group]
    right_positions = [box.x + box.width for box in boxes_in_group]
    
    left_variance = max(left_positions) - min(left_positions)
    right_variance = max(right_positions) - min(right_positions)
    
    threshold = 10
    
    if left_variance <= threshold:
        return TextAlignment.LEFT
    if right_variance <= threshold:
        return TextAlignment.RIGHT
    return TextAlignment.CENTER

def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union for two bounding boxes"""
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def are_boxes_nearby(box1: BoundingBox, box2: BoundingBox) -> bool:
    """Check if two bounding boxes are nearby"""
    height_ratio = max(box1.height, box2.height) / min(box1.height, box2.height)
    if height_ratio >= 1.2:
        return False
    
    vertical_gap = abs(box2.y - (box1.y + box1.height))
    min_height = min(box1.height, box2.height)
    if vertical_gap >= (min_height * 0.5):
        return False
    
    box1_center = box1.x + (box1.width / 2)
    box2_center = box2.x + (box2.width / 2)
    horizontal_offset = abs(box1_center - box2_center)
    max_width = max(box1.width, box2.width)
    
    return horizontal_offset < (max_width * 0.8)

def merge_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    """Merge multiple bounding boxes into one"""
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

def load_image(image_content: bytes) -> Tuple[Image.Image, torch.Tensor]:
    """Load and preprocess image"""
    import GroundingDINO.groundingdino.datasets.transforms as T
    
    # Save temporary image to buffer
    image_buffer = io.BytesIO(image_content)
    image_pil = Image.open(image_buffer).convert("RGB")
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def save_visualization(image: np.ndarray, masks: List[torch.Tensor], boxes: List[torch.Tensor], phrases: List[str]) -> io.BytesIO:
    """Generate visualization of detection results"""
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

def show_mask(mask: np.ndarray, ax, random_color: bool = False):
    """Display segmentation mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box: np.ndarray, ax, label: str):
    """Display bounding box with label"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_masked_output(image: np.ndarray, masks: List[torch.Tensor], boxes: List[torch.Tensor], padding: int = 5) -> io.BytesIO:
    """Generate masked output image"""
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