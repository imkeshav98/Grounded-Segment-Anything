# File: app/utils/helpers.py

from contextlib import asynccontextmanager
import torch
import matplotlib.pyplot as plt
import gc
from typing import List, Dict
from app.models.schemas import BoundingBox, TextAlignment, DetectedObject

@asynccontextmanager
async def managed_resource():
    """Context manager for cleaning up resources after processing"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        plt.close('all')

def determine_text_alignment(bbox: BoundingBox, boxes_in_group: List[BoundingBox] = None) -> TextAlignment:
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

def group_text_objects(objects: List[DetectedObject]) -> List[DetectedObject]:
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

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) != find(j) and are_boxes_nearby(objects[i].bbox, objects[j].bbox):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    merged_objects = []
    for group_indices in groups.values():
        group_objects = [objects[i] for i in group_indices]
        
        y_centers = [(obj.bbox.y + obj.bbox.height/2, i) for i, obj in enumerate(group_objects)]
        sorted_indices = [i for _, i in sorted(y_centers)]
        sorted_objects = [group_objects[i] for i in sorted_indices]

        min_object_id = min(obj.object_id for obj in sorted_objects)
        group_boxes = [obj.bbox for obj in sorted_objects]
        merged_bbox = merge_boxes(group_boxes)
        merged_text = '\n'.join(obj.detected_text for obj in sorted_objects)
        avg_confidence = sum(obj.confidence for obj in sorted_objects) / len(sorted_objects)

        merged_objects.append(DetectedObject(
            object_id=min_object_id,
            object="text",
            bbox=merged_bbox,
            confidence=avg_confidence,
            detected_text=merged_text,
            text_alignment=determine_text_alignment(merged_bbox, group_boxes),
            line_count=len(sorted_objects)
        ))

    return merged_objects

def is_mostly_contained(inner: BoundingBox, outer: BoundingBox, tolerance: float = 0.2) -> bool:
    """
    Check if one bbox is mostly contained within another, allowing for some overflow
    """
    # Calculate overlap area
    x_overlap = max(0, min(inner.x + inner.width, outer.x + outer.width) - max(inner.x, outer.x))
    y_overlap = max(0, min(inner.y + inner.height, outer.y + outer.height) - max(inner.y, outer.y))
    overlap_area = x_overlap * y_overlap
    
    # Calculate inner box area
    inner_area = inner.width * inner.height
    
    # If overlap is at least (1-tolerance)% of inner box area, consider it contained
    return overlap_area >= inner_area * (1 - tolerance)

def calculate_zindexes(objects: List[DetectedObject]) -> List[DetectedObject]:
    """
    Calculate z-index values for all objects based on containment relationships.
    Each level of containment increases z-index by 1, starting from 1.
    """
    if not objects:
        return objects

    n = len(objects)
    # Initialize all objects with z-index 1
    zindex_map: Dict[int, int] = {obj.object_id: 1 for obj in objects}
    
    # Build containment relationships
    contained_by: Dict[int, set] = {obj.object_id: set() for obj in objects}
    
    # Check each pair of objects for containment
    for i in range(n):
        for j in range(n):
            if i != j:
                obj1, obj2 = objects[i], objects[j]
                if is_mostly_contained(obj1.bbox, obj2.bbox):
                    # obj1 is contained by obj2
                    contained_by[obj1.object_id].add(obj2.object_id)

    # Calculate z-indexes based on containment levels
    changed = True
    while changed:
        changed = False
        for obj_id in zindex_map:
            # If this object is contained by any other object,
            # its z-index should be one more than the maximum z-index
            # of its containers
            containers = contained_by[obj_id]
            if containers:
                max_container_zindex = max(zindex_map[container_id] for container_id in containers)
                new_zindex = max_container_zindex + 1
                if new_zindex != zindex_map[obj_id]:
                    zindex_map[obj_id] = new_zindex
                    changed = True

    # Update objects with calculated z-indexes
    for obj in objects:
        obj.z_index = zindex_map[obj.object_id]
    
    return objects