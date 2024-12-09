from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ImageFormat(str, Enum):
    """Supported image format extensions"""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"

class ProcessingStatus(str, Enum):
    """Processing status indicators"""
    SUCCESS = "success"
    ERROR = "error"

class TextAlignment(str, Enum):
    """Text alignment options"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class BoundingBox(BaseModel):
    """Bounding box coordinates and dimensions"""
    x: float = Field(..., description="X-coordinate of top-left corner")
    y: float = Field(..., description="Y-coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")

class DetectedObject(BaseModel):
    """Detected object information including position and attributes"""
    object: str = Field(..., description="Detected object phrase")
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_text: str = Field(default="")
    text_alignment: Optional[TextAlignment] = Field(default=None)
    line_count: Optional[int] = Field(default=None)

class ProcessingResponse(BaseModel):
    """API response model for image processing"""
    status: ProcessingStatus
    message: str
    visualization: Optional[str] = None
    masked_output: Optional[str] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    processing_time: float = Field(default=0.0)