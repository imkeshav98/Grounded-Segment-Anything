# File: app/models/schemas.py

from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"

class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class LayerType(str, Enum):
    BUTTON = "button"
    TEXT = "text"
    IMAGE = "image"

class FontStyle(str, Enum):
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"

class FontWeight(str, Enum):
    LIGHT = "300"
    REGULAR = "400"
    MEDIUM = "500"
    SEMIBOLD = "600"
    BOLD = "700"
    EXTRABOLD = "800"

class TextAlignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class BoundingBox(BaseModel):
    x: float = Field(..., description="X-coordinate of top-left corner")
    y: float = Field(..., description="Y-coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")

class StyleProperties(BaseModel):
    fontFamily: Optional[str] = Field(None, description="Google Font name")
    fontSize: Optional[float] = None
    fontWeight: Optional[FontWeight] = None
    fontStyles: Optional[List[FontStyle]] = []
    color: Optional[str] = None
    backgroundColor: Optional[str] = None
    borderRadius: Optional[float] = None

class DetectedObject(BaseModel):
    object_id: int = Field(..., description="Unique identifier for the detected object")
    object: str = Field(..., description="Detected object type or label")
    layer_type: Optional[LayerType] = Field(..., description="Type of layer (button, text, image)")
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_text: Optional[str] = ""
    styles: Optional[StyleProperties] = None
    text_alignment: Optional[TextAlignment] = Field(default=None)
    line_count: Optional[int] = Field(default=None)

class ThemeProperties(BaseModel):
    primaryColor: str = Field(..., description="Primary theme color in hex")
    secondaryColor: str = Field(..., description="Secondary theme color in hex")
    backgroundColor: str = Field(..., description="Background color in hex")
    fontStyles: Dict[str, str] = Field(..., description="Theme font styles")

class ProcessingResponse(BaseModel):
    status: ProcessingStatus
    message: str
    visualization: Optional[str] = None
    masked_output: Optional[str] = None
    objects: List[DetectedObject] = Field(default_factory=list)
    theme: Optional[ThemeProperties] = None
    processing_time: float = Field(default=0.0)