# app/config.py
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set

@dataclass
class AppConfig:
    """Application configuration with type hints"""
    CONFIG_FILE: Path = Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDED_CHECKPOINT: Path = Path("groundingdino_swint_ogc.pth")
    SAM_CHECKPOINT: Path = Path("sam_vit_h_4b8939.pth")
    LOG_DIR: Path = Path("logs")
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {"png", "jpeg", "jpg"})
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    BOX_THRESHOLD: float = 0.3
    TEXT_THRESHOLD: float = 0.25
    MASK_PADDING: int = 5
    IOU_THRESHOLD: float = 0.5

    def __post_init__(self):
        """Ensure log directory exists"""
        os.makedirs(self.LOG_DIR, exist_ok=True)

# Create and export config instance
config = AppConfig()

# Make sure config is available for import
__all__ = ['AppConfig', 'config']