# File: app/core/model_manager.py

import torch
import easyocr
from pathlib import Path
import threading
from segment_anything import build_sam, SamPredictor
from app.config import AppConfig
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.config = AppConfig()
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self._initialize_models()
                    ModelManager._initialized = True

    def _initialize_models(self):
        """Initialize all models with proper error handling"""
        try:
            # Initialize GroundingDINO model
            self.grounding_model = self._load_grounding_dino()
            
            # Initialize SAM predictor
            self.sam_predictor = self._load_sam_predictor()
            
            # Initialize OCR reader
            self.reader = self._initialize_ocr()
            
            print("All models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def _load_grounding_dino(self):
        """Load GroundingDINO model"""
        try:
            args = SLConfig.fromfile(str(self.config.CONFIG_FILE))
            args.device = self.device
            model = build_model(args)
            checkpoint = torch.load(str(self.config.GROUNDED_CHECKPOINT), map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            model.eval()
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load GroundingDINO model: {str(e)}")

    def _load_sam_predictor(self):
        """Load SAM predictor"""
        try:
            sam = build_sam(checkpoint=str(self.config.SAM_CHECKPOINT)).to(self.device)
            return SamPredictor(sam)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM predictor: {str(e)}")

    def _initialize_ocr(self):
        """Initialize OCR reader"""
        try:
            return easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OCR reader: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'grounding_model'):
            self.grounding_model.cpu()
        if hasattr(self, 'sam_predictor') and hasattr(self.sam_predictor, 'model'):
            self.sam_predictor.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def models(self):
        """Return all initialized models"""
        return {
            'grounding_model': self.grounding_model,
            'sam_predictor': self.sam_predictor,
            'reader': self.reader,
            'device': self.device
        }

# Global instance
model_manager = ModelManager()