# File: app/core/model_manager.py

import torch
import easyocr
import threading
from segment_anything import build_sam, SamPredictor
from app.config import AppConfig
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
import gc
import matplotlib.pyplot as plt

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
                    try:
                        self._initialize_models()
                        ModelManager._initialized = True
                        print("ModelManager initialized successfully")
                    except Exception as e:
                        print(f"Error initializing ModelManager: {str(e)}")
                        raise

    def _initialize_models(self):
        """Initialize all models"""
        try:
            # Initialize GroundingDINO model
            self.grounding_model = self._load_grounding_dino()
            print("GroundingDINO model loaded")
            
            # Initialize SAM predictor
            self.sam_predictor = self._load_sam_predictor()
            print("SAM predictor loaded")
            
            # Initialize OCR reader
            self.reader = self._initialize_ocr()
            print("OCR reader initialized")
            
        except Exception as e:
            self.cleanup()  # Clean up any partially initialized models
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def _load_grounding_dino(self):
        """Load GroundingDINO model"""
        args = SLConfig.fromfile(str(self.config.CONFIG_FILE))
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(str(self.config.GROUNDED_CHECKPOINT), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)

    def _load_sam_predictor(self):
        """Load SAM predictor"""
        sam = build_sam(checkpoint=str(self.config.SAM_CHECKPOINT)).to(self.device)
        return SamPredictor(sam)

    def _initialize_ocr(self):
        """Initialize OCR reader"""
        return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'grounding_model'):
                self.grounding_model.cpu()
                del self.grounding_model
            
            if hasattr(self, 'sam_predictor'):
                if hasattr(self.sam_predictor, 'model'):
                    self.sam_predictor.model.cpu()
                del self.sam_predictor
            
            if hasattr(self, 'reader'):
                del self.reader
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            plt.close('all')
            
            print("ModelManager cleanup completed")
            
        except Exception as e:
            print(f"Error during ModelManager cleanup: {str(e)}")

    @property
    def is_initialized(self):
        """Check if the ModelManager is fully initialized"""
        return self._initialized

    @property
    def models(self):
        """Return all initialized models"""
        if not self._initialized:
            raise RuntimeError("ModelManager not initialized")
        
        return {
            'grounding_model': self.grounding_model,
            'sam_predictor': self.sam_predictor,
            'reader': self.reader,
            'device': self.device
        }