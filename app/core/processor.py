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
import easyocr
import time
import gc

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor

from app.config import AppConfig
from app.models.schemas import ProcessingResponse, ProcessingStatus, DetectedObject, BoundingBox
from app.utils.helpers import determine_text_alignment, group_text_objects

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

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounded_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
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
        for logit in logits_filt:
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases, logits_filt

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0-5, label, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

class ImageProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        self.model = load_model(
            str(self.config.CONFIG_FILE),
            str(self.config.GROUNDED_CHECKPOINT),
            self.device
        )
        self.predictor = SamPredictor(
            build_sam(checkpoint=str(self.config.SAM_CHECKPOINT)).to(self.device)
        )
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def cleanup_resources(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        plt.close('all')

    def process_image(self, image_content: bytes, prompt: str, auto_detect_text: bool = False) -> ProcessingResponse:
        start_time = time.time()
        temp_path = "temp_image.jpg"
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(image_content)

            image_pil, image_tensor = load_image(temp_path)
            
            boxes_filt, pred_phrases, logits_filt = get_grounded_output(
                self.model, image_tensor, prompt,
                self.config.BOX_THRESHOLD,
                self.config.TEXT_THRESHOLD,
                device=self.device
            )

            image_cv2 = cv2.imread(temp_path)
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            
            if len(boxes_filt) == 0:
                return ProcessingResponse(
                    status=ProcessingStatus.ERROR,
                    message="No objects detected",
                    processing_time=time.time() - start_time
                )

            # Process boxes
            W, H = image_pil.size
            boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])
            
            objects = []
            for i, (box, phrase, logit) in enumerate(zip(boxes_filt, pred_phrases, logits_filt)):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                roi = image_cv2[y1:y2, x1:x2]
                
                detected_text = ''
                if roi.size > 0 and auto_detect_text:
                    try:
                        text_results = self.reader.readtext(roi)
                        detected_text = ' '.join([text[1] for text in text_results]) if text_results else ''
                    except Exception:
                        detected_text = ''

                bbox = BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1)
                )

                objects.append(DetectedObject(
                    object_id=i + 1,
                    object=phrase,
                    bbox=bbox,
                    confidence=float(logit.max()),
                    detected_text=detected_text,
                    text_alignment=determine_text_alignment(bbox) if detected_text else None,
                    line_count=1 if detected_text else None
                ))

            # Create visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(image_cv2)
            for box, obj in zip(boxes_filt, objects):
                show_box(box.numpy(), plt.gca(), f"{obj.object} ({obj.confidence:.2f})")
            plt.axis('off')
            
            vis_buf = io.BytesIO()
            plt.savefig(vis_buf, format='PNG', bbox_inches='tight', dpi=150)
            plt.close()
            vis_buf.seek(0)

            return ProcessingResponse(
                status=ProcessingStatus.SUCCESS,
                message="Image processed successfully",
                visualization=base64.b64encode(vis_buf.getvalue()).decode('utf-8'),
                objects=objects,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ProcessingResponse(
                status=ProcessingStatus.ERROR,
                message=str(e),
                processing_time=time.time() - start_time
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.cleanup_resources()
            
            