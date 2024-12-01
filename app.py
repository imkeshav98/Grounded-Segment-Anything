import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import logging
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import warnings
import easyocr

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, SamPredictor

# Config class for application settings
class Config:
    CONFIG_FILE = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDED_CHECKPOINT = "groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    OUTPUT_DIR = "outputs"
    LOG_DIR = "logs"
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3
    MASK_PADDING = 5
    HOST = "0.0.0.0"  # Allow external connections
    PORT = 5000

app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
def setup_logging():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(Config.LOG_DIR, 'app.log')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=10
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

# Initialize models at startup
def init_models():
    global model, predictor, reader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Initializing models on {device}")
    
    try:
        model = load_model(Config.CONFIG_FILE, Config.GROUNDED_CHECKPOINT, device)
        predictor = SamPredictor(build_sam(checkpoint=Config.SAM_CHECKPOINT).to(device))
        reader = easyocr.Reader(['en'] , gpu=torch.cuda.is_available())
        logging.info("Models initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        return False

# Decorator for timing API calls
def timer_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logging.info(f"{f.__name__} took {end - start:.2f} seconds to execute")
        return result
    return wrapper

# Utility functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def secure_file_path(directory, filename):
    """Create a secure file path that prevents directory traversal"""
    filename = secure_filename(filename)
    return os.path.join(directory, filename)

# Model-related functions remain mostly the same, but with added logging
def load_image(image_path):
    try:
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image
    except Exception as e:
        logging.error(f"Error loading image: {str(e)}")
        raise

def load_model(model_config_path, model_checkpoint_path, device):
    try:
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def get_grounded_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    
    # Filter outputs
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    
    # Get phrases
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)
    
    return boxes_filt, pred_phrases, logits_filt

@timer_decorator
def process_image(image_path, prompt, output_dir):
    try:
        logging.info(f"Processing image: {image_path} with prompt: {prompt}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load image
        image_pil, image = load_image(image_path)
        
        # Detect objects
        boxes_filt, pred_phrases, logits_filt = get_grounded_output(
            model, image, prompt, 
            Config.BOX_THRESHOLD, 
            Config.TEXT_THRESHOLD, 
            device=device
        )
        
        if len(boxes_filt) == 0:
            logging.info("No objects detected in the image")
            return []
        
        # Set image for SAM
        image_cv2 = cv2.imread(image_path)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv2)
        
        # Process boxes and generate masks
        size = image_pil.size
        H, W = size[1], size[0]
        boxes_filt = process_boxes(boxes_filt, W, H)
        
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv2.shape[:2]).to(device)
        
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )
        
        # Save regular visualization
        save_visualization(image_cv2, masks, boxes_filt, pred_phrases, output_dir)
        
        # Save masked output with transparent background
        save_masked_output(image_cv2, masks, boxes_filt, padding= Config.MASK_PADDING, output_dir=output_dir)
        
        # Save object data
        object_data = save_object_data(boxes_filt, pred_phrases, logits_filt, output_dir, image_path)

        logging.info(f"Successfully processed image and saved outputs to {output_dir}")
        return object_data
        
    except Exception as e:
        logging.error(f"Error in process_image: {str(e)}")
        raise

def process_boxes(boxes_filt, W, H):
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt.cpu()

def save_visualization(image, masks, boxes, phrases, output_dir):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes, phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")
    plt.close()

def save_masked_output(image, masks, boxes, padding=5, output_dir=None):
    """
    Creates a PNG with only the masked portions and transparent background.
    Adds padding around the masked areas.
    
    Args:
        image: Original image (RGB format)
        masks: Predicted masks from SAM
        boxes: Detected boxes
        padding: Number of pixels to pad around each mask
        output_dir: Directory to save the output
    """
    # Create a transparent RGBA image
    height, width = image.shape[:2]
    transparent_mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Combine all masks with padding
    combined_mask = np.zeros((height, width), dtype=bool)
    for mask in masks:
        mask_np = mask.cpu().numpy()[0]  # Convert to numpy and get first mask
        
        # Add padding to the mask
        kernel = np.ones((padding*2, padding*2), np.uint8)
        padded_mask = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=1)
        combined_mask = combined_mask | padded_mask.astype(bool)
    
    # Apply the combined mask to the original image
    transparent_mask[combined_mask] = np.concatenate([image[combined_mask], np.full((combined_mask.sum(), 1), 255)], axis=1)
    
    # Convert to PIL Image and save
    masked_image = Image.fromarray(transparent_mask)
    output_path = os.path.join(output_dir, "masked_output.png")
    masked_image.save(output_path, format='PNG')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def save_object_data(boxes_filt, pred_phrases, logits_filt, output_dir, image_path):
    object_data = []
    # Read the original image
    image = cv2.imread(image_path)

    for box, phrase, logit in zip(boxes_filt, pred_phrases, logits_filt):
         # Get box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Extract region of interest (ROI)
        roi = image[y1:y2, x1:x2]
        
        # Perform OCR on the ROI
        if roi.size > 0:  # Check if ROI is not empty
            try:
                # Detect text in the ROI
                text_results = reader.readtext(roi)
                # Extract all detected text
                detected_text = ' '.join([text[1] for text in text_results]) if text_results else ''
            except Exception as e:
                logging.error(f"Error in OCR: {str(e)}")
                detected_text = ''
        else:
            detected_text = ''

        object_data.append({
            "object": phrase,
            "bbox": {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1])
            },
            "confidence": float(logit.max()),
            "detected_text": detected_text
        })
    
    with open(os.path.join(output_dir, "object_data.json"), "w") as f:
        json.dump(object_data, f, indent=4)
    
    return object_data

# API endpoints
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

@app.route('/api/v1/process_image', methods=['POST'])
@timer_decorator
def process_image_route():
    try:
        # Validate request
        if 'image' not in request.files:
            logging.warning("No image file in request")
            return jsonify({"error": "No image uploaded"}), 400
        
        image = request.files['image']
        if not image or not allowed_file(image.filename):
            logging.warning(f"Invalid file: {image.filename if image else 'None'}")
            return jsonify({"error": "Invalid file type"}), 400
        
        prompt = request.form.get('prompt', '')
        if not prompt:
            logging.warning("No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400
        
        # Process image
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        image_path = secure_file_path(Config.OUTPUT_DIR, "uploaded_image.jpg")
        image.save(image_path)
        
        object_data = process_image(image_path, prompt, Config.OUTPUT_DIR)
        
        return jsonify({
            "status": "success",
            "message": "Image processed successfully",
            "output_files": {
                "detection_image": "grounded_sam_output.jpg",
                "masked_output": "masked_output.png",  # Add this line
                "object_data": "object_data.json"
            },
            "objects": object_data
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    logging.error(f"404 Error: {request.url}")
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"500 Error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# Initialize app
if __name__ == '__main__':
    setup_logging()
    
    # Startup logging messages
    logging.info("Starting Grounded-SAM API Server...")
    logging.info(f"Running on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize models only once
    if init_models():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        logging.info(f"Server is running on http://{Config.HOST}:{Config.PORT}")
        # Set use_reloader=False to prevent double loading of models
        app.run(debug=True, host=Config.HOST, port=Config.PORT, use_reloader=False)
    else:
        logging.error("Failed to initialize models. Exiting...")
        exit(1)