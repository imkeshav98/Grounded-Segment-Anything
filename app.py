import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import flask
from flask import Flask, request, jsonify, send_file

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, SamPredictor

app = Flask(__name__)

# Configuration paths
CONFIG_FILE = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDED_CHECKPOINT = "groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
OUTPUT_DIR = "outputs"

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

# Detect objects
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

def process_image(image_path, prompt, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image and model
    image_pil, image = load_image(image_path)
    model = load_model(CONFIG_FILE, GROUNDED_CHECKPOINT, device=device)
    
    # Detect objects
    boxes_filt, pred_phrases, logits_filt = get_grounded_output(
        model, image, prompt, box_threshold=0.3, text_threshold=0.25, device=device
    )
    
    # If no objects detected, return empty list
    if len(boxes_filt) == 0:
        return []
    
    # Initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=SAM_CHECKPOINT).to(device))
    image_cv2 = cv2.imread(image_path)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv2)
    
    # Process bounding boxes
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv2.shape[:2]).to(device)
    
    # Generate masks
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    # Visualize detected objects
    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv2)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")
    
    # Create mask for inpainting
    mask = masks[0][0].cpu().numpy()
    mask_pil = Image.fromarray(mask)
    
    # Resize images for inpainting
    image_pil_resized = image_pil.resize((512, 512))
    mask_pil_resized = mask_pil.resize((512, 512))
    
    # Prepare data for JSON output
    object_data = []
    for i, (box, phrase, logit) in enumerate(zip(boxes_filt, pred_phrases, logits_filt)):
        object_data.append({
            "object": phrase,
            "bbox": {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1])
            },
            "confidence": float(logit.max())
        })
    
    # Save object data as JSON
    with open(os.path.join(output_dir, "object_data.json"), "w") as f:
        json.dump(object_data, f, indent=4)
    
    # Save mask image
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil.save(os.path.join(output_dir, "mask_image.jpg"))
    
    return object_data

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']
    prompt = request.form.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Save uploaded image
    image_path = os.path.join(OUTPUT_DIR, "uploaded_image.jpg")
    image.save(image_path)
    
    try:
        object_data = process_image(image_path, prompt, OUTPUT_DIR)
        return jsonify({
            "message": "Image processed successfully",
            "output_files": {
                "detection_image": "grounded_sam_output.jpg",
                "mask_image": "mask_image.jpg",
                "object_data": "object_data.json"
            },
            "objects": object_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)