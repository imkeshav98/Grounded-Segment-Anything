import argparse
import os
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import build_sam, SamPredictor 

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval()

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    image_tensor = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(image, None)[0]
    
    caption = caption.lower().strip() + "."
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    pred_items = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_item = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_items.append(pred_item)
    
    return boxes_filt, pred_items

def process_image(config_file, grounded_checkpoint, sam_checkpoint, image_path, det_prompt, 
                  output_dir, box_threshold=0.3, text_threshold=0.25, device="cuda"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image_np = load_image(image_path)
    H, W, _ = image_np.shape
    
    # Load models
    dino_model = load_model(config_file, grounded_checkpoint, device)
    
    # Run Grounding DINO
    boxes_filt, pred_items = get_grounding_output(
        dino_model, image_np, det_prompt, box_threshold, text_threshold, device
    )
    
    # Initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    predictor.set_image(image_np)
    
    # Adjust boxes
    boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])
    
    # Predict masks
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2]).to(device)
    
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')
    
    # Draw masks
    for mask in masks:
        mask_np = mask[0].cpu().numpy()
        color = np.random.rand(3)
        plt.imshow(mask_np, alpha=0.3, cmap='gray', vmin=0, vmax=1, interpolation='nearest', color=color)
    
    # Draw bounding boxes
    for box, item in zip(boxes_filt, pred_items):
        x, y, w, h = box.numpy()
        plt.gca().add_patch(plt.Rectangle((x, y), w-x, h-y, 
                                           fill=False, edgecolor='red', linewidth=2))
        plt.text(x, y, item, color='red', fontsize=10)
    
    # Save visualization
    plt.savefig(os.path.join(output_dir, "detection_visualization.jpg"), 
                bbox_inches="tight", pad_inches=0)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Visualization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--det_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    process_image(
        config_file=args.config,
        grounded_checkpoint=args.grounded_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        image_path=args.input_image,
        det_prompt=args.det_prompt,
        output_dir=args.output_dir
    )