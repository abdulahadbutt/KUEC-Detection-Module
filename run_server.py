from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import SAM
import os
import time
import xml.etree.ElementTree as ET



app = Flask(__name__)

# Load models
sam_model = SAM("sam2_b.pt")
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)


def distance_to_detected_center(bounding_box, depth_map):
    x, y, w, h = bounding_box
    center_x = x + w // 2
    center_y = y + h // 2

    # Ensure the center coordinates are within the depth map bounds
    center_x = min(max(center_x, 0), depth_map.shape[1] - 1)
    center_y = min(max(center_y, 0), depth_map.shape[0] - 1)

    # Get the depth value at the center of the bounding box
    depth_value = depth_map[center_y, center_x]

    print(f"Depth at detected center ({center_x}, {center_y}): {depth_value}")
    return depth_value



def generate_point(image, prompt):
    '''
    Given an image and a prompt, generate the coordinates of the object in the image.
    '''
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            inputs = processor.process(images=[image], text="Point the " + prompt)
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text



def generate_vlm_request(image, prompt):
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            inputs = processor.process(images=[image], text='Answer in one complete sentence: ' + prompt)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    inputs[k] = v.to(torch.bfloat16)
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


def parse_coord(text, height, width):
    text = text.strip()
    points = []

    if not text.startswith("<"):
        return points

    root = ET.fromstring(text)

    if text.startswith("<points"):
        num_points = (len(root.items()) - 1) // 2
        for i in range(1, num_points + 1):
            x = float(root.get(f'x{i}'))
            y = float(root.get(f'y{i}'))
            x = int(x * width / 100)
            y = int(y * height / 100)
            points.append([x, y])

    elif text.startswith("<point"):
        x = float(root.get('x'))
        y = float(root.get('y'))
        x = int(x * width / 100)
        y = int(y * height / 100)
        points.append([x, y])

    points = np.array(points)
    return points


def get_mask(image, coords):
    bboxes = []
    masks = []
    if len(coords) > 0:
        sam_output = sam_model(image, points=coords, labels=np.ones(len(coords)), verbose=False)
        if sam_output[0].masks:
            mask_tensor = sam_output[0].masks.data.cpu().numpy()
            pred_mask = (mask_tensor > 0).astype(np.uint8)
            for i in range(pred_mask.shape[0]):
                contours, _ = cv2.findContours(pred_mask[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    bboxes.append([x, y, w, h])
                    masks.append(pred_mask[i].tolist())
                else:
                    bboxes.append(None)
                    masks.append(None)
    return bboxes, masks


def _draw_overlays(
    base_img_bgr,
    points_xy,
    bboxes_xywh,
    masks=None,
    y_offset: int = 0,
    point_color = (0, 255, 255),    # BGR (yellow)
    box_color = (0, 140, 255),      # BGR (orange)
    mask_color = (0, 255, 0),       # BGR (green)
    point_radius: int = 6,
    box_thickness: int = 2,
    mask_alpha: float = 0.3,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray:
    """Draw SAM masks, Molmo points and SAM boxes on a copy of the input image."""
    overlay = base_img_bgr.copy()

    # Draw masks
    if masks is not None:
        for j, mask in enumerate(masks):
            if mask is not None:
                mask_array = np.array(mask, dtype=np.uint8) * 255
                mask_colored = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
                mask_colored[:,:] = mask_color
                # Apply mask only where mask_array is > 0
                mask_region = mask_array > 0
                overlay[mask_region] = cv2.addWeighted(overlay[mask_region], 1 - mask_alpha, mask_colored[mask_region], mask_alpha, 0)
            break 


    # Draw Molmo points
    if points_xy is not None and len(points_xy) > 0:
        for point in points_xy:
            x, y = int(point[0]), int(point[1])
            cv2.circle(overlay, (x, y), point_radius, (255, 0, 0), -1, lineType=cv2.LINE_AA)  # Blue for Molmo points
            cv2.putText(
                overlay, "Molmo Point",
                (x + 10, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA
            )
            break 


    # Draw boxes and midpoints
    for j, (bx, by, bw, bh) in enumerate(bboxes_xywh):
        yy = by + y_offset
        mid_x = bx + bw // 2
        mid_y = by + bh // 2 + y_offset
        cv2.circle(overlay, (mid_x, mid_y), point_radius, point_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay, "Midpoint",
            (mid_x + 10, mid_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, point_color, font_thickness, cv2.LINE_AA
        )
        cv2.rectangle(overlay, (bx, yy), (bx + bw, yy + bh), box_color, box_thickness)
        cv2.putText(
            overlay, f"box {j+1}",
            (bx, max(0, yy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, font_thickness, cv2.LINE_AA
        )
        break 

    return overlay


def detect(image_path, prompt):
    '''
    Main detection function that processes the image and prompt to return bounding boxes.
    1) Load image
    '''
    cv_img = cv2.imread(image_path)
    cv_image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width = cv_image_rgb.shape[:2]

    pil_image_rgb = Image.fromarray(cv_image_rgb)
    molmo_out = generate_point(pil_image_rgb, prompt)
    obj_coord = parse_coord(molmo_out, height, width)

    print(f"Molmo output: {molmo_out}")
    print(f"Parsed object coordinates: {obj_coord}")
    if not len(obj_coord):
        raise ValueError("No object coordinates found")

    sam_bbs, sam_masks = get_mask(pil_image_rgb, obj_coord)

    # Adjust bounding boxes by adding the y_offset to y coordinates (like in draw_bbox)
    adjusted_bboxes = []
    adjusted_masks = []
    for bbox, mask in zip(sam_bbs, sam_masks):
        if bbox:
            x, y, w, h = bbox
            # Adjust the y-coordinate
            adjusted_bboxes.append([x, y, w, h])
            adjusted_masks.append(mask)

    annotated = _draw_overlays(cv_img, obj_coord, adjusted_bboxes, masks=adjusted_masks, y_offset=0)

    # 5) Optionally save
    save_annotated_to = "data/annotated_output.jpg"
    if save_annotated_to:
        ok = cv2.imwrite(save_annotated_to, annotated)
        if not ok:
            raise IOError(f"Failed to write annotated image to: {save_annotated_to}")

    return adjusted_bboxes, adjusted_masks

# Flask route to handle object detection
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files or 'prompt' not in request.form or 'depth' not in request.files:
        return jsonify({"error": "Image file, prompt and depth map are required"}), 400

    # * Getting inputs
    file = request.files['image']
    depth_map = request.files['depth']
    prompt = request.form['prompt']
    

    # * Save the uploaded files temporarily
    temp_image_path = "data/temp_image.jpg"
    temp_depth_path = "data/temp_depth.png"

    file.save(temp_image_path)
    depth_map.save(temp_depth_path)

    try:
        bounding_boxes, masks = detect(temp_image_path, prompt)
        print(f"Detected bounding boxes: {bounding_boxes}")
        distance = distance_to_detected_center(bounding_boxes[0], cv2.imread(temp_depth_path, cv2.IMREAD_UNCHANGED))
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    os.remove(temp_image_path)
    
    try:
        return jsonify({"bounding_box": bounding_boxes[0], "distance": float(distance)})
    except Exception as e:
        print(distance, type(distance))
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    DEBUG = True # Set to False in production
    app.run(host='0.0.0.0', port=5007)
