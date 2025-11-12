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

def crop_panorama(image):
    height, width = image.shape[:2]
    middle_height_start = height // 3
    middle_height_end = (height * 2) // 3
    middle_image = image[middle_height_start:middle_height_end, :]
    return middle_image, middle_height_start

def generate_point(image, prompt):
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
                else:
                    bboxes.append(None)
    return bboxes


def detect(image_path, prompt):
    cv_img = cv2.imread(image_path)
    # cropped_cv_image, y_offset = crop_panorama(cv_img)
    cropped_cv_image = cv_img
    y_offset = 0
    cropped_cv_image_rgb = cv2.cvtColor(cropped_cv_image, cv2.COLOR_BGR2RGB)
    height, width = cropped_cv_image_rgb.shape[:2]

    cropped_pil_image_rgb = Image.fromarray(cropped_cv_image_rgb)
    molmo_out = generate_point(cropped_pil_image_rgb, prompt)
    obj_coord = parse_coord(molmo_out, height, width)

    if not len(obj_coord): ### TODO: check here if fails
        raise ValueError("No object coordinates found")

    sam_out = get_mask(cropped_pil_image_rgb, obj_coord)

    # Adjust bounding boxes by adding the y_offset to y coordinates (like in draw_bbox)
    adjusted_bboxes = []
    for bbox in sam_out:
        if bbox:
            x, y, w, h = bbox
            # Adjust the y-coordinate
            adjusted_bboxes.append([x, y + y_offset, w, h])

    return adjusted_bboxes

# Flask route to handle object detection
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image file and prompt are required"}), 400

    file = request.files['file']
    depth_map = request.files['depth']
    prompt = request.form['prompt']
    
    temp_image_path = "/tmp/temp_image.jpg"
    temp_depth_path = "/tmp/temp_depth.png"

    file.save(temp_image_path)
    depth_map.save(temp_depth_path)
    depth_map = cv2.imread(temp_depth_path)
    
    try:
        bounding_boxes = detect(temp_image_path, prompt)
        print(f"Detected bounding boxes: {bounding_boxes}, {depth_map.shape}")
        distance = distance_to_detected_center(bounding_boxes[0], cv2.imread(temp_depth_path, cv2.IMREAD_UNCHANGED))
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    os.remove(temp_image_path)
    
    try:
        return jsonify({"bounding_box": bounding_boxes[0], "distance": float(distance)})
    except Exception as e:
        print(distance, type(distance))
        return jsonify({"error": str(e)}), 500



def get_prompt(image_path, prompt):
    cv_img = cv2.imread(image_path)
    cropped_cv_image, y_offset = crop_panorama(cv_img)
    cropped_cv_image_rgb = cv2.cvtColor(cropped_cv_image, cv2.COLOR_BGR2RGB)
    height, width = cropped_cv_image_rgb.shape[:2]

    cropped_pil_image_rgb = Image.fromarray(cropped_cv_image_rgb)
    molmo_out = generate_vlm_request(cropped_pil_image_rgb, prompt)
    return molmo_out

# Flask route to handle object detection
@app.route('/prompt_vlm', methods=['POST'])
def prompt_vlm():
    if 'file' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image file and prompt are required"}), 400

    file = request.files['file']
    prompt = request.form['prompt']
    
    print(f"prompt: {prompt}")
    temp_image_path = "/tmp/temp_image.jpg"
    file.save(temp_image_path)
    
    try:
        response = get_prompt(temp_image_path, prompt)
        print("Response:", response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    os.remove(temp_image_path)
    
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)
