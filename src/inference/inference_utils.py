import base64
import json
import os
from io import BytesIO
import sys
import numpy as np
from PIL import Image


def _augment_ld_library_path():
    candidate_dirs = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
    ]

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    merged = []
    for path in candidate_dirs + existing_parts:
        if path and os.path.isdir(path) and path not in merged:
            merged.append(path)

    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


_augment_ld_library_path()


def base64_to_image(img_str):
    img_data = base64.b64decode(img_str)
    with Image.open(BytesIO(img_data)) as image:
        if image.mode == "L":
            return np.array(image.convert("RGB"))
        if image.mode == "RGBA":
            return np.array(image)
        return np.array(image.convert("RGB"))


def image_to_base64(image):
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="L")
    elif len(image.shape) == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGBA")
    else:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_inference_input():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python script.py <input_json> [output_json]")
    
    input_json_path = sys.argv[1]
    with open(input_json_path, 'r') as f:
        return json.load(f)


def save_inference_output(output_data):
    if len(sys.argv) < 3:
        raise ValueError("Output JSON path not specified in command line args")
    
    output_json_path = sys.argv[2]
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f)
