import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output


def main():
    input_data = load_inference_input()

    image_base64 = input_data.get("image_base64")
    depth_npy_path = input_data.get("depth_npy_path")
    hf_device = int(input_data.get("hf_device", 0))
    depth_units = str(input_data.get("depth_units", "m")).strip().lower()

    if image_base64 is None:
        raise ValueError("Missing required field: image_base64")
    if depth_npy_path is None:
        raise ValueError("Missing required field: depth_npy_path")
    if depth_units not in {"m", "mm"}:
        raise ValueError("Invalid depth_units. Expected 'm' or 'mm'")

    model = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        device=hf_device,
    )

    image = base64_to_image(image_base64)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    input_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    depth = model(input_image)["depth"]

    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()

    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map from depth model, got shape {depth.shape}")

    if depth_units == "mm":
        depth = depth * 1000.0

    np.save(depth_npy_path, depth.astype(np.float32))

    output_data = {
        "ok": True,
        "depth_npy_path": depth_npy_path,
        "depth_units": depth_units,
    }
    save_inference_output(output_data)


if __name__ == "__main__":
    main()
