import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output

import depth_pro


def main():
	input_data = load_inference_input()

	image_base64 = input_data.get("image_base64")
	depth_npy_path = input_data.get("depth_npy_path")
	f_px = input_data.get("f_px")

	if image_base64 is None:
		raise ValueError("Missing required field: image_base64")
	if depth_npy_path is None:
		raise ValueError("Missing required field: depth_npy_path")
	if f_px is None:
		raise ValueError("Missing required field: f_px")

	f_px = float(f_px)

	model, transform = depth_pro.create_model_and_transforms()
	model.eval()

	if torch.cuda.is_available():
		model = model.cuda()

	image = base64_to_image(image_base64)
	if image.ndim == 3 and image.shape[2] == 4:
		image = image[:, :, :3]

	transformed_image = transform(image)
	if torch.cuda.is_available():
		transformed_image = transformed_image.cuda()

	prediction = model.infer(transformed_image, f_px=f_px)
	depth = prediction["depth"]

	if torch.is_tensor(depth):
		depth = depth.detach().cpu().numpy()

	depth = np.asarray(depth, dtype=np.float32).squeeze()
	if depth.ndim != 2:
		raise ValueError(f"Expected 2D depth map from depth-pro, got shape {depth.shape}")

	np.save(depth_npy_path, depth)

	output_data = {
		"ok": True,
		"depth_npy_path": depth_npy_path,
		"depth_units": "m",
	}
	save_inference_output(output_data)


if __name__ == "__main__":
	main()