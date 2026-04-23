import os
import sys
import shutil

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output

import depth_pro


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_depth_pro_checkpoint(input_data):
	override = input_data.get("checkpoint_path")
	if isinstance(override, str) and override.strip():
		checkpoint_path = os.path.abspath(override.strip())
		if os.path.isfile(checkpoint_path):
			return checkpoint_path

	env_checkpoint = os.environ.get("DEPTH_PRO_CHECKPOINT")
	if isinstance(env_checkpoint, str) and env_checkpoint.strip():
		checkpoint_path = os.path.abspath(env_checkpoint.strip())
		if os.path.isfile(checkpoint_path):
			return checkpoint_path

	candidates = [
		os.path.join(os.getcwd(), "checkpoints", "depth_pro.pt"),
		os.path.join(WORKSPACE_ROOT, "checkpoints", "depth_pro.pt"),
		os.path.join(WORKSPACE_ROOT, "ml-depth-pro", "checkpoints", "depth_pro.pt"),
		os.path.join(os.path.dirname(WORKSPACE_ROOT), "ml-depth-pro", "checkpoints", "depth_pro.pt"),
	]

	for candidate in candidates:
		if os.path.isfile(candidate):
			return os.path.abspath(candidate)

	for scan_root in [WORKSPACE_ROOT, os.path.dirname(WORKSPACE_ROOT)]:
		if not os.path.isdir(scan_root):
			continue
		for root, _, files in os.walk(scan_root):
			if "depth_pro.pt" in files:
				return os.path.abspath(os.path.join(root, "depth_pro.pt"))

	raise FileNotFoundError(
		"Could not locate depth_pro.pt. Set input field 'checkpoint_path' or env var DEPTH_PRO_CHECKPOINT."
	)


def _stage_checkpoint_for_depth_pro(checkpoint_path):
	expected_dir = os.path.join(os.getcwd(), "checkpoints")
	expected_path = os.path.join(expected_dir, "depth_pro.pt")

	if os.path.isfile(expected_path):
		return

	os.makedirs(expected_dir, exist_ok=True)
	try:
		if os.path.islink(expected_path) or os.path.exists(expected_path):
			os.remove(expected_path)
		os.symlink(checkpoint_path, expected_path)
	except Exception:
		shutil.copy2(checkpoint_path, expected_path)


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
	f_px_tensor = torch.tensor([f_px], dtype=torch.float32)
	if torch.cuda.is_available():
		f_px_tensor = f_px_tensor.cuda()
	checkpoint_path = _find_depth_pro_checkpoint(input_data)
	_stage_checkpoint_for_depth_pro(checkpoint_path)

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

	prediction = model.infer(transformed_image, f_px=f_px_tensor)
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