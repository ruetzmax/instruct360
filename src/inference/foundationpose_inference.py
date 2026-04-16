import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
foundationpose_root = os.path.join(workspace_root, "FoundationPose")
if foundationpose_root not in sys.path:
    sys.path.insert(0, foundationpose_root)

import estimater as fp  # type: ignore[reportMissingImports]
from src.util import read_trimesh

from inference_utils import base64_to_image, load_inference_input, save_inference_output


fp.set_logging_format()
fp.set_seed(0)

input_data = load_inference_input()
is_first_frame = input_data.get("is_first_frame", False)
rgb_base64 = input_data.get("rgb_base64")
depth_npy_path = input_data.get("depth_npy_path")
mask_base64 = input_data.get("mask_base64")
unposed_mesh_path = input_data.get("unposed_mesh_path")
intrinsics = input_data.get("intrinsics")
debug = int(input_data.get("debug", 0))
debug_dir = input_data.get("debug_dir", os.path.join(workspace_root, "temp", "foundationpose_debug"))
os.makedirs(debug_dir, exist_ok=True)

rgb = base64_to_image(rgb_base64)
depth = np.load(depth_npy_path).astype(np.float32)
mask = base64_to_image(mask_base64) if mask_base64 is not None else None
intrinsics = np.asarray(intrinsics, dtype=np.float32)

if intrinsics.shape != (3, 3):
    raise ValueError(f"Expected intrinsics shape (3, 3), got {intrinsics.shape}")
if depth.ndim != 2:
    raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")

mesh = read_trimesh(unposed_mesh_path)

to_origin, _ = trimesh.bounds.oriented_bounds(mesh)

scorer = fp.ScorePredictor()
refiner = fp.PoseRefinePredictor()
glctx = fp.dr.RasterizeCudaContext()
est = fp.FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)

if mask is None:
    mask = depth > 0.0
else:
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask = mask.astype(bool)

pose = est.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
    
center_pose = pose@np.linalg.inv(to_origin)
    
output_data = {
    "pose": pose.tolist(),
    "center_pose": center_pose.tolist(),
}

save_inference_output(output_data)