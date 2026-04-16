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

from inference_utils import base64_to_image, load_inference_input, save_inference_output


def _make_foundationpose_mesh_compatible(mesh):
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        material = getattr(mesh.visual, "material", None)
        texture_image = getattr(material, "image", None) if material is not None else None
        image_is_usable = texture_image is not None and hasattr(texture_image, "convert")

        if not image_is_usable:
            vertex_colors = getattr(mesh.visual, "vertex_colors", None)
            if vertex_colors is None or len(vertex_colors) != len(mesh.vertices):
                vertex_colors = np.tile(
                    np.array([128, 128, 128, 255], dtype=np.uint8),
                    (len(mesh.vertices), 1),
                )
            else:
                vertex_colors = np.asarray(vertex_colors)
                if vertex_colors.ndim == 2 and vertex_colors.shape[1] == 3:
                    alpha_channel = np.full((len(vertex_colors), 1), 255, dtype=vertex_colors.dtype)
                    vertex_colors = np.concatenate([vertex_colors, alpha_channel], axis=1)
                vertex_colors = vertex_colors.astype(np.uint8)

            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)

    return mesh


def read_trimesh(mesh_or_path):
    if isinstance(mesh_or_path, (str, os.PathLike)):
        mesh = trimesh.load(mesh_or_path)
    else:
        mesh = mesh_or_path

    if isinstance(mesh, trimesh.Scene):
        sub_meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not sub_meshes:
            raise ValueError("GLB scene does not contain any mesh geometry")

        textured = [
            g for g in sub_meshes
            if getattr(getattr(g.visual, "material", None), "image", None) is not None
            or getattr(getattr(g.visual, "material", None), "baseColorTexture", None) is not None
        ]
        selected_mesh = textured[0] if textured else max(sub_meshes, key=lambda g: len(g.faces))
        return _make_foundationpose_mesh_compatible(selected_mesh)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh or trimesh.Scene, got {type(mesh)}")

    return _make_foundationpose_mesh_compatible(mesh)


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