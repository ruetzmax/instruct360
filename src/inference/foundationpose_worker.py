import argparse
import json
import os
import site
import sys
import traceback
import glob

import numpy as np
import trimesh
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
foundationpose_root = os.path.join(workspace_root, "FoundationPose")
if foundationpose_root not in sys.path:
    sys.path.insert(0, foundationpose_root)


RESULT_PREFIX = "__INFER_DONE__"


def _augment_ld_library_path_for_torch_cuda():
    candidate_dirs = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
    ]

    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            site_dirs.append(user_site)
        elif isinstance(user_site, (list, tuple)):
            site_dirs.extend(user_site)
    except Exception:
        pass

    for sys_path in sys.path:
        if isinstance(sys_path, str) and "site-packages" in sys_path:
            site_dirs.append(sys_path)

    deduped_site_dirs = []
    for site_dir in site_dirs:
        if site_dir and site_dir not in deduped_site_dirs:
            deduped_site_dirs.append(site_dir)

    for site_dir in deduped_site_dirs:
        candidate_dirs.extend([
            os.path.join(site_dir, "torch", "lib"),
            os.path.join(site_dir, "nvidia", "cuda_nvrtc", "lib"),
            os.path.join(site_dir, "nvidia", "cuda_nvrtc", "lib64"),
            os.path.join(site_dir, "nvidia", "cuda_nvrtc", "targets", "x86_64-linux", "lib"),
            os.path.join(site_dir, "nvidia", "cuda_runtime", "lib"),
            os.path.join(site_dir, "nvidia", "cuda_runtime", "lib64"),
            os.path.join(site_dir, "nvidia", "cuda_runtime", "targets", "x86_64-linux", "lib"),
            os.path.join(site_dir, "nvidia", "cudnn", "lib"),
            os.path.join(site_dir, "nvidia", "cudnn", "lib64"),
            os.path.join(site_dir, "nvidia", "cudnn", "targets", "x86_64-linux", "lib"),
            os.path.join(site_dir, "nvidia", "cublas", "lib"),
            os.path.join(site_dir, "nvidia", "cublas", "lib64"),
            os.path.join(site_dir, "nvidia", "cublas", "targets", "x86_64-linux", "lib"),
            os.path.join(site_dir, "nvidia", "cusolver", "lib"),
            os.path.join(site_dir, "nvidia", "cusolver", "lib64"),
            os.path.join(site_dir, "nvidia", "cusolver", "targets", "x86_64-linux", "lib"),
            os.path.join(site_dir, "nvidia", "curand", "lib"),
            os.path.join(site_dir, "nvidia", "curand", "lib64"),
            os.path.join(site_dir, "nvidia", "curand", "targets", "x86_64-linux", "lib"),
            os.path.join(site_dir, "nvidia", "cufft", "lib"),
            os.path.join(site_dir, "nvidia", "cufft", "lib64"),
            os.path.join(site_dir, "nvidia", "cufft", "targets", "x86_64-linux", "lib"),
        ])

        nvidia_root = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_root):
            for package_dir in os.listdir(nvidia_root):
                package_root = os.path.join(nvidia_root, package_dir)
                candidate_dirs.extend([
                    os.path.join(package_root, "lib"),
                    os.path.join(package_root, "lib64"),
                    os.path.join(package_root, "targets", "x86_64-linux", "lib"),
                ])

    # Fallback for environments that ship only versioned NVRTC libraries (e.g. libnvrtc.so.12)
    # but attempt to dlopen libnvrtc.so.
    nvrtc_exists = False
    nvrtc_versioned = []
    for path in candidate_dirs:
        if not path or not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "libnvrtc.so")):
            nvrtc_exists = True
            break
        nvrtc_versioned.extend(glob.glob(os.path.join(path, "libnvrtc.so.*")))

    if not nvrtc_exists and nvrtc_versioned:
        link_dir = os.path.join(workspace_root, "temp", "cuda_lib_links")
        os.makedirs(link_dir, exist_ok=True)
        target = sorted(nvrtc_versioned)[-1]
        link_path = os.path.join(link_dir, "libnvrtc.so")
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(target, link_path)
            candidate_dirs.insert(0, link_dir)
        except Exception:
            pass

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    merged = []

    for path in candidate_dirs + existing_parts:
        if path and os.path.isdir(path) and path not in merged:
            merged.append(path)

    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


_augment_ld_library_path_for_torch_cuda()

import torch

import estimater as fp  # type: ignore[reportMissingImports]
import learning.training.predict_pose_refine as predict_pose_refine  # type: ignore[reportMissingImports]
import learning.training.predict_score as predict_score  # type: ignore[reportMissingImports]
import Utils as fp_utils  # type: ignore[reportMissingImports]

from inference_utils import base64_to_image


def _compute_crop_window_tf_batch_float32(
    pts=None,
    H=None,
    W=None,
    poses=None,
    K=None,
    crop_ratio=1.2,
    out_size=None,
    rgb=None,
    uvs=None,
    method='min_box',
    mesh_diameter=None,
):
    if method != 'box_3d':
        raise RuntimeError("Only method='box_3d' is supported in this patched path")

    if poses is None or K is None or mesh_diameter is None or out_size is None:
        raise ValueError("poses, K, mesh_diameter and out_size are required")

    if torch.is_tensor(poses):
        poses_t = poses.to(dtype=torch.float32)
        device = poses_t.device
    else:
        poses_t = torch.as_tensor(np.asarray(poses, dtype=np.float32), dtype=torch.float32, device='cuda')
        device = poses_t.device

    K_t = torch.as_tensor(np.asarray(K, dtype=np.float32), dtype=torch.float32, device=device)

    B = len(poses_t)
    radius = float(mesh_diameter) * float(crop_ratio) / 2.0
    offsets = torch.tensor(
        [0, 0, 0, radius, 0, 0, -radius, 0, 0, 0, radius, 0, 0, -radius, 0],
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 3)

    pts_t = poses_t[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
    projected = (K_t @ pts_t.reshape(-1, 3).T).T
    uvs_t = projected[:, :2] / projected[:, 2:3]
    uvs_t = uvs_t.reshape(B, -1, 2)
    center = uvs_t[:, 0]
    radius_t = torch.abs(uvs_t - center.reshape(-1, 1, 2)).reshape(B, -1).max(axis=-1)[0].reshape(-1)

    left = center[:, 0].round() - radius_t.round()
    right = center[:, 0].round() + radius_t.round()
    top = center[:, 1].round() - radius_t.round()
    bottom = center[:, 1].round() + radius_t.round()

    tf = torch.eye(3, dtype=torch.float32, device=device)[None].expand(B, -1, -1).contiguous()
    tf[:, 0, 2] = -left
    tf[:, 1, 2] = -top

    new_tf = torch.eye(3, dtype=torch.float32, device=device)[None].expand(B, -1, -1).contiguous()
    new_tf[:, 0, 0] = float(out_size[0]) / (right - left)
    new_tf[:, 1, 1] = float(out_size[1]) / (bottom - top)

    return new_tf @ tf


predict_pose_refine.compute_crop_window_tf_batch = _compute_crop_window_tf_batch_float32
predict_score.compute_crop_window_tf_batch = _compute_crop_window_tf_batch_float32
fp_utils.compute_crop_window_tf_batch = _compute_crop_window_tf_batch_float32


def _coerce_mesh_dtypes(mesh):
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh.faces = np.asarray(mesh.faces, dtype=np.int32)
    _ = mesh.vertex_normals
    return mesh


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

    return _coerce_mesh_dtypes(mesh)


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


class FoundationPoseWorker:
    def __init__(self):
        fp.set_logging_format()
        fp.set_seed(0)

        print("[foundationpose_worker] Loading FoundationPose networks...", file=sys.stderr)
        self.scorer = fp.ScorePredictor()
        self.refiner = fp.PoseRefinePredictor()
        self.glctx = fp.dr.RasterizeCudaContext()
        print("[foundationpose_worker] FoundationPose networks loaded.", file=sys.stderr)

    def run_inference(self, input_data):
        is_first_frame = input_data.get("is_first_frame", False)
        rgb_base64 = input_data.get("rgb_base64")
        depth_npy_path = input_data.get("depth_npy_path")
        mask_base64 = input_data.get("mask_base64")
        unposed_mesh_path = input_data.get("unposed_mesh_path")
        intrinsics = input_data.get("intrinsics")
        debug = int(input_data.get("debug", 0))
        debug_dir = input_data.get("debug_dir", os.path.join(workspace_root, "temp", "foundationpose_debug"))
        debug_image_path = input_data.get("debug_image_path", os.path.join(debug_dir, "debug_rgb.png"))
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(os.path.dirname(debug_image_path), exist_ok=True)

        rgb = base64_to_image(rgb_base64)
        if not depth_npy_path:
            raise ValueError("depth_npy_path is required; depth estimation is not supported in foundationpose_worker")
        depth = np.load(depth_npy_path).astype(np.float32)
        mask = base64_to_image(mask_base64) if mask_base64 is not None else None
        intrinsics = np.asarray(intrinsics, dtype=np.float32)

        if intrinsics.shape != (3, 3):
            raise ValueError(f"Expected intrinsics shape (3, 3), got {intrinsics.shape}")
        if depth.ndim != 2:
            raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")

        mesh = read_trimesh(unposed_mesh_path)

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        est = fp.FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=debug_dir,
            debug=debug,
            glctx=self.glctx,
        )

        if hasattr(est, "mesh") and est.mesh is not None:
            est.mesh.vertices = np.asarray(est.mesh.vertices, dtype=np.float32)
            _ = est.mesh.vertex_normals

        if mask is None:
            mask = depth > 0.0
        else:
            if mask.ndim == 3:
                mask = mask[..., -1]
            mask = mask.astype(bool)

        valid_depth = np.isfinite(depth) & (depth >= 0.001)
        if not np.any(valid_depth):
            raise ValueError("No valid depth pixels found (depth >= 0.001)")

        if not np.any(mask & valid_depth):
            print(
                "[foundationpose_worker] provided mask has no overlap with valid depth; "
                "falling back to valid-depth mask.",
                file=sys.stderr,
            )
            mask = valid_depth

        iteration = 5 if is_first_frame else 1
        pose = est.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)

        center_pose = pose @ np.linalg.inv(to_origin)

        if debug >= 3:
            try:
                transformed_mesh = mesh.copy()
                transformed_mesh.apply_transform(pose)
                transformed_mesh.export(os.path.join(debug_dir, "model_tf.obj"))

                xyz_map = fp_utils.depth2xyzmap(depth, intrinsics)
                valid = depth >= 0.001
                if np.any(valid):
                    pcd = fp_utils.toOpen3dCloud(xyz_map[valid], rgb[valid])
                    fp_utils.o3d.io.write_point_cloud(
                        os.path.join(debug_dir, "scene_complete.ply"),
                        pcd,
                    )
            except Exception as exc:
                print(
                    f"[foundationpose_worker] failed to write debug>=3 artifacts in {debug_dir}: {exc}",
                    file=sys.stderr,
                )

        if debug >= 1:
            try:
                debug_image_dir = os.path.dirname(debug_image_path)
                debug_image_name = os.path.basename(debug_image_path)
                debug_image_stem, debug_image_ext = os.path.splitext(debug_image_name)
                if not debug_image_ext:
                    debug_image_ext = ".png"

                rgb_debug_path = os.path.join(debug_image_dir, f"{debug_image_stem}_rgb{debug_image_ext}")
                mask_debug_path = os.path.join(debug_image_dir, f"{debug_image_stem}_mask{debug_image_ext}")

                cv2.imwrite(rgb_debug_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                mask_image = (mask.astype(np.uint8) * 255)
                cv2.imwrite(mask_debug_path, mask_image)

                vis = fp_utils.draw_posed_3d_box(
                    intrinsics,
                    img=rgb.copy(),
                    ob_in_cam=center_pose,
                    bbox=bbox,
                )
                vis = fp_utils.draw_xyz_axis(
                    vis,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=intrinsics,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
                cv2.imwrite(debug_image_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            except Exception as exc:
                print(
                    f"[foundationpose_worker] failed to write debug frame to {debug_image_path}: {exc}",
                    file=sys.stderr,
                )

        return {
            "pose": pose.tolist(),
            "center_pose": center_pose.tolist(),
        }


def process_request(worker, input_json, output_json):
    with open(input_json, "r") as f:
        input_data = json.load(f)

    output_data = worker.run_inference(input_data)

    with open(output_json, "w") as f:
        json.dump(output_data, f)


def run_persistent(worker):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            if req.get("shutdown"):
                print(RESULT_PREFIX + json.dumps({"ok": True, "shutdown": True}), flush=True)
                break

            input_json = req["input_json"]
            output_json = req["output_json"]
            process_request(worker, input_json, output_json)
            print(RESULT_PREFIX + json.dumps({"ok": True, "output_json": output_json}), flush=True)
        except Exception:
            error_text = traceback.format_exc()
            print(RESULT_PREFIX + json.dumps({"ok": False, "error": error_text}), flush=True)


def run_single(worker, input_json, output_json):
    process_request(worker, input_json, output_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?")
    parser.add_argument("output_json", nargs="?")
    parser.add_argument("--persistent", action="store_true")
    args = parser.parse_args()

    worker = FoundationPoseWorker()

    if args.persistent:
        run_persistent(worker)
        return

    if not args.input_json or not args.output_json:
        raise ValueError("Usage: python foundationpose_worker.py <input_json> <output_json> or --persistent")

    run_single(worker, args.input_json, args.output_json)


if __name__ == "__main__":
    main()
