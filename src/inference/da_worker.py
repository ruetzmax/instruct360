import json
import os
import site
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output


def _augment_ld_library_path_for_torch_cuda():
    candidate_dirs = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
    ]

    for site_dir in site.getsitepackages():
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
from PIL import Image
from transformers import pipeline


def _summarize_depth(depth: np.ndarray) -> dict:
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return {
            "has_finite": False,
        }

    finite_depth = depth[finite_mask]
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    p05, p50, p95 = np.percentile(finite_depth, [5, 50, 95])
    return {
        "has_finite": True,
        "shape": [int(h), int(w)],
        "min": float(np.min(finite_depth)),
        "max": float(np.max(finite_depth)),
        "mean": float(np.mean(finite_depth)),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "center": float(depth[cy, cx]),
    }


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

    # if depth_units == "mm":
    #     depth = depth * 1000.0

    depth_debug = _summarize_depth(depth)

    if depth_debug.get("has_finite"):
        print(
            (
                f"[da_worker] depth units={depth_units} "
                f"shape={tuple(depth_debug['shape'])} "
                f"min={depth_debug['min']:.4f} "
                f"max={depth_debug['max']:.4f} "
                f"mean={depth_debug['mean']:.4f} "
                f"p05={depth_debug['p05']:.4f} p50={depth_debug['p50']:.4f} p95={depth_debug['p95']:.4f} "
                f"center={depth_debug['center']:.4f}"
            )
        )
    else:
        print(f"[da_worker] depth units={depth_units} no finite depth values")

    
    np.save(depth_npy_path, depth.astype(np.float32))

    output_data = {
        "ok": True,
        "depth_npy_path": depth_npy_path,
        "depth_units": depth_units,
        "depth_debug": depth_debug,
    }
    save_inference_output(output_data)


if __name__ == "__main__":
    main()
