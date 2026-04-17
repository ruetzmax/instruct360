import argparse
import json
import os
import sys
import traceback

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.append("sam3d-objects")

from sam3d_objects.model import build_model  # type: ignore[reportMissingImports]
from sam3d_objects.segment import segment_image  # type: ignore[reportMissingImports]
from sam3d_objects.utils import get_point_cloud_mask  # type: ignore[reportMissingImports]

from inference_utils import base64_to_image, image_to_base64

RESULT_PREFIX = "__INFER_DONE__"


class Sam3dWorker:
    def __init__(self):
        print("[sam3d_worker] Loading SAM3D model...", file=sys.stderr)
        self.model = build_model("sam3d-objects/checkpoints/sam3d.pth")
        print("[sam3d_worker] SAM3D model loaded.", file=sys.stderr)

    def run_inference(self, input_data):
        image = base64_to_image(input_data["image_base64"])
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        image_pil = Image.fromarray(image)

        points = np.array(input_data["points"])

        if points.size == 0:
            masks_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            return {
                "masks_base64": image_to_base64(masks_image),
                "masks_3d": points.tolist(),
            }

        depth = np.array(input_data["depth"], dtype=np.float32)
        intrinsics = np.array(input_data["intrinsics"], dtype=np.float32)

        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        target_mask = segment_image(image_cv, self.model, points)
        masked_points = get_point_cloud_mask(points, target_mask, intrinsics, image.shape)

        masks_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        masks_image[:, :, :3] = image
        masks_image[:, :, 3] = target_mask * 255

        return {
            "masks_base64": image_to_base64(masks_image),
            "masks_3d": masked_points.tolist(),
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

    worker = Sam3dWorker()

    if args.persistent:
        run_persistent(worker)
        return

    if not args.input_json or not args.output_json:
        raise ValueError("Usage: python sam3d_worker.py <input_json> <output_json> or --persistent")

    run_single(worker, args.input_json, args.output_json)


if __name__ == "__main__":
    main()
