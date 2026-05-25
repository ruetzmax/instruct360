import argparse
import json
import os
import sys
import traceback

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.append("sam3")
from sam3.model_builder import build_sam3_image_model  # type: ignore[reportMissingImports]
from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[reportMissingImports]

from inference_utils import base64_to_image, image_to_base64

RESULT_PREFIX = "__INFER_DONE__"


class Sam3Worker:
    def __init__(self):
        print("[sam3_worker] Loading SAM3 model...", file=sys.stderr)
        model = build_sam3_image_model()
        self.processor = Sam3Processor(model)
        print("[sam3_worker] SAM3 model loaded.", file=sys.stderr)

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    @staticmethod
    def _mask_to_rgba_b64(image, mask):
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = image[:, :, :3]
        rgba[:, :, 3] = (mask > 0).astype(np.uint8) * 255
        return image_to_base64(rgba)

    @staticmethod
    def _xyxy_pixels_to_normalized_cxcywh(boxes_xyxy, width, height):
        boxes = boxes_xyxy.astype(np.float32).copy()
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

        x1 = np.minimum(boxes[:, 0], boxes[:, 2])
        y1 = np.minimum(boxes[:, 1], boxes[:, 3])
        x2 = np.maximum(boxes[:, 0], boxes[:, 2])
        y2 = np.maximum(boxes[:, 1], boxes[:, 3])

        cx = ((x1 + x2) * 0.5) / width
        cy = ((y1 + y2) * 0.5) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        return np.stack([cx, cy, w, h], axis=1)

    def run_inference(self, input_data):
        chunk_images_b64 = input_data["chunk_images_base64"]
        prompt = input_data["prompt"]
        box_threshold = float(input_data.get("box_threshold", 0.0))

        all_masks_b64 = []
        all_boxes = []
        all_scores = []

        for idx, input_image_b64 in enumerate(chunk_images_b64):
            chunk_image = base64_to_image(input_image_b64)
            if chunk_image.ndim == 3 and chunk_image.shape[2] == 4:
                chunk_image = chunk_image[..., :3]
            chunk_image_pil = Image.fromarray(chunk_image.astype(np.uint8))
            image_h, image_w = chunk_image.shape[:2]
            inference_state = self.processor.set_image(chunk_image_pil)
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

            masks = self._to_numpy(output.get("masks"))
            boxes = self._to_numpy(output.get("boxes"))
            scores = self._to_numpy(output.get("scores"))

            has_output = (
                masks is not None
                and boxes is not None
                and scores is not None
                and masks.size > 0
                and boxes.size > 0
                and scores.size > 0
            )

            if not has_output:
                empty_rgba = np.zeros((chunk_image.shape[0], chunk_image.shape[1], 4), dtype=np.uint8)
                all_masks_b64.append(image_to_base64(empty_rgba))
                all_boxes.append([])
                all_scores.append([])
                continue

            scores_flat = scores.reshape(-1)
            boxes_array = np.asarray(boxes).reshape(-1, 4)
            keep_mask = scores_flat >= box_threshold

            if not np.any(keep_mask):
                empty_rgba = np.zeros((chunk_image.shape[0], chunk_image.shape[1], 4), dtype=np.uint8)
                all_masks_b64.append(image_to_base64(empty_rgba))
                all_boxes.append([])
                all_scores.append([])
                continue

            filtered_indices = np.where(keep_mask)[0]
            filtered_scores = scores_flat[filtered_indices]
            best_idx = int(filtered_indices[np.argmax(filtered_scores)])

            best_mask = np.asarray(masks[best_idx]).squeeze()
            filtered_boxes_xyxy = boxes_array[filtered_indices]
            filtered_boxes_cxcywh = self._xyxy_pixels_to_normalized_cxcywh(
                filtered_boxes_xyxy,
                image_w,
                image_h,
            )
            boxes_list = filtered_boxes_cxcywh.tolist()
            scores_list = [float(score) for score in filtered_scores.tolist()]

            all_masks_b64.append(self._mask_to_rgba_b64(chunk_image, best_mask))
            all_boxes.append(boxes_list)
            all_scores.append(scores_list)

        return {
            "masks_base64": all_masks_b64,
            "boxes": all_boxes,
            "scores": all_scores,
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

    worker = Sam3Worker()

    if args.persistent:
        run_persistent(worker)
        return

    if not args.input_json or not args.output_json:
        raise ValueError("Usage: python sam3_worker.py <input_json> <output_json> or --persistent")

    run_single(worker, args.input_json, args.output_json)


if __name__ == "__main__":
    main()
