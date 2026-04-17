import argparse
import json
import os
import sys
import traceback
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, image_to_base64

sys.path.append("ov-seg")
from open_vocab_seg.utils import VisualizationDemo  # type: ignore[reportMissingImports]
from open_vocab_seg import add_ovseg_config  # type: ignore[reportMissingImports]
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

RESULT_PREFIX = "__INFER_DONE__"


def build_model():
    print("[ovseg_worker] Loading OV-Seg model...", file=sys.stderr)
    ov_seg_cfg = get_cfg()
    add_deeplab_config(ov_seg_cfg)
    add_ovseg_config(ov_seg_cfg)
    ov_seg_cfg.merge_from_file("ov-seg/configs/ovseg_swinB_vitL_demo.yaml")
    ov_seg_cfg.MODEL.WEIGHTS = "ov-seg/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth"
    ov_seg_cfg.DATALOADER.NUM_WORKERS = 0
    if not torch.cuda.is_available():
        ov_seg_cfg.MODEL.DEVICE = "cpu"

    model = VisualizationDemo(ov_seg_cfg)
    print("[ovseg_worker] OV-Seg model loaded.", file=sys.stderr)
    return model


def run_inference(model, input_data):
    chunk_images_b64 = input_data["chunk_images_base64"]
    prompt = input_data["prompt"]

    all_masks_b64 = []

    for chunk_image_b64 in chunk_images_b64:
        chunk_image = base64_to_image(chunk_image_b64)
        predictions, _ = model.run_on_image(chunk_image, [prompt])

        if "sem_seg" not in predictions:
            rgba_image = np.zeros((chunk_image.shape[0], chunk_image.shape[1], 4), dtype=np.uint8)
            all_masks_b64.append(image_to_base64(rgba_image))
            continue

        sem_seg = predictions["sem_seg"]
        blank_area = (sem_seg[0] == 0).cpu().numpy()

        rgba_image = np.zeros((chunk_image.shape[0], chunk_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = chunk_image
        rgba_image[:, :, 3] = 255
        rgba_image[blank_area, 3] = 0

        all_masks_b64.append(image_to_base64(rgba_image))

    return {"masks_base64": all_masks_b64}


def process_request(model, input_json, output_json):
    with open(input_json, "r") as f:
        input_data = json.load(f)

    output_data = run_inference(model, input_data)

    with open(output_json, "w") as f:
        json.dump(output_data, f)


def run_persistent(model):
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
            process_request(model, input_json, output_json)
            print(RESULT_PREFIX + json.dumps({"ok": True, "output_json": output_json}), flush=True)
        except Exception:
            error_text = traceback.format_exc()
            print(RESULT_PREFIX + json.dumps({"ok": False, "error": error_text}), flush=True)


def run_single(model, input_json, output_json):
    process_request(model, input_json, output_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?")
    parser.add_argument("output_json", nargs="?")
    parser.add_argument("--persistent", action="store_true")
    args = parser.parse_args()

    model = build_model()

    if args.persistent:
        run_persistent(model)
        return

    if not args.input_json or not args.output_json:
        raise ValueError("Usage: python ovseg_worker.py <input_json> <output_json> or --persistent")

    run_single(model, args.input_json, args.output_json)


if __name__ == "__main__":
    main()
