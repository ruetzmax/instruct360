import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.append("GroundingDINO")
from groundingdino.util.inference import load_model, predict  # type: ignore[reportMissingImports]
import groundingdino.datasets.transforms as T  # type: ignore[reportMissingImports]

from inference_utils import base64_to_image

RESULT_PREFIX = "__INFER_DONE__"


def image_to_tensor(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    image_transformed, _ = transform(image, None)
    return image_transformed


def build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dino_worker] Loading GroundingDINO on {device}...", file=sys.stderr)
    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "ovmono3d/checkpoints/groundingdino_swinb_cogcoor.pth",
        device=device,
    )
    print("[dino_worker] GroundingDINO model loaded.", file=sys.stderr)
    return model, device


def run_inference(model, device, input_data):
    image_b64 = input_data["image_base64"]
    prompt = input_data["prompt"]
    box_threshold = input_data.get("box_threshold", 0.35)
    text_threshold = input_data.get("text_threshold", 0.25)

    image = base64_to_image(image_b64)
    image_tensor = image_to_tensor(image)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return {
        "boxes": boxes.cpu().numpy().tolist(),
        "logits": logits.cpu().numpy().tolist(),
        "phrases": phrases,
    }


def process_request(model, device, input_json, output_json):
    with open(input_json, "r") as f:
        input_data = json.load(f)

    output_data = run_inference(model, device, input_data)

    with open(output_json, "w") as f:
        json.dump(output_data, f)


def run_persistent(model, device):
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
            process_request(model, device, input_json, output_json)
            print(RESULT_PREFIX + json.dumps({"ok": True, "output_json": output_json}), flush=True)
        except Exception:
            error_text = traceback.format_exc()
            print(RESULT_PREFIX + json.dumps({"ok": False, "error": error_text}), flush=True)


def run_single(model, device, input_json, output_json):
    process_request(model, device, input_json, output_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", nargs="?")
    parser.add_argument("output_json", nargs="?")
    parser.add_argument("--persistent", action="store_true")
    args = parser.parse_args()

    model, device = build_model()

    if args.persistent:
        run_persistent(model, device)
        return

    if not args.input_json or not args.output_json:
        raise ValueError("Usage: python dino_worker.py <input_json> <output_json> or --persistent")

    run_single(model, device, args.input_json, args.output_json)


if __name__ == "__main__":
    main()
