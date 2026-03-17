import sys
import numpy as np
from PIL import Image

from inference_utils import (
    base64_to_image,
    image_to_base64,
    load_inference_input,
    save_inference_output,
)

sys.path.append("sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

input_data = load_inference_input()

# load sam3
print("Loading SAM3 model...")
model = build_sam3_image_model()
processor = Sam3Processor(model)

chunk_images_b64 = input_data["chunk_images_base64"]
prompt = input_data["prompt"]
box_threshold = float(input_data.get("box_threshold", 0.0))

all_masks_b64 = []
all_boxes = []
all_scores = []


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _mask_to_rgba_b64(image, mask):
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image[:, :, :3]
    rgba[:, :, 3] = (mask > 0).astype(np.uint8) * 255
    return image_to_base64(rgba)


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

for idx, input_image_b64 in enumerate(chunk_images_b64):
    chunk_image = base64_to_image(input_image_b64)
    if chunk_image.ndim == 3 and chunk_image.shape[2] == 4:
        chunk_image = chunk_image[..., :3]
    chunk_image_pil = chunk_image.astype(np.uint8)
    chunk_image_pil = Image.fromarray(chunk_image_pil)
    image_h, image_w = chunk_image.shape[:2]
    inference_state = processor.set_image(chunk_image_pil)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = _to_numpy(output.get("masks"))
    boxes = _to_numpy(output.get("boxes"))
    scores = _to_numpy(output.get("scores"))

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
        print(f"Processed chunk {idx+1}/{len(chunk_images_b64)} (no detections)")
        continue

    scores_flat = scores.reshape(-1)
    boxes_array = np.asarray(boxes).reshape(-1, 4)
    keep_mask = scores_flat >= box_threshold

    if not np.any(keep_mask):
        empty_rgba = np.zeros((chunk_image.shape[0], chunk_image.shape[1], 4), dtype=np.uint8)
        all_masks_b64.append(image_to_base64(empty_rgba))
        all_boxes.append([])
        all_scores.append([])
        print(f"Processed chunk {idx+1}/{len(chunk_images_b64)} (no detections after threshold)")
        continue

    filtered_indices = np.where(keep_mask)[0]
    filtered_scores = scores_flat[filtered_indices]
    best_idx = int(filtered_indices[np.argmax(filtered_scores)])

    best_mask = np.asarray(masks[best_idx]).squeeze()
    filtered_boxes_xyxy = boxes_array[filtered_indices]
    filtered_boxes_cxcywh = _xyxy_pixels_to_normalized_cxcywh(
        filtered_boxes_xyxy,
        image_w,
        image_h,
    )
    boxes_list = filtered_boxes_cxcywh.tolist()
    scores_list = [float(score) for score in filtered_scores.tolist()]

    all_masks_b64.append(_mask_to_rgba_b64(chunk_image, best_mask))
    all_boxes.append(boxes_list)
    all_scores.append(scores_list)
    print(f"Processed chunk {idx+1}/{len(chunk_images_b64)}")


output_data = {
    "masks_base64": all_masks_b64,
    "boxes": all_boxes,
    "scores": all_scores,
}

save_inference_output(output_data)
print(f"Saved {len(all_masks_b64)} masks and boxes")

