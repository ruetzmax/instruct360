import os
import sys
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
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T

from inference_utils import base64_to_image, load_inference_input, save_inference_output


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

input_data = load_inference_input()

# load model
print("Loading GroundingDINO model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dino_model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
    "ovmono3d/checkpoints/groundingdino_swinb_cogcoor.pth",
    device=device
)
print("GroundingDINO model loaded.")

# run inference
image_b64 = input_data["image_base64"]
prompt = input_data["prompt"]
box_threshold = input_data.get("box_threshold", 0.35)
text_threshold = input_data.get("text_threshold", 0.25)

image = base64_to_image(image_b64)
image_tensor = image_to_tensor(image)

boxes, logits, phrases = predict(
    model=dino_model,
    image=image_tensor,
    caption=prompt,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
    device=device
)

if torch.cuda.is_available():
    torch.cuda.synchronize()

boxes_list = boxes.cpu().numpy().tolist()
output_data = {
    "boxes": boxes_list,
    "logits": logits.cpu().numpy().tolist(),
    "phrases": phrases
}

save_inference_output(output_data)
print(f"Saved {len(boxes_list)} bounding boxes")
