import base64
import os
import sys, json
import numpy as np
import cv2


def base64_to_image(img_str):
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Read input data from JSON file path
input_json_path = sys.argv[1]
with open(input_json_path, 'r') as f:
    input_data = json.load(f)

sys.path.append("sam-3d-objects")
from inference import Inference
sam3d_model = Inference("sam-3d-objects/checkpoints/hf/pipeline.yaml", compile=False)

chunk_images_base64 = input_data["chunk_images_base64"]
chunk_masks_base64 = input_data["chunk_masks_base64"]
save_dir = input_data["save_dir"]

#clear save dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

for idx, (chunk_image_b64, chunk_mask_b64) in enumerate(zip(chunk_images_base64, chunk_masks_base64)):
    chunk_image = base64_to_image(chunk_image_b64)
    chunk_mask = base64_to_image(chunk_mask_b64)
    save_path = os.path.join(save_dir, f"reconstructed_mesh_{idx}.ply")
    
    # # save image (numpy_array) in save dir
    # chunk_image_pil = Image.fromarray(chunk_image)
    # chunk_image_pil.save(os.path.join(save_dir, f"chunk_image_{idx}.png"))
    
    reconstruction_output = sam3d_model(chunk_image, chunk_mask, seed=42)
    reconstruction_output["gs"].save_ply(save_path)

