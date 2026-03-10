import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output

input_data = load_inference_input()

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(workspace_root, "sam-3d-objects", "notebook"))
from inference import Inference
sam3d_model = Inference(os.path.join(workspace_root, "sam-3d-objects/checkpoints/hf/pipeline.yaml"), compile=False)

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

ply_paths = []
for idx, (chunk_image_b64, chunk_mask_b64) in enumerate(zip(chunk_images_base64, chunk_masks_base64)):
    chunk_image = base64_to_image(chunk_image_b64)
    chunk_mask = base64_to_image(chunk_mask_b64)
    save_path = os.path.join(save_dir, f"reconstructed_mesh_{idx}.ply")
    
    # # save image (numpy_array) in save dir
    # chunk_image_pil = Image.fromarray(chunk_image)
    # chunk_image_pil.save(os.path.join(save_dir, f"chunk_image_{idx}.png"))
    
    reconstruction_output = sam3d_model(chunk_image, chunk_mask, seed=42)
    reconstruction_output["gs"].save_ply(save_path)
    
    ply_paths.append(save_path)
    print(f"Saved point cloud {idx+1}/{len(chunk_images_base64)} to {save_path}")

output_data = {
    "ply_paths": ply_paths
}

save_inference_output(output_data)
print(f"Generated {len(ply_paths)} point clouds")

