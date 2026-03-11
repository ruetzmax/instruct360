import os
import sys
import numpy as np
import torch
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
import trimesh


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output

def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


# https://github.com/facebookresearch/sam-3d-objects/issues/56
_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T
def make_scene_untextured_mesh(*outputs, in_place=False):

    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_meshes = []
    for output in outputs:
        mesh = output["glb"]
        if mesh is None:
            continue

        # GLB is Y-up, transforms are Z-up; convert, apply, convert back
        vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
        R_l2c = quaternion_to_matrix(output["rotation"])
        l2c_transform = compose_transform(
            scale=output["scale"],
            rotation=R_l2c,
            translation=output["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        mesh.vertices = vertices.squeeze(0).cpu().numpy() @ _R_ZUP_TO_YUP
        all_meshes.append(mesh)

    if not all_meshes:
        return None

    if len(all_meshes) == 1:
        return all_meshes[0]

    return trimesh.util.concatenate(all_meshes)

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

glb_paths = []
for idx, (chunk_image_b64, chunk_mask_b64) in enumerate(zip(chunk_images_base64, chunk_masks_base64)):
    chunk_image = base64_to_image(chunk_image_b64)
    chunk_mask = base64_to_image(chunk_mask_b64)

    if len(chunk_mask.shape) > 2:
        chunk_mask = chunk_mask[..., 0]

    save_path = os.path.join(save_dir, f"reconstructed_mesh_{idx}.glb")
    
    # # save image (numpy_array) in save dir
    # chunk_image_pil = Image.fromarray(chunk_image)
    # chunk_image_pil.save(os.path.join(save_dir, f"chunk_image_{idx}.png"))
    
    reconstruction_output = sam3d_model(chunk_image, chunk_mask, seed=42)
    posed_glb = make_scene_untextured_mesh(reconstruction_output)
    posed_glb.export(save_path)
    # reconstruction_output["gs"].save_ply(save_path)
    
    glb_paths.append(save_path)
    print(f"Saved mesh {idx+1}/{len(chunk_images_base64)} to {save_path}")

output_data = {
    "glb_paths": glb_paths
}

save_inference_output(output_data)
print(f"Generated {len(glb_paths)} meshes")

