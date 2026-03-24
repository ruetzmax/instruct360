import os
import sys
import numpy as np
import torch
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
import trimesh
from omegaconf import OmegaConf
from hydra.utils import instantiate
import importlib.util

def has_nvdiffrast() -> bool:
    return importlib.util.find_spec("nvdiffrast") is not None

print("nvdiffrast: ", has_nvdiffrast())


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_utils import base64_to_image, load_inference_input, save_inference_output



def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)

def has_inria_rasterizer() -> bool:
    return importlib.util.find_spec("diff_gaussian_rasterization") is not None


def apply_sam3d_runtime_patches() -> None:
    from sam3d_objects.model.backbone.tdfy_dit.renderers import gaussian_render
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils

    inria_available = has_inria_rasterizer()

    original_render = gaussian_render.render

    def render_with_backend_fallback(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=1.0,
        override_color=None,
        backend="inria",
    ):
        effective_backend = backend
        if backend == "inria" and not inria_available:
            effective_backend = "gsplat"
        return original_render(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            backend=effective_backend,
        )

    gaussian_render.render = render_with_backend_fallback

    def render_multiview_with_backend_fallback(sample, resolution=512, nviews=30):
        backend = "inria" if inria_available else "gsplat"
        radius = 2
        fov = 40
        cameras = [render_utils.sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
        yaws = [camera[0] for camera in cameras]
        pitches = [camera[1] for camera in cameras]
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitches, radius, fov
        )
        rendered = render_utils.render_frames(
            sample,
            extrinsics,
            intrinsics,
            {"resolution": resolution, "bg_color": (0, 0, 0), "backend": backend},
        )
        return rendered["color"], extrinsics, intrinsics

    render_utils.render_multiview = render_multiview_with_backend_fallback
    postprocessing_utils.render_multiview = render_multiview_with_backend_fallback


class Sam3DInference:
    def __init__(self, config_file: str, compile: bool = False):
        config = OmegaConf.load(config_file)
        config.rendering_engine = "nvdiffrast" # "pytorch3d"
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        self._pipeline = instantiate(config)

    @staticmethod
    def merge_mask_to_rgba(image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        return np.concatenate([image[..., :3], mask], axis=-1)

    def __call__(
        self,
        image,
        mask,
        seed=42,
        pointmap=None,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        use_vertex_color=True,
    ):
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            with_layout_postprocess=False,
            use_vertex_color=use_vertex_color,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )



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
sam3d_root = os.path.join(workspace_root, "sam-3d-objects")
sys.path.append(sam3d_root)
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
apply_sam3d_runtime_patches()
sam3d_model = Sam3DInference(
    os.path.join(sam3d_root, "checkpoints/hf/pipeline.yaml"),
    compile=False,
)

chunk_images_base64 = input_data["chunk_images_base64"]
chunk_masks_base64 = input_data["chunk_masks_base64"]
save_dir = input_data["save_dir"]
generate_texture = input_data.get("generate_texture", False)

if generate_texture and not has_nvdiffrast():
    raise RuntimeError("nvdiffrast required for texture generation")

#clear save dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

glb_paths = []
unposed_glb_paths = []
scales = []
rotations = []
translations = []

for idx, (chunk_image_b64, chunk_mask_b64) in enumerate(zip(chunk_images_base64, chunk_masks_base64)):
    chunk_image = base64_to_image(chunk_image_b64)
    chunk_mask = base64_to_image(chunk_mask_b64)

    chunk_mask = chunk_mask > 0  # Convert to boolean
    if len(chunk_mask.shape) > 2:
        chunk_mask = chunk_mask[..., -1]

    if len(chunk_mask.shape) > 2:
        chunk_mask = chunk_mask[..., 0]

    save_path = os.path.join(save_dir, f"reconstructed_mesh_{idx}.glb")
    unposed_save_path = os.path.join(save_dir, f"reconstructed_mesh_unposed_{idx}.glb")
    
    # # save image (numpy_array) in save dir
    # chunk_image_pil = Image.fromarray(chunk_image)
    # chunk_image_pil.save(os.path.join(save_dir, f"chunk_image_{idx}.png"))
    
    reconstruction_output = sam3d_model(
        chunk_image,
        chunk_mask,
        seed=42,
        with_mesh_postprocess=generate_texture,
        with_texture_baking=generate_texture,
        use_vertex_color=not generate_texture,
    )
    
    # Save posed mesh
    posed_glb = make_scene_untextured_mesh(reconstruction_output)
    posed_glb.export(save_path)
    glb_paths.append(save_path)
    
    # Extract and save unposed mesh (in Y-up coordinate system)
    mesh = reconstruction_output["glb"]
    if mesh is not None:
        unposed_mesh = deepcopy(mesh)
        unposed_mesh.export(unposed_save_path)
        unposed_glb_paths.append(unposed_save_path)
        
        # Extract raw model transforms without coordinate adjustments
        scale = reconstruction_output["scale"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        if scale.size != 3:
            raise ValueError(f"Unexpected scale shape: {reconstruction_output['scale'].shape}")
        scales.append(scale.tolist())

        rotation = reconstruction_output["rotation"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        if rotation.size != 4:
            raise ValueError(f"Unexpected rotation shape: {reconstruction_output['rotation'].shape}")
        rotations.append(rotation.tolist())

        translation = reconstruction_output["translation"].detach().cpu().numpy().astype(np.float32)
        if translation.shape == (1, 3):
            translation = translation[0]
        elif translation.shape == (3, 1):
            translation = translation[:, 0]
        else:
            translation = translation.reshape(-1)

        if translation.size != 3:
            raise ValueError(f"Unexpected translation shape: {reconstruction_output['translation'].shape}")
        translations.append(translation.tolist())
    
    print(f"Saved mesh {idx+1}/{len(chunk_images_base64)} to {save_path}")

output_data = {
    "glb_paths": glb_paths,
    "unposed_glb_paths": unposed_glb_paths,
    "scales": scales,
    "rotations": rotations,
    "translations": translations,
}

save_inference_output(output_data)
print(f"Generated {len(glb_paths)} meshes")

