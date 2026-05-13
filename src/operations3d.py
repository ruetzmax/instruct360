import os
import numpy as np
import tempfile

import open3d
import trimesh
from src.operations2d import ImageChunk, get_masks_from_image_chunks
from src.util import normalize_rotation_matrix, normalize_translation_vector, normalize_scale_vector, read_trimesh, write_depth_image
from src.inference.conda_inference import CondaInferenceRunner, ThreadedCondaInferenceRunner
from src.inference.inference_utils import image_to_base64
import torch


_sam3d_runner = None
_foundationpose_runner = None
_depth_pro_runner = None

moge_model = None

def _get_sam3d_runner(env_name="sam3d-objects"):
    global _sam3d_runner
    if _sam3d_runner is None:
        _sam3d_runner = CondaInferenceRunner(env_name, "sam3d_inference.py")
    return _sam3d_runner


def _get_foundationpose_runner(env_name="instruct360"):
    global _foundationpose_runner
    if _foundationpose_runner is None:
        _foundationpose_runner = ThreadedCondaInferenceRunner(env_name, "foundationpose_worker.py")
    return _foundationpose_runner

def _get_depth_pro_runner(env_name="instruct360"):
    global _depth_pro_runner
    if _depth_pro_runner is None:
        _depth_pro_runner = CondaInferenceRunner(env_name, "depth-pro_worker.py")
    return _depth_pro_runner

#https://github.com/facebookresearch/sam-3d-objects/issues/144#issuecomment-3835610725
def estimate_intrinsics_for_chunk(chunk: ImageChunk):
    global moge_model
    
    from moge.model.v1 import MoGeModel  # type: ignore[reportMissingImports]

    image_tensor = (
        torch.from_numpy(np.array(chunk.image)).float().permute(2, 0, 1) / 255.0
    ).to("cpu")

    if moge_model is None:
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cpu")
        
    moge_model.eval()
    with torch.no_grad():
        moge_output = moge_model.infer(image_tensor)
        
    intrinsics = moge_output["intrinsics"].cpu().numpy()
    
    cx_norm, cy_norm = intrinsics[0, 2], intrinsics[1, 2]
    fx_norm, fy_norm = intrinsics[0, 0], intrinsics[1, 1]

    h, w, _ = chunk.image.shape
    fx_abs = fx_norm * w
    fy_abs = fy_norm * h
    cx_abs = cx_norm * w
    cy_abs = cy_norm * h
    fx_abs = fy_abs

    K = np.array([[fx_abs, 0.0, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]])
    return K


def estimate_pose_for_image_chunk(
    chunk: ImageChunk,
    unposed_mesh_path,
    class_name,
    K,
    foundationpose_env="foundationpose",
    depth_pro_env="depth-pro",
    depth_debug_image_path=None,
    foundationpose_debug_image_path=None,
    foundationpose_debug_level=1,
):
    os.makedirs("temp", exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", prefix="moge_depth_", dir="temp", delete=False) as depth_file:
        depth_npy_path = depth_file.name

    image_base64 = image_to_base64(chunk.image)

    depth_runner = _get_depth_pro_runner(depth_pro_env)
    fx_px = float(K[0, 0])
    fy_px = float(K[1, 1])
    f_px = 0.5 * (fx_px + fy_px)
    depth_input_data = {
        "image_base64": image_base64,
        "depth_npy_path": depth_npy_path,
        "f_px": f_px,
    }
    depth_output_data = depth_runner.run(depth_input_data)

    if depth_debug_image_path:
        depth_debug = depth_output_data.get("depth_debug")
        if isinstance(depth_debug, dict):
            if depth_debug.get("has_finite"):
                print(
                    "[depth] "
                    f"units={depth_output_data.get('depth_units', 'm')} "
                    f"shape={tuple(depth_debug.get('shape', []))} "
                    f"min={depth_debug.get('min', 0.0):.4f} "
                    f"max={depth_debug.get('max', 0.0):.4f} "
                    f"mean={depth_debug.get('mean', 0.0):.4f} "
                    f"p50={depth_debug.get('p50', 0.0):.4f} "
                    f"center={depth_debug.get('center', 0.0):.4f}"
                )
            else:
                print(f"[depth] units={depth_output_data.get('depth_units', 'm')} no finite values")  
        try:
            write_depth_image(depth_npy_path, depth_debug_image_path)
        except Exception as exc:
            print(f"[depth] failed to write debug image to {depth_debug_image_path}: {exc}")
    
    frame_mask = get_masks_from_image_chunks([chunk], prompt=class_name)[0]
    mask_base64 = image_to_base64(frame_mask)

    runner = _get_foundationpose_runner(foundationpose_env)
    input_data = {
        "is_first_frame": True,
        "rgb_base64": image_base64,
        "depth_npy_path": depth_npy_path,
        "mask_base64": mask_base64,
        "unposed_mesh_path": str(unposed_mesh_path),
        "intrinsics": K.tolist(),
    }
    if foundationpose_debug_image_path:
        input_data["debug_image_path"] = foundationpose_debug_image_path
        input_data["debug_dir"] = str(os.path.dirname(foundationpose_debug_image_path))
        input_data["debug"] = int(foundationpose_debug_level)

    try:
        output_data = runner.run(input_data)
    finally:
        if os.path.exists(depth_npy_path):
            os.remove(depth_npy_path)

    pose = np.array(output_data["pose"], dtype=np.float32)

    estimated_rotation = pose[:3, :3]
    estimated_translation = pose[:3, 3]

    return estimated_rotation, estimated_translation

def adjust_transforms_by_chunk_rotation(
    rotation_matrices,
    translation_vectors,
    chunk: ImageChunk,
    invert: bool = False
):
    angle_horizontal_rad, angle_vertical_rad = chunk.angle

    # horizontal angle rotates around Z-axis
    rotation_yaw = np.array([
        [np.cos(angle_horizontal_rad), -np.sin(angle_horizontal_rad), 0],
        [np.sin(angle_horizontal_rad), np.cos(angle_horizontal_rad), 0],
        [0, 0, 1]
    ])
    
    # vertical angle rotates around X-axis (inverted direction)
    rotation_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(-angle_vertical_rad), -np.sin(-angle_vertical_rad)],
        [0, np.sin(-angle_vertical_rad), np.cos(-angle_vertical_rad)]
    ])
    
    chunk_rotation = rotation_pitch @ rotation_yaw

    if invert:
        chunk_rotation = chunk_rotation.T
    
    adjusted_rotations = []
    adjusted_translations = []
    
    for rotation_matrix in rotation_matrices:
        normalized_rotation = normalize_rotation_matrix(rotation_matrix)
        adjusted_rotation = chunk_rotation @ normalized_rotation
        adjusted_rotations.append(adjusted_rotation)
    
    for translation_vector in translation_vectors:
        normalized_translation = normalize_translation_vector(translation_vector)
        adjusted_translation = chunk_rotation @ normalized_translation
        adjusted_translations.append(adjusted_translation)
    
    return adjusted_rotations, adjusted_translations

def adjust_pose_by_camera_pose(geometry, camera_translation, camera_rotation):
    # create copy
    if isinstance(geometry, open3d.geometry.TriangleMesh):
        adjusted_geometry = open3d.geometry.TriangleMesh(geometry)
    elif isinstance(geometry, open3d.geometry.PointCloud):
        adjusted_geometry = open3d.geometry.PointCloud(geometry)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")
        
    # rotate opencv geometry by camera rotation
    rot_mat = open3d.geometry.get_rotation_matrix_from_quaternion(camera_rotation)
    rot_180_z = open3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi])
    rot_mat = rot_180_z @ rot_mat
    adjusted_geometry.rotate(rot_mat, center=[0, 0, 0])
    
    # translate opencv geometry by camera translation
    camera_translation = [camera_translation[2], camera_translation[1], camera_translation[0]]
    adjusted_geometry.translate(camera_translation)
    
    return adjusted_geometry

def reconstruct_meshes_for_chunks(
    chunks,
    masks,
    sam3d_env="sam3d-objects",
    generate_texture=True,
):
    runner = _get_sam3d_runner(sam3d_env)
    
    save_dir = "temp/sam3d_output/"
    input_data = {
        "chunk_images_base64": [image_to_base64(chunk.image) for chunk in chunks],
        "chunk_masks_base64": [image_to_base64(mask) for mask in masks],
        "save_dir": save_dir,
        "generate_texture": generate_texture,
    }
    
    output_data = runner.run(input_data)
    
    meshes = []
    unposed_meshes = []
    scales = []
    rotations = []
    translations = []
    
    for glb_path in output_data["glb_paths"]:
        trimesh_mesh = read_trimesh(glb_path)
        meshes.append(trimesh_mesh)
    
    for unposed_glb_path in output_data["unposed_glb_paths"]:
        trimesh_mesh = read_trimesh(unposed_glb_path)
        unposed_meshes.append(trimesh_mesh)
    
    for scale_vector in output_data.get("scales", []):
        scales.append(np.array(scale_vector))
    
    for rotation_quaternion in output_data.get("rotations", []):
        rotations.append(np.array(rotation_quaternion))

    for translation_vector in output_data.get("translations", []):
        translations.append(np.array(translation_vector))

    if not scales and rotations:
        scales = [np.ones(3, dtype=np.float32) for _ in rotations]
    
    return meshes, unposed_meshes, scales, rotations, translations

def apply_mesh_transforms(unposed_mesh, rotation_matrix, translation_vector, scale_vector=None):
    if not isinstance(unposed_mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise TypeError(f"Expected trimesh.Trimesh or trimesh.Scene, got {type(unposed_mesh)}")

    normalized_rotation = normalize_rotation_matrix(rotation_matrix)
    normalized_translation = normalize_translation_vector(translation_vector)
    normalized_scale = np.ones(3, dtype=np.float32) if scale_vector is None else normalize_scale_vector(scale_vector)
    
    transformed_mesh = unposed_mesh.copy()
    
    # create 4x4 transformation matrix
    transform_4x4 = np.eye(4, dtype=np.float32)
    scale_matrix = np.diag(normalized_scale)
    transform_4x4[:3, :3] = normalized_rotation @ scale_matrix
    transform_4x4[:3, 3] = normalized_translation
    
    transformed_mesh.apply_transform(transform_4x4)
    
    return transformed_mesh

# converts transforms from sam3d coordinate system (RH z-up) to GLTF system (x left, y up, z forward)
def sam3d_transforms_to_trimesh(scale_vector, rotation_input, translation_vector):
    _R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    _R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

    scale = normalize_scale_vector(scale_vector)
    rotation = np.asarray(rotation_input, dtype=np.float32)
    if rotation.reshape(-1).size == 4:
        # rotation is a quaternion (w, x, y, z) in z-up coordinates
        rotation_matrix_zup = open3d.geometry.get_rotation_matrix_from_quaternion(rotation.reshape(-1).tolist())
    else:
        # rotation is already a 3x3 matrix in z-up coordinates
        rotation_matrix_zup = normalize_rotation_matrix(rotation)

    translation_zup = normalize_translation_vector(translation_vector)

    # Convert z-up transforms into GLTF/trimesh (y-up) coordinates
    adjusted_rotation = _R_ZUP_TO_YUP @ rotation_matrix_zup @ _R_YUP_TO_ZUP
    adjusted_translation = _R_ZUP_TO_YUP @ translation_zup

    return scale, adjusted_rotation, adjusted_translation