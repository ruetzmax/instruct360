import os
import numpy as np
import tempfile

import open3d
import trimesh
from src.operations2d import ImageChunk, get_masks_from_image_chunks
from src.util import normalize_rotation_matrix, normalize_translation_vector, normalize_scale_vector, read_trimesh
from src.inference.conda_inference import CondaInferenceRunner
from src.inference.inference_utils import image_to_base64
import torch
from transformers import pipeline


_ovmono_runner = None
_sam3d_runner = None
_foundationpose_runner = None

moge_model = None
da_model = None


def _get_ovmono_runner(env_name="ovmono3d"):
    global _ovmono_runner
    if _ovmono_runner is None:
        _ovmono_runner = CondaInferenceRunner(env_name, "ovmono_inference.py")
    return _ovmono_runner


def _get_sam3d_runner(env_name="sam3d-objects"):
    global _sam3d_runner
    if _sam3d_runner is None:
        _sam3d_runner = CondaInferenceRunner(env_name, "sam3d_inference.py")
    return _sam3d_runner


def _get_foundationpose_runner(env_name="instruct360"):
    global _foundationpose_runner
    if _foundationpose_runner is None:
        _foundationpose_runner = CondaInferenceRunner(env_name, "foundationpose_inference.py")
    return _foundationpose_runner
        

def get_intrinsics_for_chunk(chunk: ImageChunk):
    fov_x, fov_y = chunk.fov
    h, w, _ = chunk.image.shape
    focal_length_x = (w / 2) / np.tan(np.radians(fov_x) / 2)
    focal_length_y = (h / 2) / np.tan(np.radians(fov_y) / 2)
    
    principal_point = (w / 2, h / 2)

    K = np.array([
        [focal_length_x, 0.0, principal_point[0]], 
        [0.0, focal_length_y, principal_point[1]], 
        [0.0, 0.0, 1.0]
    ])
    
    return K

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
    is_first_frame,
    foundationpose_env="foundationpose",
):
    global da_model

    if da_model is None:
        da_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
        
    depth = da_model(chunk.image)["depth"]

    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map from MoGe, got shape {depth.shape}")

    os.makedirs("temp", exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", prefix="moge_depth_", dir="temp", delete=False) as depth_file:
        depth_npy_path = depth_file.name
        np.save(depth_file, depth.astype(np.float32))

    mask_base64 = None
    if is_first_frame:
        first_frame_mask = get_masks_from_image_chunks([chunk], prompt=class_name, use_gpu=True)[0]
        mask_base64 = image_to_base64(first_frame_mask)

    runner = _get_foundationpose_runner(foundationpose_env)
    input_data = {
        "is_first_frame": is_first_frame,
        "rgb_base64": image_to_base64(chunk.image),
        "depth_npy_path": depth_npy_path,
        "mask_base64": mask_base64,
        "unposed_mesh_path": str(unposed_mesh_path),
        "intrinsics": K.tolist(),
    }
    try:
        output_data = runner.run(input_data)
    finally:
        if os.path.exists(depth_npy_path):
            os.remove(depth_npy_path)

    pose = np.array(output_data["pose"], dtype=np.float32)
    center_pose = np.array(output_data["center_pose"], dtype=np.float32)

    estimated_rotation = pose[:3, :3]
    estimated_translation = pose[:3, 3]
    center_rotation = center_pose[:3, :3]
    center_translation = center_pose[:3, 3]

    return estimated_rotation, estimated_translation


def get_3d_bounding_boxes(chunk: ImageChunk, prompt: str, threshold=0.3, ovmono_env="ovmono3d"):
    runner = _get_ovmono_runner(ovmono_env)
    
    h, w, _ = chunk.image.shape
    K = get_intrinsics_for_chunk(chunk)
    
    input_data = {
        "image_base64": image_to_base64(chunk.image),
        "prompt": prompt,
        "threshold": threshold,
        "intrinsics": K.tolist(),
        "height": h,
        "width": w
    }
    
    output_data = runner.run(input_data)
    
    centers = [np.array(c) for c in output_data["centers"]]
    dimensions = [np.array(d) for d in output_data["dimensions"]]
    poses = [np.array(p) for p in output_data["poses"]]
    
    return centers, dimensions, poses
        
def adjust_bounding_boxes_by_chunk_rotation(centers, poses, chunk: ImageChunk):
    rotated_centers = []
    rotated_poses = []
    
    angle_horizontal_rad, angle_vertical_rad = chunk.angle
    
    # horizontal angle rotates around Y-axis
    rotation_yaw = np.array([
        [np.cos(angle_horizontal_rad), 0, np.sin(angle_horizontal_rad)],
        [0, 1, 0],
        [-np.sin(angle_horizontal_rad), 0, np.cos(angle_horizontal_rad)]
    ])
    
    # vertical angle rotates around X-axis
    rotation_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(-angle_vertical_rad), -np.sin(-angle_vertical_rad)],
        [0, np.sin(-angle_vertical_rad), np.cos(-angle_vertical_rad)]
    ])
    
    rotation = rotation_pitch @ rotation_yaw
    
    for center in centers:
        rotated_center = rotation @ center.T
        rotated_centers.append(rotated_center)
    
    for pose in poses:
        pose_array = np.array(pose).reshape(3, 3)
        rotated_pose = rotation @ pose_array
        rotated_poses.append(rotated_pose)
    
    return rotated_centers, rotated_poses

def adjust_transforms_by_chunk_rotation(
    rotation_matrices,
    translation_vectors,
    chunk: ImageChunk,
    invert: bool = False
):
    angle_horizontal_rad, angle_vertical_rad = chunk.angle

    # horizontal angle rotates around Y-axis
    rotation_yaw = np.array([
        [np.cos(angle_horizontal_rad), 0, np.sin(angle_horizontal_rad)],
        [0, 1, 0],
        [-np.sin(angle_horizontal_rad), 0, np.cos(angle_horizontal_rad)]
    ])
    
    # vertical angle rotates around X-axis
    rotation_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(angle_vertical_rad), -np.sin(angle_vertical_rad)],
        [0, np.sin(angle_vertical_rad), np.cos(angle_vertical_rad)]
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

def adjust_transforms_between_cameras(
        rotation_matrices,
        translation_vectors,
        cam1_translation,
        cam1_rotation,
        cam2_translation,
        cam2_rotation,
):
        # camera poses in world frame
        R_w_c1 = open3d.geometry.get_rotation_matrix_from_quaternion(cam1_rotation)
        R_w_c2 = open3d.geometry.get_rotation_matrix_from_quaternion(cam2_rotation)
        t_w_c1 = normalize_translation_vector(cam1_translation)
        t_w_c2 = normalize_translation_vector(cam2_translation)

        R_c2_w = R_w_c2.T

        adjusted_rotations = []
        adjusted_translations = []

        for rotation_matrix, translation_vector in zip(rotation_matrices, translation_vectors):
                R_c1_o = normalize_rotation_matrix(rotation_matrix)
                t_c1_o = normalize_translation_vector(translation_vector)

                # object pose in world frame
                R_w_o = R_w_c1 @ R_c1_o
                t_w_o = R_w_c1 @ t_c1_o + t_w_c1

                # object pose in camera 2 frame
                R_c2_o = R_c2_w @ R_w_o
                t_c2_o = R_c2_w @ (t_w_o - t_w_c2)

                adjusted_rotations.append(R_c2_o)
                adjusted_translations.append(t_c2_o)

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
    
def _create_box_mesh(center, dimensions, pose, color=(0, 0, 255)):
    # create opencv box mesh
    center = np.array(center).flatten()
    dimensions = np.array(dimensions).flatten()
    pose_matrix = np.array(pose).reshape(3, 3)
    
    w, h, d = dimensions
    vertices_local = np.array([
        [-w/2, -h/2, -d/2],  # 0: front-bottom-left
        [ w/2, -h/2, -d/2],  # 1: front-bottom-right
        [ w/2,  h/2, -d/2],  # 2: front-top-right
        [-w/2,  h/2, -d/2],  # 3: front-top-left
        [-w/2, -h/2,  d/2],  # 4: back-bottom-left
        [ w/2, -h/2,  d/2],  # 5: back-bottom-right
        [ w/2,  h/2,  d/2],  # 6: back-top-right
        [-w/2,  h/2,  d/2],  # 7: back-top-left
    ])
    
    vertices = (pose_matrix @ vertices_local.T).T + center
    
    # flip x, y and z to match Open3D coordinate system
    vertices = vertices * [-1, -1, -1]
    
    triangles = np.array([
        # front face
        [0, 1, 2], [0, 2, 3],
        # back face
        [4, 6, 5], [4, 7, 6],
        # left face
        [0, 3, 7], [0, 7, 4],
        # right face
        [1, 5, 6], [1, 6, 2],
        # bottom face
        [0, 4, 5], [0, 5, 1],
        # top face
        [3, 2, 6], [3, 6, 7],
    ])
    
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(triangles)
    
    color_normalized = np.array(color) / 255.0 if max(color) > 1.0 else np.array(color)
    mesh.paint_uniform_color(color_normalized)
    
    mesh.compute_vertex_normals()
    
    return mesh

def get_box_mesh(box, color=(0, 0, 255)):
    center, dimension, pose = box
    return _create_box_mesh(center, dimension, pose, color=color)

def get_box_meshes(boxes, color=(0, 0, 255)):
    centers, dimensions, poses = boxes
    meshes = []
    for box_idx in range(len(centers)):
        center = centers[box_idx]
        dimension = dimensions[box_idx]
        pose = poses[box_idx]
        
        mesh = _create_box_mesh(center, dimension, pose, color=color)
        meshes.append(mesh)
    return meshes

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

def pose_trimesh_with_sam3d_transform(mesh, rotation_input, translation_vector, scale_vector=None):

    # Load mesh and ensure we have a Trimesh instance
    base_mesh = read_trimesh(mesh)
    if not isinstance(base_mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(base_mesh)}")

    # Axis conversion matrices between Z-up and Y-up
    _R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    _R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

    # Convert mesh vertices from Y-up to Z-up
    vertices_yup = np.asarray(base_mesh.vertices, dtype=np.float32)
    vertices_zup = vertices_yup @ _R_YUP_TO_ZUP

    # Normalize transform inputs (Z-up space)
    if rotation_input is None:
        rotation_zup = np.eye(3, dtype=np.float32)
    else:
        rotation = np.asarray(rotation_input, dtype=np.float32)
        flat = rotation.reshape(-1)
        if flat.size == 4:
            # Quaternion (w, x, y, z)
            rotation_zup = open3d.geometry.get_rotation_matrix_from_quaternion(flat.tolist())
        else:
            rotation_zup = normalize_rotation_matrix(rotation)

    translation_zup = normalize_translation_vector(translation_vector)

    if scale_vector is None:
        scale = np.ones(3, dtype=np.float32)
    else:
        scale = normalize_scale_vector(scale_vector)

    # Apply scale, then rotation, then translation in Z-up
    vertices_zup_scaled = vertices_zup * scale[None, :]
    vertices_zup_rot = (rotation_zup @ vertices_zup_scaled.T).T
    vertices_zup_posed = vertices_zup_rot + translation_zup[None, :]

    # Convert posed vertices back to Y-up
    vertices_yup_posed = vertices_zup_posed @ _R_ZUP_TO_YUP

    posed_mesh = base_mesh.copy()
    posed_mesh.vertices = vertices_yup_posed.astype(np.float32)

    return posed_mesh