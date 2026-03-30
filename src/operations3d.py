import os
import numpy as np

import open3d
import trimesh
from src.operations2d import ImageChunk
from src.util import normalize_rotation_matrix, normalize_translation_vector, normalize_scale_vector, read_trimesh
from src.inference.conda_inference import CondaInferenceRunner
from src.inference.inference_utils import image_to_base64
import torch


_ovmono_runner = None
_sam3d_runner = None


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
    from moge.model.v1 import MoGeModel

    image_tensor = (
        torch.from_numpy(np.array(chunk.image)).float().permute(2, 0, 1) / 255.0
    )
    image_tensor = image_tensor.to("cpu")
    
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
        [0, np.cos(-angle_vertical_rad), -np.sin(-angle_vertical_rad)],
        [0, np.sin(-angle_vertical_rad), np.cos(-angle_vertical_rad)]
    ])
    
    chunk_rotation = rotation_pitch @ rotation_yaw
    
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

def sam3d_transforms_to_trimesh(scale_vector, rotation_input, translation_vector):
    _R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    _R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

    scale = normalize_scale_vector(scale_vector)
    rotation = np.asarray(rotation_input, dtype=np.float32)
    if rotation.reshape(-1).size == 4:
        rotation_matrix_zup = open3d.geometry.get_rotation_matrix_from_quaternion(rotation.reshape(-1).tolist())
    else:
        rotation_matrix_zup = normalize_rotation_matrix(rotation)

    rotation_matrix_zup = rotation_matrix_zup.T

    translation_zup = normalize_translation_vector(translation_vector)

    adjusted_rotation = _R_ZUP_TO_YUP @ rotation_matrix_zup @ _R_YUP_TO_ZUP
    adjusted_translation = _R_ZUP_TO_YUP @ translation_zup

    mirror_z_open3d = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    adjusted_rotation = adjusted_rotation @ mirror_z_open3d

    return scale, adjusted_rotation, adjusted_translation