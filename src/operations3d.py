import os
import numpy as np

import open3d
from src.operations2d import ImageChunk
from src.inference.conda_inference import CondaInferenceRunner
from src.inference.inference_utils import image_to_base64


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

def adjust_mesh_by_chunk_rotation(mesh, chunk: ImageChunk, scale=1.0):
    rotated_mesh = open3d.geometry.TriangleMesh(mesh)
    
    angle_horizontal_rad, angle_vertical_rad = chunk.angle
    
    # horizontal angle rotates around Y-axis (inverted)
    rotation_yaw = np.array([
        [np.cos(-angle_horizontal_rad), 0, np.sin(-angle_horizontal_rad)],
        [0, 1, 0],
        [-np.sin(-angle_horizontal_rad), 0, np.cos(-angle_horizontal_rad)]
    ])
    
    # vertical angle rotates around X-axis
    rotation_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(-angle_vertical_rad), -np.sin(-angle_vertical_rad)],
        [0, np.sin(-angle_vertical_rad), np.cos(-angle_vertical_rad)]
    ])
    
    rotation = rotation_pitch @ rotation_yaw
    
    rotated_mesh.rotate(rotation, center=[0, 0, 0])
    rotated_mesh.scale(scale, center=[0, 0, 0])
    
    return rotated_mesh

def adjust_meshes_by_chunk_rotation(meshes, chunks, scale=1.0):
    adjusted_meshes = []
    for mesh, chunk in zip(meshes, chunks):
        adjusted_mesh = adjust_mesh_by_chunk_rotation(mesh, chunk, scale=scale)
        adjusted_meshes.append(adjusted_mesh)
    return adjusted_meshes

# def adjust_pointcloud_by_chunk_rotation(pointcloud, chunk: ImageChunk):
#     rotated_pointcloud = open3d.geometry.PointCloud(pointcloud)
    
#     angle_horizontal_rad, angle_vertical_rad = chunk.angle
    
#     # horizontal angle rotates around Y-axis
#     rotation_yaw = np.array([
#         [np.cos(-angle_horizontal_rad), 0, np.sin(-angle_horizontal_rad)],
#         [0, 1, 0],
#         [-np.sin(-angle_horizontal_rad), 0, np.cos(-angle_horizontal_rad)]
#     ])
    
#     # vertical angle rotates around X-axis
#     rotation_pitch = np.array([
#         [1, 0, 0],
#         [0, np.cos(-angle_vertical_rad), -np.sin(-angle_vertical_rad)],
#         [0, np.sin(-angle_vertical_rad), np.cos(-angle_vertical_rad)]
#     ])
    
#     rotation = rotation_pitch @ rotation_yaw
    
#     rotated_pointcloud.rotate(rotation, center=[0, 0, 0])
    
#     return rotated_pointcloud

# def adjust_pointclouds_by_chunk_rotation(pointclouds, chunks):
#     adjusted_pointclouds = []
#     for pointcloud, chunk in zip(pointclouds, chunks):
#         adjusted_pointcloud = adjust_pointcloud_by_chunk_rotation(pointcloud, chunk)
#         adjusted_pointclouds.append(adjusted_pointcloud)
#     return adjusted_pointclouds

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
    for glb_path in output_data["glb_paths"]:
        mesh = open3d.io.read_triangle_mesh(glb_path)
        meshes.append(mesh)
    
    return meshes
        