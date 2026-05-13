import cv2 as cv
import open3d
import numpy as np
import os

import trimesh

from src.operations2d import insv_to_equirect    

def normalize_rotation_matrix(rotation_matrix):
    rotation = np.asarray(rotation_matrix, dtype=np.float32)
    if rotation.shape == (3, 3):
        return rotation

    flat = rotation.reshape(-1)
    if flat.size != 9:
        raise ValueError(f"Expected rotation matrix with 9 values, got shape {rotation.shape}")

    return flat.reshape(3, 3)


def normalize_translation_vector(translation_vector):
    translation = np.asarray(translation_vector, dtype=np.float32)
    if translation.shape == (3,):
        return translation
    if translation.shape == (1, 3):
        return translation[0]
    if translation.shape == (3, 1):
        return translation[:, 0]

    flat = translation.reshape(-1)
    if flat.size != 3:
        raise ValueError(f"Expected translation vector with 3 values, got shape {translation.shape}")

    return flat


def normalize_scale_vector(scale_vector):
    scale = np.asarray(scale_vector, dtype=np.float32)
    if scale.shape == (3,):
        return scale
    if scale.shape == (1, 3):
        return scale[0]
    if scale.shape == (3, 1):
        return scale[:, 0]

    flat = scale.reshape(-1)
    if flat.size != 3:
        raise ValueError(f"Expected scale vector with 3 values, got shape {scale.shape}")

    return flat


def write_depth_image(depth_npy_path: str, output_path: str):
    depth = np.load(depth_npy_path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        debug_image = np.zeros(depth.shape, dtype=np.uint8)
    else:
        min_depth = float(np.min(depth[finite_mask]))
        max_depth = float(np.max(depth[finite_mask]))

        if max_depth > min_depth:
            normalized = np.zeros_like(depth, dtype=np.float32)
            normalized[finite_mask] = (depth[finite_mask] - min_depth) / (max_depth - min_depth)
            debug_image = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            debug_image = np.zeros(depth.shape, dtype=np.uint8)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(output_path, debug_image)


def read_video_frames(video_path=None, left_video_path=None, right_video_path=None):
    if not video_path:
        if not left_video_path or not right_video_path:
            raise ValueError("Either video_path or both left_video_path and right_video_path must be provided.")
        video_path = "temp/equirect_input.mp4"
        insv_to_equirect(left_video_path, right_video_path, video_path)
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open vid")
        exit()
    frames = []
    while True:
        ret, last_frame = cap.read()
        if not ret:
            break
        last_frame = cv.cvtColor(last_frame, cv.COLOR_BGR2RGB)
        frames.append(last_frame)
    return frames
    
def get_character_placeholder(scale = 0.5):
    # get open3d mesh of rectangular character placeholder
    width = 0.5 * scale
    height = 1.8 * scale
    depth = 0.5 * scale
    camera_offset = 0.5 * scale
    
    character_placeholder = open3d.geometry.OrientedBoundingBox(
        center=[0, -height / 2, camera_offset],
        R=np.eye(3),
        extent=[width, height, depth]
    )
    character_placeholder = open3d.geometry.TriangleMesh.create_from_oriented_bounding_box(character_placeholder)
    character_placeholder.paint_uniform_color([0.0, 0.0, 0.0])
    
    return character_placeholder


def read_trimesh(mesh_or_path):
    if isinstance(mesh_or_path, (str, os.PathLike)):
        mesh = trimesh.load(mesh_or_path)
    else:
        mesh = mesh_or_path

    if isinstance(mesh, trimesh.Scene):
        sub_meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not sub_meshes:
            raise ValueError("GLB scene does not contain any mesh geometry")
        textured = [
            g for g in sub_meshes
            if getattr(getattr(g.visual, "material", None), "image", None) is not None
            or getattr(getattr(g.visual, "material", None), "baseColorTexture", None) is not None
        ]
        return textured[0] if textured else max(sub_meshes, key=lambda g: len(g.faces))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh or trimesh.Scene, got {type(mesh)}")

    return mesh

def trimesh_to_open3d(mesh):
    mesh = read_trimesh(mesh)

    # convert into open3d space
    vertices = np.asarray(mesh.vertices) * [1, 1, -1]
    faces = np.asarray(mesh.faces)[:, [0, 2, 1]]

    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = open3d.utility.Vector3iVector(faces)

    material = getattr(mesh.visual, "material", None)
    texture_image = None
    if material is not None:
        texture_image = getattr(material, "image", None)
        if texture_image is None:
            texture_image = getattr(material, "baseColorTexture", None)
    uv = getattr(mesh.visual, "uv", None)

    if texture_image is not None and uv is not None and len(uv) == len(mesh.vertices):
        uv = np.asarray(uv, dtype=np.float64)
        uv[:, 1] = 1.0 - uv[:, 1]
        triangle_uvs = uv[faces].reshape(-1, 2)
        o3d_mesh.triangle_uvs = open3d.utility.Vector2dVector(triangle_uvs)

        texture_np = np.asarray(texture_image)
        if texture_np.ndim == 2:
            texture_np = np.stack([texture_np, texture_np, texture_np], axis=-1)
        if texture_np.dtype != np.uint8:
            texture_np = np.clip(texture_np, 0, 255).astype(np.uint8)

        o3d_mesh.textures = [open3d.geometry.Image(texture_np)]
        o3d_mesh.triangle_material_ids = open3d.utility.IntVector(
            np.zeros(len(mesh.faces), dtype=np.int32)
        )

    vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if not o3d_mesh.has_textures() and vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        vertex_colors = np.asarray(vertex_colors)[:, :3].astype(np.float64)
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
        o3d_mesh.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)

    o3d_mesh.compute_vertex_normals()
                                    
    return o3d_mesh
    
