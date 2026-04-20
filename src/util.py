import cv2 as cv
import matplotlib.pyplot as plt
import open3d
import torch
from torchvision.ops import box_convert
import numpy as np
import base64
import sys
import os

import trimesh

from src.operations2d import ImageChunk, insv_to_equirect    

from open3d.visualization import draw_geometries

import plotly.graph_objects as go



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

def draw_bounding_box(image, box, color=(255, 0, 0), thickness=2):
    #convert [0,1] (cx,cy,w,h) -> pixel (x1,y1,x2,y2)
    h, w, _ = image.shape
    box = box * [w, h, w, h]
    box_tensor = torch.tensor(box) if not isinstance(box, torch.Tensor) else box
    box = box_convert(box_tensor, in_fmt="cxcywh", out_fmt="xyxy")
    
    x1, y1, x2, y2 = map(int, box)
    image_with_box = image.copy()
    cv.rectangle(image_with_box, (x1, y1), (x2, y2), color, thickness)
    return image_with_box

def draw_bounding_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    image_with_boxes = image.copy()
    for box in boxes:
        image_with_boxes = draw_bounding_box(image_with_boxes, box, color=color, thickness=thickness)
    return image_with_boxes

def draw_3d_bounding_boxes(chunk: ImageChunk, centers, dimensions, poses):
    
    ovmono_path = os.path.join(os.getcwd(), 'ovmono3d')
    if ovmono_path not in sys.path:
        sys.path.insert(0, ovmono_path)

    from ovmono3d.cubercnn import util, vis
    from src.operations3d import get_intrinsics_for_chunk
    
    boxes = []
    for bb_idx in range(len(centers)):
        center = centers[bb_idx].flatten().tolist() if isinstance(centers[bb_idx], np.ndarray) else list(centers[bb_idx])
        dimension = dimensions[bb_idx].flatten().tolist() if isinstance(dimensions[bb_idx], np.ndarray) else list(dimensions[bb_idx])
        bbox3D = center + dimension
        
        pose = poses[bb_idx]
        if isinstance(pose, np.ndarray):
            pose = np.squeeze(pose).tolist()
        
        color = [c/255.0 for c in util.get_color(bb_idx)]
        box_mesh = util.mesh_cuboid(bbox3D, pose, color=color)
        boxes.append(box_mesh)
        
    K = get_intrinsics_for_chunk(chunk)
    
    image = chunk.image
    
    im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(image, K, boxes, text=None, scale=image.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
    im_drawn_rgb = np.clip(im_drawn_rgb, 0, 255).astype(np.uint8)
    
    return im_drawn_rgb
    
def display_image(image):    
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
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

def get_color_by_index(index):
    color = util.get_color(index)
    return [c / 255.0 for c in color]

def mesh_to_dict(mesh):
    mesh_dict = {
        'vertices': np.asarray(mesh.vertices),
        'faces': np.asarray(mesh.triangles)
    }
    
    if mesh.has_vertex_colors():
        mesh_dict['colors'] = np.asarray(mesh.vertex_colors)
    else:
        mesh_dict['colors'] = None
        
    return mesh_dict


def dict_to_mesh(mesh_dict):
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_dict['vertices'])
    
    faces = np.asarray(mesh_dict['faces'])
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    
    if mesh_dict.get('colors') is not None:
        mesh.vertex_colors = open3d.utility.Vector3dVector(mesh_dict['colors'])
    
    mesh.compute_vertex_normals()
    
    return mesh

    
def _mesh_to_plotly(mesh):
    # transpose z and y axes and flip y to match Open3D coords
    mesh.vertices = open3d.utility.Vector3dVector(np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, -1, 1])
    
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    
    plotly_mesh = go.Mesh3d(
            x=vertices[:,0],
            y=vertices[:,1],
            z=vertices[:,2],
            i=triangles[:,0],
            j=triangles[:,1],
            k=triangles[:,2],
            vertexcolor=colors,
            opacity=0.50)
    
    return plotly_mesh

def render_scene(meshes):
    plotly_meshes = [_mesh_to_plotly(mesh) for mesh in meshes]
    fig = go.Figure(
        data=[*plotly_meshes],
        layout=dict(
            scene=dict(
                aspectmode='data',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()
        
def pointcloud_to_mesh(pointcloud, color=(0.5, 0.5, 0.5)):
    if isinstance(pointcloud, open3d.geometry.PointCloud):
        pcd = pointcloud
    else:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pointcloud)
    pcd.paint_uniform_color(color)
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    radius = avg_distance * 1.5
    
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, open3d.utility.DoubleVector([radius, radius * 2]))
    
    return mesh


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


_TRIMESH_TO_OPENCV_AXIS_MAP = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)

# convert GLTF system (x left, y up, z forward) to OpenCV system (x right, y down, z forward)
def trimesh_to_opencv(mesh_or_path, rotation_matrix, translation_vector):
    mesh = read_trimesh(mesh_or_path)
    vertices_src = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    mapping = _TRIMESH_TO_OPENCV_AXIS_MAP.copy()
    vertices_cv = (vertices_src @ mapping.T).astype(np.float32)

    rotation_src = normalize_rotation_matrix(rotation_matrix)
    translation_src = normalize_translation_vector(translation_vector)

    rotation_cv = (mapping @ rotation_src @ mapping.T).astype(np.float32)
    translation_cv = (mapping @ translation_src).astype(np.float32).reshape(3, 1)

    return vertices_cv, faces, rotation_cv, translation_cv


def opencv_to_trimesh_pose(rotation_matrix_cv, translation_vector_cv):
    mapping = _TRIMESH_TO_OPENCV_AXIS_MAP.copy()
    rotation_cv = normalize_rotation_matrix(rotation_matrix_cv)
    translation_cv = normalize_translation_vector(translation_vector_cv)

    rotation_src = (mapping.T @ rotation_cv @ mapping).astype(np.float32)
    translation_src = (mapping.T @ translation_cv).astype(np.float32)

    return rotation_src, translation_src


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
    
    rot_90_x = open3d.geometry.get_rotation_matrix_from_axis_angle([np.pi / 2, 0.0, 0.0])
    o3d_mesh.rotate(rot_90_x, center=[0.0, 0.0, 0.0])
                                    
    return o3d_mesh

def render_contour_with_correspondences(image, contour_points_2d, correspondences_2d=None, center_2d=None, bundle_src_locations=None):
    overlay = image.copy()
    contour = contour_points_2d.reshape(-1, 2).astype(np.int32)
    cv.polylines(overlay, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
    for pt in contour:
        cv.circle(overlay, tuple(pt), 2, (0, 255, 255), -1)
    if center_2d is not None:
        cv.circle(overlay, tuple(np.asarray(center_2d, dtype=int)), 5, (0, 0, 255), -1)
    if bundle_src_locations is not None:
        overlay = cv.rapid.drawSearchLines(
            overlay,
            bundle_src_locations,
            (255, 0, 255)
        )
    if correspondences_2d is not None:
        for pt in correspondences_2d.reshape(-1, 2).astype(np.int32):
            cv.circle(overlay, tuple(pt), 2, (0, 255, 255), -1)
    return overlay
    
