import cv2 as cv
import matplotlib.pyplot as plt
import open3d
import torch
from torchvision.ops import box_convert
import numpy as np

from src.operations2d import ImageChunk
from src.operations3d import get_intrinsics_for_chunk

from ovmono3d.cubercnn import util, vis

from open3d.visualization import draw_geometries

import plotly.graph_objects as go


def read_video_frames(video_path):
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
    
    character_placeholder = open3d.geometry.OrientedBoundingBox(
        center=[0, -height / 2, 0],
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
    faces = faces[:, [0, 2, 1]]  # reverse winding order
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
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()
        

    
