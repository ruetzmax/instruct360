import open3d
from src.operations2d import ImageChunk
import logging
import os
import sys
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.data import transforms as T

sys.path.insert(0, os.path.join(os.getcwd(), 'ovmono3d'))

from ovmono3d.cubercnn.modeling.meta_arch import build_model
from ovmono3d.cubercnn import util, vis

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from ovmono3d.cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone

CONFIG_PATH = "configs/OVMono3D_dinov2_SFP.yaml"
CHECKPOINT_PATH = "checkpoints/ovmono3d_lift.pth"

def _get_config():
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    global CONFIG_PATH, CHECKPOINT_PATH

    # store locally if needed
    if CONFIG_PATH.startswith(util.CubeRCNNHandler.PREFIX):    
        CONFIG_PATH = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, CONFIG_PATH)

    cfg.merge_from_file(CONFIG_PATH)
    
    cfg.MODEL.ROI_HEADS.NAME = "ROIHeads3DGDINO"
    
    cfg.freeze()
    default_setup(cfg, None)
    return cfg

# change directory to ovmono3d during model loading
ovmono_model = None
original_dir = os.getcwd()
os.chdir(os.path.join(original_dir, 'ovmono3d'))

try:
    cfg = _get_config()
    ovmono_model = build_model(cfg)

    DetectionCheckpointer(ovmono_model, save_dir="temp").resume_or_load(
        CHECKPOINT_PATH, resume=True
    )
finally:
        os.chdir(original_dir)
        

def get_intrinsics_for_chunk(chunk: ImageChunk):
    #calculate focal length from fov
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

def get_3d_bounding_boxes(chunk: ImageChunk, prompt: str, threshold=0.3):
    
    global ovmono_model

    ovmono_model.eval()
    
    h, w, _ = chunk.image.shape
    K = get_intrinsics_for_chunk(chunk)
    categories = [prompt]

    batched = [{
        'image': torch.as_tensor(np.ascontiguousarray(chunk.image.transpose(2, 0, 1))).cpu(), 
        'height': h, 'width': w, 'K': K, 'category_list': categories
    }]
    predictions = ovmono_model(batched)[0]['instances']
    
    centers, dimensions, poses = [], [], []
    for pred_idx in range(len(predictions)):
        pred = predictions[pred_idx]
        if pred.scores.item() < threshold:
            continue
        
        centers.append(pred.pred_center_cam.detach().cpu().numpy())
        dimensions.append(pred.pred_dimensions.detach().cpu().numpy())
        poses.append(pred.pred_pose.detach().cpu().numpy())
    
    return centers, dimensions, poses
        
def adjust_rotation_by_chunk_rotation(centers, poses, chunk: ImageChunk):
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
        [0, np.cos(angle_vertical_rad), -np.sin(angle_vertical_rad)],
        [0, np.sin(angle_vertical_rad), np.cos(angle_vertical_rad)]
    ])
    
    rotation = rotation_pitch @ rotation_yaw
    
    for center in centers:
        rotated_center = rotation @ center.T
        # rotated_center = center
        rotated_centers.append(rotated_center)
    
    for pose in poses:
        pose_array = np.array(pose).reshape(3, 3)
        rotated_pose = rotation @ pose_array
        # rotated_pose = pose_array
        rotated_poses.append(rotated_pose)
    
    return rotated_centers, rotated_poses
    
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
    
    # flip x an z to match Open3D coordinate system
    vertices = vertices * [-1, 1, -1]
    
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
        