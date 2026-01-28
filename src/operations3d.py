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
    
    # change directory to ovmono3d during model loading and inference
    original_dir = os.getcwd()
    os.chdir(os.path.join(original_dir, 'ovmono3d'))
    
    try:
        cfg = _get_config()
        model = build_model(cfg)

        DetectionCheckpointer(model, save_dir="temp").resume_or_load(
            CHECKPOINT_PATH, resume=True
        )

        model.eval()
        
        h, w, _ = chunk.image.shape
        K = get_intrinsics_for_chunk(chunk)
        categories = [prompt]

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(chunk.image.transpose(2, 0, 1))).cpu(), 
            'height': h, 'width': w, 'K': K, 'category_list': categories
        }]
        predictions = model(batched)[0]['instances']
        
        centers, dimensions, poses = [], [], []
        for pred_idx in range(len(predictions)):
            pred = predictions[pred_idx]
            if pred.scores.item() < threshold:
                continue
            
            centers.append(pred.pred_center_cam.detach().numpy())
            dimensions.append(pred.pred_dimensions.detach().numpy())
            poses.append(pred.pred_pose.detach().numpy())
        
        return centers, dimensions, poses
    finally:
        os.chdir(original_dir)
        
def adjust_centers_by_chunk_rotation(centers, chunk: ImageChunk):
    rotated_centers = []
    
    angle_horizontal_rad, angle_vertical_rad = chunk.angle
    rotation = np.array([
        [np.cos(angle_horizontal_rad), -np.sin(angle_horizontal_rad), 0],
        [np.sin(angle_horizontal_rad), np.cos(angle_horizontal_rad), 0],
        [0, 0, 1]
    ])
    for center in centers:
        rotated_center = rotation @ center.T
        rotated_centers.append(rotated_center)
        
    return rotated_centers

def _torch_mesh_to_open3d(torch_mesh):
    verts = torch_mesh.verts_packed().detach().cpu().numpy()
    faces = torch_mesh.faces_packed().detach().cpu().numpy()
    
    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = open3d.utility.Vector3iVector(faces)
    
    if torch_mesh.textures is not None and hasattr(torch_mesh.textures, 'verts_features_packed'):
        colors = torch_mesh.textures.verts_features_packed().detach().cpu().numpy()
        if colors.max() > 1.0:
            colors = colors / 255.0
        o3d_mesh.vertex_colors = open3d.utility.Vector3dVector(colors[:, :3])
    
    o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh

def get_box_meshes(boxes, color=(0, 0, 255)):
    centers, dimensions, poses = boxes
    meshes = []
    for box_idx in range(len(centers)):
        center = centers[box_idx].flatten().tolist() if isinstance(centers[box_idx], np.ndarray) else list(centers[box_idx])
        dimension = dimensions[box_idx].flatten().tolist() if isinstance(dimensions[box_idx], np.ndarray) else list(dimensions[box_idx])
        bbox3d = center + dimension
        
        pose = poses[box_idx]
        if isinstance(pose, np.ndarray):
            pose = np.squeeze(pose).tolist()
        
        mesh = util.mesh_cuboid(bbox3d, pose, color=color)
        mesh = _torch_mesh_to_open3d(mesh)
        meshes.append(mesh)
    return meshes
        