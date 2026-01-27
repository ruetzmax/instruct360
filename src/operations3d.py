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

def get_3d_bounding_boxes(chunk: ImageChunk, prompt: str):
    
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
            if pred.scores.item() < 0.3:
                continue
            
            centers.append(pred.pred_center_cam.detach().numpy())
            dimensions.append(pred.pred_dimensions.detach().numpy())
            poses.append(pred.pred_pose.detach().numpy())
        
        return centers, dimensions, poses
    finally:
        os.chdir(original_dir)