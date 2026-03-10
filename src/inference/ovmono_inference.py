import os
import sys
import numpy as np
import torch
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup

from inference_utils import base64_to_image, load_inference_input, save_inference_output

# load model
ovmono_path = os.path.join(os.getcwd(), 'ovmono3d')
if ovmono_path not in sys.path:
    sys.path.insert(0, ovmono_path)

from cubercnn.modeling.meta_arch import build_model
from cubercnn import util
import cubercnn.modeling.backbone 
import cubercnn.modeling.roi_heads
import cubercnn.modeling.proposal_generator

logger = logging.getLogger("detectron2")
sys.dont_write_bytecode = True
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults

CONFIG_PATH = "configs/OVMono3D_dinov2_SFP.yaml"
CHECKPOINT_PATH = "checkpoints/ovmono3d_lift.pth"


def get_config():
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    global CONFIG_PATH, CHECKPOINT_PATH

    # Store locally if needed
    if CONFIG_PATH.startswith(util.CubeRCNNHandler.PREFIX):    
        CONFIG_PATH = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, CONFIG_PATH)

    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.ROI_HEADS.NAME = "ROIHeads3DGDINO"
    cfg.freeze()
    default_setup(cfg, None)
    return cfg


input_data = load_inference_input()

print("Loading OVMono3D model...")
original_dir = os.getcwd()
os.chdir(os.path.join(original_dir, 'ovmono3d'))

try:
    cfg = get_config()
    ovmono_model = build_model(cfg)
    DetectionCheckpointer(ovmono_model, save_dir="temp").resume_or_load(
        CHECKPOINT_PATH, resume=True
    )
    print("OVMono3D model loaded.")
finally:
    os.chdir(original_dir)

ovmono_model.eval()

# run inference
image_b64 = input_data["image_base64"]
prompt = input_data["prompt"]
threshold = input_data.get("threshold", 0.3)
intrinsics = np.array(input_data["intrinsics"])
height = input_data["height"]
width = input_data["width"]

image = base64_to_image(image_b64)

categories = [prompt]
batched = [{
    'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cpu(), 
    'height': height, 
    'width': width, 
    'K': intrinsics, 
    'category_list': categories
}]

predictions = ovmono_model(batched)[0]['instances']

centers, dimensions, poses = [], [], []
for pred_idx in range(len(predictions)):
    pred = predictions[pred_idx]
    if pred.scores.item() < threshold:
        continue
    
    centers.append(pred.pred_center_cam.detach().cpu().numpy().tolist())
    dimensions.append(pred.pred_dimensions.detach().cpu().numpy().tolist())
    poses.append(pred.pred_pose.detach().cpu().numpy().tolist())

output_data = {
    "centers": centers,
    "dimensions": dimensions,
    "poses": poses
}

save_inference_output(output_data)
print(f"Found {len(centers)} 3D bounding boxes")
