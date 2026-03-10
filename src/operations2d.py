import os
import subprocess
import numpy as np
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from typing import Tuple
from PIL import Image
from py360convert import e2p

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# if os.getenv("FORCE_CPU_MODE", "0") == "1":
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import sys
# sys.path.append("segment-anything")
# from segment_anything import sam_model_registry, SamPredictor

dino_model = None
ov_seg_model = None


# sam_model = sam_model_registry["vit_h"](checkpoint="ovmono3d/checkpoints/sam_vit_h_4b8939.pth")
# sam_predictor = SamPredictor(sam_model)



class ImageChunk:
    def __init__(self, image: np.array, center: Tuple[float, float], angle: Tuple[float, float], fov: Tuple[float, float]):
        self.image = image
        self.center = center
        self.angle = angle
        self.fov = fov
        

def _image_to_tensor(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
        
    image_transformed, _ = transform(image, None)
    return image_transformed

# see: https://github.com/peterbraden/insv-to-yt, https://www.arj.no/2025/12/19/insta360-to-equirectangular/
def insv_to_equirect(left_video_path, right_video_path, output_video_path, stitched_path="temp/stitched.mp4"):
    if not os.path.exists(left_video_path):
        raise FileNotFoundError(f"File not found: {left_video_path}")
    if not os.path.exists(right_video_path):
        raise FileNotFoundError(f"File not found: {right_video_path}")
    
    # stitch both videos side by side
    stitch_cmd = [
        "ffmpeg",
        "-i", "left_video_path",
        "-i", "right_video_path",
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]; [0:a][1:a]amerge[a]"
        "-map", "[v]",
        "-map", "[a]",
        "-ac", "2",
        stitched_path
    ]
    subprocess.run(stitch_cmd, check=True)
    
    # convert to equirect
    undistort_cmd = [
        "ffmpeg",
        "-i", stitched_path,
        "-vf", "v360=dfisheye:e:yaw=-90",
        output_video_path
    ]
    subprocess.run(undistort_cmd, check=True)
    print(f"Saved equirectangular video to: {output_video_path}")
    

def _get_dino_model():
    global dino_model
    if dino_model is None:
        print("Loading GroundingDINO model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        dino_model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
            "ovmono3d/checkpoints/groundingdino_swinb_cogcoor.pth",
            device=device
        )
        print(f"GroundingDINO model loaded.")
            
    return dino_model

def get_2d_bounding_boxes(image, prompt, threshold=0.35):
    model = _get_dino_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_tensor = _image_to_tensor(image)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=threshold,
        text_threshold=0.25,
        device=device
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return boxes.numpy()

def bounding_boxes_to_image_chunks(image, bounding_boxes, chunk_size=(700, 700), orientation='vertical'):
    # create an ImageChunk for each bounding box
    image_chunks = []
    h, w, _ = image.shape
    for box in bounding_boxes:
        # box format: [0,1](cx, cy, w, h)
        box_center_pixel = (int(box[0] * w), int(box[1] * h))
        chunk_center_pixel = box_center_pixel
        
        #convert chunk center to normalized coordinates
        chunk_center_normalized = (chunk_center_pixel[0] / w, chunk_center_pixel[1] / h)
        
        #calculate angle from image center
        if orientation == 'vertical':
            scene_angle_horizontal_rad = (0.5 - chunk_center_normalized[1]) * 2 * np.pi 
            scene_angle_vertical_rad = (chunk_center_normalized[0] - 0.5) * np.pi 
        elif orientation == 'horizontal':
            lookup_angle_horizontal_rad = (chunk_center_normalized[0] - 0.5) * 2 * np.pi
            lookup_angle_vertical_rad = (0.5 - chunk_center_normalized[1]) * np.pi
            
            scene_angle_horizontal_rad = (chunk_center_normalized[0] - 0.25) * 2 * np.pi 
            scene_angle_vertical_rad = (chunk_center_normalized[1]-0.5) * np.pi
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'")
            
        angle = (scene_angle_horizontal_rad, scene_angle_vertical_rad)

        # calculate fov
        fov_x = 360 * (chunk_size[0] / w)
        fov_y = 180 * (chunk_size[1] / h)
        fov = (fov_x, fov_y)
        
        # extract image chunk by projecting equirectangular to perspective
        image_chunk = e2p(image, fov_deg=(fov_x, fov_y), u_deg=np.degrees(lookup_angle_horizontal_rad), v_deg=np.degrees(lookup_angle_vertical_rad), out_hw=chunk_size)
        
        image_chunks.append(ImageChunk(image=image_chunk, center=chunk_center_normalized, angle=angle, fov=fov))
    
    return image_chunks

def image_chunk_from_undistorted(image: np.array):
    return ImageChunk(image=image, center=(0.5, 0.5), angle=(0.0, 0.0), fov=(117, 117)) #using specs of OnePlus 7 Pro

def _get_ov_seg_model():
    global ov_seg_model
    if ov_seg_model is None:
        print("Loading OV-Seg model...")
        import sys
        sys.path.append("ov-seg")
        from open_vocab_seg.utils import VisualizationDemo
        from open_vocab_seg import add_ovseg_config
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        
        ov_seg_cfg = get_cfg()
        add_deeplab_config(ov_seg_cfg)
        add_ovseg_config(ov_seg_cfg)
        ov_seg_cfg.merge_from_file("ov-seg/configs/ovseg_swinB_vitL_demo.yaml")
        ov_seg_cfg.MODEL.WEIGHTS = "ov-seg/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth"
        ov_seg_cfg.DATALOADER.NUM_WORKERS = 0
        if not torch.cuda.is_available():
            ov_seg_cfg.MODEL.DEVICE = "cpu"
        ov_seg_model = VisualizationDemo(ov_seg_cfg)
        print("OV-Seg model loaded.")
    return ov_seg_model

def get_masks_from_image_chunks(image_chunks, prompt): 
    model = _get_ov_seg_model()

    all_masks = []
    for image_chunk in image_chunks:
        predictions, _ = model.run_on_image(image_chunk.image, [prompt])
        
        if not "sem_seg" in predictions:
            return []
        
        #apply the pred mask to the original image chunk to get the masked image
        sem_seg = predictions["sem_seg"]
        blank_area = (sem_seg[0] == 0)

        masked_image = image_chunk.image.copy()
        rgba_image = np.zeros((masked_image.shape[0], masked_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = masked_image
        rgba_image[:, :, 3] = 255 
        rgba_image[blank_area, 3] = 0 

        all_masks.append(rgba_image)
        
    return all_masks
    