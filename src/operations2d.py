import os
import subprocess
import numpy as np
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from typing import Tuple
from PIL import Image
from py360convert import e2p
import torch

dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "ovmono3d/checkpoints/groundingdino_swinb_cogcoor.pth")


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
    

def get_2d_bounding_boxes(image, prompt, threshold=0.35):
    global dino_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_tensor = _image_to_tensor(image)
    
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=threshold,
        text_threshold=0.25,
        device=device
    )

    return boxes.numpy()

def bounding_boxes_to_image_chunks(image, bounding_boxes, chunk_size=(700, 700)):
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
        angle_horizontal_rad = (0.5 - chunk_center_normalized[1]) * 2 * np.pi 
        angle_vertical_rad = (chunk_center_normalized[0] - 0.5) * np.pi 
        angle = (angle_horizontal_rad, angle_vertical_rad)

        # calculate fov
        fov_x = 360 * (chunk_size[0] / w)
        fov_y = 180 * (chunk_size[1] / h)
        fov = (fov_x, fov_y)
        
        # extract image chunk by projecting equirectangular to perspective
        image_chunk = e2p(image, fov_deg=(fov_x, fov_y), u_deg=np.degrees(angle_horizontal_rad), v_deg=np.degrees(angle_vertical_rad), out_hw=chunk_size)
        
        image_chunks.append(ImageChunk(image=image_chunk, center=chunk_center_normalized, angle=angle, fov=fov))
    
    return image_chunks
