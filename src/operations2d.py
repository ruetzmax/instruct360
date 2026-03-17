import os
import subprocess
import numpy as np
from typing import Tuple
from PIL import Image
from py360convert import e2p

from src.inference.conda_inference import CondaInferenceRunner
from src.inference.inference_utils import image_to_base64, base64_to_image

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class ImageChunk:
    def __init__(self, image: np.array, center: Tuple[float, float], angle: Tuple[float, float], fov: Tuple[float, float]):
        self.image = image
        self.center = center
        self.angle = angle
        self.fov = fov


_dino_runner = None
_ovseg_runner = None
_sam3_runner = None


def _get_dino_runner(env_name="grounding_dino"):
    global _dino_runner
    if _dino_runner is None:
        _dino_runner = CondaInferenceRunner(env_name, "dino_inference.py")
    return _dino_runner


def _get_ovseg_runner(env_name="ovseg"):
    global _ovseg_runner
    if _ovseg_runner is None:
        _ovseg_runner = CondaInferenceRunner(env_name, "ovseg_inference.py")
    return _ovseg_runner


def _get_sam3_runner(env_name="sam3d-objects"):
    global _sam3_runner
    if _sam3_runner is None:
        _sam3_runner = CondaInferenceRunner(env_name, "sam3_inference.py")
    return _sam3_runner

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
    

def get_2d_bounding_boxes(
    image,
    prompt,
    threshold=0.35,
    use_gpu=False,
    dino_env="grounding_dino",
    sam3_env="sam3d-objects",
):
    if use_gpu:
        runner = _get_sam3_runner(sam3_env)
        input_data = {
            "chunk_images_base64": [image_to_base64(image)],
            "prompt": prompt,
            "box_threshold": threshold,
        }
        output_data = runner.run(input_data)
        boxes_for_image = output_data.get("boxes", [[]])[0]
        if not boxes_for_image:
            return np.empty((0, 4), dtype=np.float32)
        return np.array(boxes_for_image, dtype=np.float32)

    runner = _get_dino_runner(dino_env)
    input_data = {
        "image_base64": image_to_base64(image),
        "prompt": prompt,
        "box_threshold": threshold,
        "text_threshold": 0.25,
    }
    output_data = runner.run(input_data)
    return np.array(output_data["boxes"])

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

def get_masks_from_image_chunks(
    image_chunks,
    prompt,
    use_gpu=False,
    ovseg_env="ovseg",
    sam3_env="sam3d-objects",
):
    runner = _get_sam3_runner(sam3_env) if use_gpu else _get_ovseg_runner(ovseg_env)
    
    input_data = {
        "chunk_images_base64": [image_to_base64(chunk.image) for chunk in image_chunks],
        "prompt": prompt
    }
    
    output_data = runner.run(input_data)
    all_masks = [base64_to_image(mask_b64) for mask_b64 in output_data["masks_base64"]]
    
    return all_masks    