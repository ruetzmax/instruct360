import os
import numpy as np
from typing import Tuple
from py360convert import e2p

from src.inference.conda_inference import CondaInferenceRunner, ThreadedCondaInferenceRunner
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

    @classmethod
    def from_image_point(
        cls,
        equirect: np.array,
        chunk_center: Tuple[float, float],
        chunk_size: Tuple[int, int] = (700, 700),
        orientation: str = 'horizontal',
    ):
        h, w, _ = equirect.shape

        cx, cy = chunk_center
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            chunk_center_normalized = (float(cx), float(cy))
        else:
            chunk_center_normalized = (float(cx) / w, float(cy) / h)

        lookup_angle_horizontal_rad = (chunk_center_normalized[0] - 0.5) * 2 * np.pi
        lookup_angle_vertical_rad = (0.5 - chunk_center_normalized[1]) * np.pi

        if orientation == 'vertical':
            scene_angle_horizontal_rad = (0.5 - chunk_center_normalized[1]) * 2 * np.pi
            scene_angle_vertical_rad = (chunk_center_normalized[0] - 0.5) * np.pi
        elif orientation == 'horizontal':
            scene_angle_horizontal_rad = (chunk_center_normalized[0] - 0.25) * 2 * np.pi
            scene_angle_vertical_rad = (chunk_center_normalized[1] - 0.5) * np.pi
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'")

        angle = (scene_angle_horizontal_rad, scene_angle_vertical_rad)

        fov_x = 360 * (chunk_size[0] / w)
        fov_y = 180 * (chunk_size[1] / h)
        fov = (fov_x, fov_y)

        image_chunk = e2p(
            equirect,
            fov_deg=(fov_x, fov_y),
            u_deg=np.degrees(lookup_angle_horizontal_rad),
            v_deg=np.degrees(lookup_angle_vertical_rad),
            out_hw=chunk_size,
        )

        return cls(image=image_chunk, center=chunk_center_normalized, angle=angle, fov=fov)

def find_closest_image_chunk(image_chunk: ImageChunk, candidate_chunks):
    target_center = np.asarray(image_chunk.center, dtype=np.float32)
    closest_chunk = None
    closest_distance = float('inf')
    for candidate_chunk in candidate_chunks:
        candidate_center = np.asarray(candidate_chunk.center, dtype=np.float32)
        distance = np.linalg.norm(candidate_center - target_center)
        if distance < closest_distance:
            closest_distance = distance
            closest_chunk = candidate_chunk
    return closest_chunk

_sam3_runner = None

def _get_sam3_runner(env_name="sam3d-objects", is_threaded=True):
    global _sam3_runner
    if _sam3_runner is None:
        if is_threaded:
            _sam3_runner = ThreadedCondaInferenceRunner(env_name, "sam3_worker.py")
        else:
            _sam3_runner = CondaInferenceRunner(env_name, "sam3_inference.py")

    return _sam3_runner
    

def get_2d_bounding_boxes(
    image,
    prompt,
    threshold=0.35,
    sam3_env="sam3",
    use_persistent_runner=True
):
    threshold = 0.7
    runner = _get_sam3_runner(sam3_env, is_threaded=use_persistent_runner)
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


def bounding_boxes_to_image_chunks(image, bounding_boxes, chunk_size=(700, 700), orientation='vertical'):
    # create an ImageChunk for each bounding box
    image_chunks = []
    for box in bounding_boxes:
        # box format: [0,1](cx, cy, w, h)
        chunk_center = (float(box[0]), float(box[1]))
        image_chunks.append(
            ImageChunk.from_image_point(
                equirect=image,
                chunk_center=chunk_center,
                chunk_size=chunk_size,
                orientation=orientation,
            )
        )
    
    return image_chunks

def get_masks_from_image_chunks(
    image_chunks,
    prompt,
    sam3_env="sam3",
    use_persistent_runner=True
):
    runner = _get_sam3_runner(sam3_env, is_threaded=use_persistent_runner)
    
    input_data = {
        "chunk_images_base64": [image_to_base64(chunk.image) for chunk in image_chunks],
        "prompt": prompt
    }
    
    output_data = runner.run(input_data)
    all_masks = [base64_to_image(mask_b64) for mask_b64 in output_data["masks_base64"]]
    
    return all_masks    