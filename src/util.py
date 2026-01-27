import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_convert
import numpy as np

from src.operations2d import ImageChunk
from src.operations3d import get_intrinsics_for_chunk

from ovmono3d.cubercnn import util, vis


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