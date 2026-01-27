import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_convert

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

def display_image(image):    
    plt.imshow(image)
    plt.axis('off')
    plt.show()