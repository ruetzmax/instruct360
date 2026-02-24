from pathlib import Path
import sys
import pickle
import cv2
import argparse
import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util import draw_3d_bounding_boxes, read_video_frames
from src.operations2d import bounding_boxes_to_image_chunks

FPS = 24

def unadjust_rotation_by_chunk_rotation(centers, poses, chunk):
    # convert from world space to chunk space
    unadjusted_centers = []
    unadjusted_poses = []
    
    angle_horizontal_rad, angle_vertical_rad = chunk.angle
    
    rotation_yaw = np.array([
        [np.cos(angle_horizontal_rad), 0, np.sin(angle_horizontal_rad)],
        [0, 1, 0],
        [-np.sin(angle_horizontal_rad), 0, np.cos(angle_horizontal_rad)]
    ])
    
    rotation_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(-angle_vertical_rad), -np.sin(-angle_vertical_rad)],
        [0, np.sin(-angle_vertical_rad), np.cos(-angle_vertical_rad)]
    ])
    
    rotation = rotation_pitch @ rotation_yaw
    
    inverse_rotation = rotation.T
    
    for center in centers:
        center_array = np.array(center).reshape(3, 1)
        unadjusted_center = inverse_rotation @ center_array
        unadjusted_centers.append(unadjusted_center)
    
    for pose in poses:
        pose_array = np.array(pose).reshape(3, 3)
        unadjusted_pose = inverse_rotation @ pose_array
        unadjusted_poses.append(unadjusted_pose)
    
    return unadjusted_centers, unadjusted_poses

def visualize_image_chunks(input_video_path, class_name, object_pkl_path, output_video_path, orientation='horizontal', draw_3d_bb=False):
    frames = read_video_frames(input_video_path)
    
    with open(object_pkl_path, 'rb') as f:
        frames_data = pickle.load(f)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    for frame_idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        frame_data = frames_data[frame_idx]
        class_dicts = frame_data['classes']
        for class_dict in class_dicts:
            class_dict_name = class_dict['class_name']
            if class_dict_name != class_name:
                continue
            
            bb2ds = class_dict['bb2ds']
            image_chunks = bounding_boxes_to_image_chunks(frame, bb2ds, orientation=orientation)
            
            chunk_to_draw = image_chunks[0] if image_chunks else None
            if chunk_to_draw is None:
                continue
        
            chunk_image = chunk_to_draw.image
            if video_writer is None:
                h, w, _ = chunk_image.shape
                video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (w, h))
                
            if draw_3d_bb:
                centers = class_dict['centers']
                dimensions = class_dict['dimensions']
                poses = class_dict['poses']
                centers_unadjusted, poses_unadjusted = unadjust_rotation_by_chunk_rotation(centers, poses, chunk_to_draw)
                chunk_image = draw_3d_bounding_boxes(chunk_to_draw, centers_unadjusted, dimensions, poses_unadjusted)
                
            video_writer.write(chunk_image)
        
    video_writer.release()
    print(f"Video with image chunks saved to: {output_video_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize image chunks from video frames')
    parser.add_argument('--input_video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--object_pkl', type=str, required=True,
                        help='Path to object pickle file containing bounding box data')
    parser.add_argument('--class_name', type=str, required=True,
                        help='Name of the object class to visualize (e.g., "chair")')
    parser.add_argument('--orientation', type=str, default='horizontal', choices=['horizontal', 'vertical'],
                        help='Orientation of the input video.')
    parser.add_argument('--draw_3d_bb', type=bool, default=False,
                        help='Whether to draw 3D bounding boxes.')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Path to output video file')
    
    args = parser.parse_args()
    
    visualize_image_chunks(
        input_video_path=args.input_video,
        class_name=args.class_name,
        object_pkl_path=args.object_pkl,
        output_video_path=args.output_video,
        orientation=args.orientation,
        draw_3d_bb=args.draw_3d_bb
    )
    
