from pathlib import Path
import sys
import pickle
import cv2
import argparse


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util import draw_bounding_boxes, read_video_frames

FPS = 24

class_colors = {
    'cupboard': (0, 0, 255),
    'cup': (0, 255, 0),
    'chair': (255, 0, 0),
}

def visualize_2d_bounding_boxes(input_video_path, object_pkl_path, output_video_path):
    frames = read_video_frames(input_video_path)
    
    with open(object_pkl_path, 'rb') as f:
        frames_data = pickle.load(f)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (w, h))
    
    for frame_idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        frame_data = frames_data[frame_idx]
        class_dicts = frame_data['classes']
        for class_dict in class_dicts:
            class_name = class_dict['class_name']
            bb2ds = class_dict['bb2ds']
            
            color = class_colors.get(class_name, (255, 255, 255))
            
            frame = draw_bounding_boxes(frame, bb2ds, color=color)
        
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video with 2D bounding boxes saved to: {output_video_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 2D bounding boxes on video frames')
    parser.add_argument('--input_video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--object_pkl', type=str, required=True,
                        help='Path to object pickle file containing bounding box data')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Path to output video file')
    
    args = parser.parse_args()
    
    visualize_2d_bounding_boxes(args.input_video, args.object_pkl, args.output_video)
    