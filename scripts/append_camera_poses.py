from pathlib import Path
import pickle
import sys
import argparse


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tracking import track_camera_poses


def append_camera_poses(input_video_path, object_pkl_path, output_pkl_path):
    camera_poses = track_camera_poses(input_video_path)
    
    with open(object_pkl_path, 'rb') as f:
        frames_data = pickle.load(f)
        
    for frame_idx, frame_data in enumerate(frames_data):
        
        if frame_idx == len(camera_poses):
            continue
        
        frame_pose = camera_poses[frame_idx]
        frame_data['camera_translation'] = frame_pose["translation"]
        frame_data['camera_rotation'] = frame_pose["rotation"]
        
    if output_pkl_path is None:
        output_pkl_path = object_pkl_path
        
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(frames_data, f)
    
    print(f"Saved output with camera poses to: {output_pkl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append camera poses to tracked objects"
    )
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="Path to the input object pickle file"
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        help="Path to save the output pickle file with camera poses"
    )
    
    args = parser.parse_args()
    
    append_camera_poses(
        args.input_video,
        args.input_pkl,
        args.output_pkl
    )