from pathlib import Path
import pickle
import sys
import argparse
import open3d as o3d


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util import read_video_frames
from src.tracking import reconstruct_pointclouds_for_class

def reconstruct_pointclouds(video_path, classes, output_dir, input_pkl=None, output_pkl=None, frame_index=0):
    input_frames = read_video_frames(video_path)
    frame = input_frames[frame_index]
    
    all_pointclouds = []
    for class_name in classes:
        class_pointclouds = reconstruct_pointclouds_for_class(frame, class_name)
        all_pointclouds.extend(class_pointclouds)
    
    # Combine all pointclouds into one
    if not all_pointclouds:
        print("No pointclouds found to combine")
        return
    
    combined_pointcloud = all_pointclouds[0]
    for pc in all_pointclouds[1:]:
        combined_pointcloud += pc
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/{frame_index}.ply"
    o3d.io.write_point_cloud(output_file, combined_pointcloud)
    print(f"Saved combined pointcloud to {output_file}")
    
    if input_pkl:
        if not output_pkl:
            output_pkl = input_pkl
        
        with open(input_pkl, 'rb') as f:
            frames_data = pickle.load(f)
        
        frames_data[frame_index]['pointcloud_path'] = output_file
        
        with open(output_pkl, 'wb') as f:
            pickle.dump(frames_data, f)

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct and combine pointclouds from video frames for specific object classes"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of object class names to reconstruct (e.g., 'chair' 'table')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the output PLY file"
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        help="Path to the input object pickle file"
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        help="Path to save the output pickle file with pointcloud paths"
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index of the frame to process (default: 0)"
    )
    
    args = parser.parse_args()
        
    reconstruct_pointclouds(
        video_path=args.video_path,
        classes=args.classes,
        output_dir=args.output_dir,
        frame_index=args.frame_index,
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl
    )

if __name__ == "__main__":
    main()
