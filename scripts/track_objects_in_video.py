import argparse
from copyreg import pickle
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking import track_objects_in_video    

    
def do_tracking(video_path, classes, threshold_2d, threshold_3d, export_meshes, colors, output_path):
    tracking_results = track_objects_in_video(
        video_path=video_path,
        classes=classes,
        threshold_2d=threshold_2d,
        threshold_3d=threshold_3d,
        export_meshes=export_meshes,
        colors=colors
    )
    with open(output_path, 'wb') as f:
        pickle.dump(tracking_results, f)
    print(f"Tracking results saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track 3D bounding boxes for specified object from a video."
    )
    
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of object classes to track (e.g., 'chair' 'table' 'person')"
    )
    
    parser.add_argument(
        "--threshold_2d",
        type=float,
        default=0.25,
        help="Confidence threshold for 2D bounding box detection (default: 0.25)"
    )
    
    parser.add_argument(
        "--threshold_3d",
        type=float,
        default=0.3,
        help="Confidence threshold for 3D bounding box detection (default: 0.3)"
    )
    
    parser.add_argument(
        "--export_meshes",
        type=bool,
        default=False,
        help="Export Open3D meshes for each bounding box"
    )
    
    parser.add_argument(
        "--colors",
        type=float,
        nargs="+",
        help="RGB colors for each class (e.g., 1.0 0.0 0.0 0.0 1.0 0.0 for red and green). Must be 3*len(classes) values."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the tracking results."
    )
    
    args = parser.parse_args()
    
    colors = None
    if args.colors:
        if len(args.colors) % 3 != 0:
            raise ValueError("Colors must be provided as RGB triplets (3 values per color)")
        if len(args.colors) // 3 != len(args.classes):
            raise ValueError(f"Number of colors ({len(args.colors) // 3}) must match number of classes ({len(args.classes)})")
        
        colors = []
        for i in range(0, len(args.colors), 3):
            colors.append([args.colors[i], args.colors[i+1], args.colors[i+2]])
         
    output_path = None      
    if args.output is None:
        output_path = args.video_path.rsplit('.', 1)[0] + '_tracking_results.pkl'
    else:
        output_path = args.output
        
    do_tracking(
        video_path=args.video_path,
        classes=args.classes,
        threshold_2d=args.threshold_2d,
        threshold_3d=args.threshold_3d,
        export_meshes=args.export_meshes,
        colors=colors,
        output_path=output_path
    )