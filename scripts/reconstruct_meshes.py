from pathlib import Path
import pickle
import sys
import argparse
import trimesh
import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.util import read_video_frames
from src.tracking import reconstruct_meshes_for_class

def reconstruct_meshes(
    video_path,
    classes,
    output_dir,
    input_pkl=None,
    output_pkl=None,
    frame_index=0,
    generate_texture=True,
):
    input_frames = read_video_frames(video_path)
    frame = input_frames[frame_index]

    frames_data = None
    if input_pkl:
        with open(input_pkl, 'rb') as f:
            frames_data = pickle.load(f)
    
    reconstructed_meshes_by_class = {}
    
    for class_name in classes:
        unposed_meshes, adjusted_meshes, adjusted_scales, adjusted_rotations, adjusted_translations, image_chunk_centers, image_chunk_sizes = reconstruct_meshes_for_class(
            frame,
            class_name,
            generate_texture=generate_texture,
            use_gpu=True,
        )
        
        if not unposed_meshes:
            continue
        
        class_dir = f"{output_dir}/{class_name}"
        Path(class_dir).mkdir(parents=True, exist_ok=True)

        for existing_file in Path(class_dir).glob("unposed_*.glb"):
            existing_file.unlink()
        
        class_meshes_data = []
        
        for idx, (unposed_mesh, _, scale, rotation, translation, image_chunk_center, image_chunk_size) in enumerate(zip(
            unposed_meshes, adjusted_meshes, adjusted_scales, adjusted_rotations, adjusted_translations, image_chunk_centers, image_chunk_sizes
        )):
            # Save unposed mesh
            unposed_path = f"{class_dir}/unposed_{idx}.glb"
            unposed_mesh.export(unposed_path)
            
            class_meshes_data.append({
                'unposed_mesh_path': unposed_path,
                'scale': scale.tolist() if isinstance(scale, np.ndarray) else scale,
                'rotation': rotation.tolist() if isinstance(rotation, np.ndarray) else rotation,
                'translation': translation.tolist() if isinstance(translation, np.ndarray) else translation,
                'image_chunk_center': list(image_chunk_center) if isinstance(image_chunk_center, tuple) else image_chunk_center,
                'image_chunk_size': list(image_chunk_size) if isinstance(image_chunk_size, tuple) else image_chunk_size,
            })
        
        reconstructed_meshes_by_class[class_name] = class_meshes_data
    
    if not reconstructed_meshes_by_class:
        print("No meshes found to reconstruct")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saved meshes for {len(reconstructed_meshes_by_class)} classes")
    
    if input_pkl:
        if not output_pkl:
            output_pkl = input_pkl

        frame_data = frames_data[frame_index]
        frame_classes = frame_data.get('classes', [])
        classes_by_name = {
            class_entry.get('class_name'): class_entry
            for class_entry in frame_classes
            if isinstance(class_entry, dict) and class_entry.get('class_name')
        }

        for class_name, class_meshes in reconstructed_meshes_by_class.items():
            class_entry = classes_by_name.get(class_name)
            if class_entry is None:
                class_entry = {'class_name': class_name}
                frame_classes.append(class_entry)

            class_entry['reconstructed_meshes'] = class_meshes

        frame_data['classes'] = frame_classes
        
        with open(output_pkl, 'wb') as f:
            pickle.dump(frames_data, f)
        
        print(f"Saved mesh metadata to pickle file")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct and combine meshes from video frames for specific object classes"
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
        help="Path to save the output pickle file with mesh paths"
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index of the frame to process (default: 0)"
    )
    parser.add_argument(
        "--generate-texture",
        "--textured-mesh",
        dest="generate_texture",
        action="store_true",
        default=True,
        help="Enable texture baking for SAM3D mesh reconstruction"
    )
    
    args = parser.parse_args()
        
    reconstruct_meshes(
        video_path=args.video_path,
        classes=args.classes,
        output_dir=args.output_dir,
        frame_index=args.frame_index,
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl,
        generate_texture=args.generate_texture,
    )

if __name__ == "__main__":
    main()
