import subprocess
import os

from src.operations2d import get_2d_bounding_boxes, bounding_boxes_to_image_chunks
from src.operations3d import get_3d_bounding_boxes, adjust_rotation_by_chunk_rotation, get_box_meshes
from src.util import read_video_frames, get_color_by_index, mesh_to_dict
from tqdm import tqdm

def get_bounding_boxes_for_class(image, class_name, threshold_2d=0.25, threshold_3d=0.3, orientation='vertical'):
    bounding_boxes_2d = get_2d_bounding_boxes(image, class_name, threshold=threshold_2d)
    
    image_chunks = bounding_boxes_to_image_chunks(image, bounding_boxes_2d, orientation=orientation)
    
    all_3d_bb_centers = []
    all_3d_bb_dimensions = []
    all_3d_bb_poses = []
    for chunk in image_chunks:
        bb_3d_centers, bb_3d_dimensions, bb_3d_poses = get_3d_bounding_boxes(chunk, class_name, threshold=threshold_3d)
        bb_3d_centers_adjusted, bb_3d_poses_adjusted = adjust_rotation_by_chunk_rotation(bb_3d_centers, bb_3d_poses, chunk)
        all_3d_bb_centers.extend(bb_3d_centers_adjusted)
        all_3d_bb_dimensions.extend(bb_3d_dimensions)
        all_3d_bb_poses.extend(bb_3d_poses_adjusted)
        
    return bounding_boxes_2d, all_3d_bb_centers, all_3d_bb_dimensions, all_3d_bb_poses

def track_objects_in_video(classes, threshold_2d=0.25, threshold_3d=0.3, export_meshes=False, colors=None, video_path=None, left_video_path=None, right_video_path=None, orientation='vertical'):
    
    if colors and len(colors) != len(classes):
        raise ValueError("Length of colors must match length of classes.")
    
    frames = read_video_frames(video_path, left_video_path, right_video_path)
    
    frame_results = []
    
    # iterate over each frame and get bounding boxes (+ open3d meshes) for each class
    for frame_idx, frame in enumerate(tqdm(frames, desc="Tracking frames", unit="frame")):
        frame_result = {
            'frame_index': frame_idx,
            'classes': []
        }
        
        for class_idx, class_name in enumerate(classes):
            bbs = get_bounding_boxes_for_class(
                frame, class_name, threshold_2d=threshold_2d, threshold_3d=threshold_3d, orientation=orientation
            )
            
            bb2ds, bb_centers, bb_dimensions, bb_poses = bbs
            
            if not bb_centers:
                continue
            
            
            class_result = {
                'class_name': class_name,
                'bb2ds': bb2ds,
                'centers': bb_centers,
                'dimensions': bb_dimensions,
                'poses': bb_poses
            }
            
            if export_meshes:
                color = colors[class_idx] if colors else get_color_by_index(class_idx)
                bb_meshes = get_box_meshes((bb_centers, bb_dimensions, bb_poses), color=color)
                bb_mesh_dicts = [mesh_to_dict(mesh) for mesh in bb_meshes]
                class_result['meshes'] = bb_mesh_dicts

            
            frame_result['classes'].append(class_result)
            
        frame_results.append(frame_result)
    
    return frame_results

def track_camera_poses(video_path):
    slam_executable = os.path.expanduser("~/lib/stella_vslam_examples/build/run_video_slam")
    slam_command = [
        slam_executable,
        "-v", "config/orb_vocab.fbow",
        "-c", "config/equirectangular.yaml",
        "-m", video_path,
        "--frame-skip", "1",
        "--temporal-mapping",
        "--viewer", "none",
        "--map-db-out", "temp/tracked.msg",
        "--eval-log-dir", "temp",
    ]
    subprocess.run(slam_command, check=True)
    
    camera_poses = []
    
    # each row contains: timestamp tx ty tz qx qy qz qw
    with open("temp/frame_trajectory.txt", "r") as f:
        for line in f:
            values = line.strip().split()
            
            frame_translation = [float(values[1]), float(values[2]), float(values[3])]
            frame_rotation = [float(values[4]), float(values[5]), float(values[6]), float(values[7])]
            frame_dict = {
                'translation': frame_translation,
                'rotation': frame_rotation
            }
            camera_poses.append(frame_dict)
    return camera_poses
        