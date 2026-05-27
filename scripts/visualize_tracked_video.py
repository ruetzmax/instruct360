
import argparse
import os
import time
from pathlib import Path
import pickle
import sys
import open3d
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operations3d import adjust_pose_by_camera_pose, apply_mesh_transforms
from src.util import get_character_placeholder, trimesh_to_open3d, read_trimesh

world_landmarks = []

static_geometries = []

def _point_cloud_from_landmarks(landmarks):       
        if len(landmarks) == 0:
            return
                
        landmark_pointcloud = open3d.geometry.PointCloud()
        landmarks = [[pos[2], -pos[1], pos[0]] for pos in landmarks]
        landmarks_np = np.array(landmarks)
        landmark_pointcloud.points = open3d.utility.Vector3dVector(landmarks_np)
        landmark_pointcloud.paint_uniform_color([0.0, 0.0, 0.0])
        return landmark_pointcloud

def _setup_video_writer(output_video_path, vis, fps):
    setup_image = vis.capture_screen_float_buffer(do_render=False)
    setup_image_np = np.asarray(setup_image)
    height, width = setup_image_np.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    return video_writer  

def _render_frame(vis):
    image = vis.capture_screen_float_buffer(do_render=False)
    image_np = np.asarray(image)
    image_np = (image_np * 255).astype(np.uint8)
    
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

def do_visualization(
    object_pkl_path: str,
    output_video_path: str = None,
    fps: float = 24,
    show_poses: bool = True,
    show_reconstructed: bool = True,
    show_landmarks: bool = True,
    object_scale: float = 2.0,
    disable_rotation: bool = False,
):
    with open(object_pkl_path, 'rb') as f:
        frames_data = pickle.load(f)
        
    print(f"Loaded {len(frames_data)} frames")
    
    if output_video_path:
        print(f"Recording video to: {output_video_path}")
        print("Recording all frames...")
    else:
        print("Controls: 'a' - previous frame, 'd' - next frame, 'space' - play/pause, 'q' - quit")
        
        
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()   
    
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(10000.0)
    ctr.set_constant_z_near(0.1) 
    
    video_writer = None     
    
    state = {'frame_idx': 0, 'playing': False}
    
    def update_frame():
        global unique_meshes_by_class
        
        frame_idx = state['frame_idx']
        frame_data = frames_data[frame_idx]
        
        frame_camera_translation = frame_data.get('camera_translation', None)
        frame_camera_rotation = frame_data.get('camera_rotation', None)
        
        meshes_to_draw = []
        
        # add reconstructed meshes
        if show_reconstructed:
            reconstructed_meshes_data = {}
            frame_classes_for_meshes = frame_data.get('classes', [])
            for class_dict in frame_classes_for_meshes:
                if not isinstance(class_dict, dict):
                    continue
                class_name = class_dict.get('class_name')
                class_meshes = class_dict.get('reconstructed_meshes', [])
                if class_name and class_meshes:
                    reconstructed_meshes_data[class_name] = class_meshes

            if reconstructed_meshes_data:
                for class_name, meshes_info_list in reconstructed_meshes_data.items():
                    for mesh_info in meshes_info_list:
                        unposed_mesh_path = mesh_info.get('unposed_mesh_path')
                        scale = np.array(mesh_info.get('scale', [1.0, 1.0, 1.0]), dtype=np.float32) * float(object_scale)
                        if disable_rotation:
                            rotation = np.eye(3, dtype=np.float32)
                        else:
                            rotation = np.array(mesh_info.get('chunk_relative_rotation', np.eye(3, dtype=np.float32)))
                        translation = np.array(mesh_info.get('translation'))

                        if unposed_mesh_path and os.path.exists(unposed_mesh_path):
                            try:
                                unposed_mesh = read_trimesh(unposed_mesh_path)
                                transformed_mesh = apply_mesh_transforms(
                                    unposed_mesh,
                                    rotation,
                                    translation,
                                    scale,
                                )
                                open3d_mesh = trimesh_to_open3d(transformed_mesh)
                                
                                if frame_camera_translation and frame_camera_rotation:
                                    open3d_mesh = adjust_pose_by_camera_pose(open3d_mesh, frame_camera_translation, frame_camera_rotation)

                                meshes_to_draw.append(open3d_mesh)
                            except Exception as e:
                                print(f"Error loading mesh from {unposed_mesh_path}: {e}")

        
        # add character
        if show_poses:
            placeholder = get_character_placeholder()
            axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

            if frame_camera_translation and frame_camera_rotation:
                placeholder = adjust_pose_by_camera_pose(placeholder, frame_camera_translation, frame_camera_rotation)
                axis = adjust_pose_by_camera_pose(axis, frame_camera_translation, frame_camera_rotation)

            meshes_to_draw.append(placeholder)
            meshes_to_draw.append(axis)

        if show_landmarks:
            global world_landmarks
            landmarks = frame_data.get('landmarks', [])
            world_landmarks.extend(landmarks)
            if world_landmarks:
                landmarks_pointcloud = _point_cloud_from_landmarks(world_landmarks)
                meshes_to_draw.append(landmarks_pointcloud)
        
        # draw geometries
        vis.clear_geometries()    
        for mesh in meshes_to_draw:
            vis.add_geometry(mesh, reset_bounding_box=frame_idx == 0)
        for mesh in static_geometries:
            vis.add_geometry(mesh, reset_bounding_box=frame_idx == 0) 
        
        print(f"Frame {frame_idx + 1}/{len(frames_data)}")
    
    def previous_frame(vis):
        if state['frame_idx'] > 0:
            state['frame_idx'] -= 1
            update_frame()
        return False
    
    def next_frame(vis):
        if state['frame_idx'] < len(frames_data) - 1:
            state['frame_idx'] += 1
            update_frame()
        return False
    
    def toggle_play(vis):
        state['playing'] = not state['playing']
        if state['playing']:
            print("Playing...")
        else:
            print("Paused")
        return False
    
    vis.register_key_callback(ord('A'), previous_frame)
    vis.register_key_callback(ord('a'), previous_frame)
    vis.register_key_callback(ord('D'), next_frame)
    vis.register_key_callback(ord('d'), next_frame)
    vis.register_key_callback(32, toggle_play)
    
    update_frame()
    
    while True:
        if state['playing']:
            if video_writer is None and output_video_path:
                video_writer = _setup_video_writer(output_video_path, vis, fps)
            
            if state['frame_idx'] < len(frames_data) - 1:
                state['frame_idx'] += 1
                update_frame()
                
                if video_writer:
                    frame_image = _render_frame(vis)
                    video_writer.write(frame_image)
                    
                time.sleep(1.0 / fps)
            else:
                state['playing'] = False
                if video_writer:
                    video_writer.release()
                    print(f"Video saved to: {output_video_path}")
                    video_writer = None
                print("Playback finished")
        
        if not vis.poll_events():
            break
        vis.update_renderer()
    
    vis.destroy_window()
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize tracked 3D bounding boxes for objects."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input pickle file containing tracked object data"
    )
    
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Path to output video file (MP4). If provided, will render all frames and save as video."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=24,
        help="Frames per second for playback and output video."
    )

    parser.add_argument(
        "--no_poses",
        action="store_true",
        help="Disable camera pose visualization (character, axis, landmarks)."
    )

    parser.add_argument(
        "--no_reconstructed",
        action="store_true",
        help="Disable reconstructed mesh visualization."
    )

    parser.add_argument(
        "--no_landmarks",
        action="store_true",
        help="Disable landmark point-cloud visualization."
    )

    parser.add_argument(
        "--object_scale",
        type=float,
        default=2.0,
        help="Global scale multiplier for rendered objects."
    )

    parser.add_argument(
        "--disable_rotation",
        action="store_true",
        help="Disable reconstructed object rotations by using identity rotation."
    )
    
    args = parser.parse_args()
    do_visualization(
        args.input,
        args.output_video,
        fps=args.fps,
        show_poses=not args.no_poses,
        show_reconstructed=not args.no_reconstructed,
        show_landmarks=not args.no_landmarks,
        object_scale=args.object_scale,
        disable_rotation=args.disable_rotation,
    )