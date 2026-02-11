
import argparse
import time
from pathlib import Path
import pickle
import sys
import open3d
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operations3d import adjust_pose_by_camera_pose
from src.util import dict_to_mesh, get_character_placeholder

FPS = 24

class_colors = {
    'cupboard': [1.0, 0.0, 0.0],
    'cup': [0.0, 1.0, 0.0],
    'chair': [0.0, 0.0, 1.0],
}

def _setup_video_writer(output_video_path, vis):
    setup_image = vis.capture_screen_float_buffer(do_render=False)
    setup_image_np = np.asarray(setup_image)
    height, width = setup_image_np.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (width, height))

    return video_writer     

def _render_frame(vis):
    image = vis.capture_screen_float_buffer(do_render=False)
    image_np = np.asarray(image)
    image_np = (image_np * 255).astype(np.uint8)
    
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

def do_visualization(object_pkl_path: str, output_video_path: str = None):
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
    
    video_writer = None     
    
    state = {'frame_idx': 0, 'playing': False}
    
    def update_frame():
        frame_idx = state['frame_idx']
        frame_data = frames_data[frame_idx]
        
        frame_camera_translation = frame_data.get('camera_translation', None)
        frame_camera_rotation = frame_data.get('camera_rotation', None)
        
        frame_meshes = []
        
        class_dicts = frame_data['classes']
        for class_dict in class_dicts:
            class_name = class_dict['class_name']
            if 'meshes' not in class_dict:
                continue
            mesh_dicts = class_dict['meshes']
            for mesh_dict in mesh_dicts:
                mesh = dict_to_mesh(mesh_dict)
                
                if class_name in class_colors:
                    mesh.paint_uniform_color(class_colors[class_name])
                    
                if frame_camera_translation and frame_camera_rotation:
                    mesh = adjust_pose_by_camera_pose(mesh, frame_camera_translation, frame_camera_rotation)
                
                frame_meshes.append(mesh)
        
        placeholder = get_character_placeholder()
        axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        if frame_camera_translation and frame_camera_rotation:
            placeholder = adjust_pose_by_camera_pose(placeholder, frame_camera_translation, frame_camera_rotation)
            axis = adjust_pose_by_camera_pose(axis, frame_camera_translation, frame_camera_rotation)
            
        frame_meshes.append(placeholder)
        frame_meshes.append(axis)
        
        vis.clear_geometries()    
        for mesh in frame_meshes:
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
                video_writer = _setup_video_writer(output_video_path, vis)
            
            if state['frame_idx'] < len(frames_data) - 1:
                state['frame_idx'] += 1
                update_frame()
                
                if video_writer:
                    frame_image = _render_frame(vis)
                    video_writer.write(frame_image)
                    
                time.sleep(1.0 / FPS)
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
    
    args = parser.parse_args()
    do_visualization(args.input, args.output_video)