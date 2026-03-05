
import argparse
import time
from pathlib import Path
import pickle
import sys
import msgpack
import open3d
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operations3d import adjust_pose_by_camera_pose, get_box_meshes
from src.util import get_character_placeholder

FPS = 24

class_colors = {
    'cupboard': [1.0, 0.0, 0.0],
    'cup': [0.0, 1.0, 0.0],
    'chair': [0.0, 0.0, 1.0],
}

unique_distance_threshold = 1.5

dimension_change_threshold = 0.5
position_change_threshold = 0.2
pose_change_threshold = 0.3

unique_meshes_by_class = {}

world_landmarks = []

ENABLE_FILTERING = False

def _point_cloud_from_landmarks(landmarks):       
        if len(landmarks) == 0:
            return
                
        landmark_pointcloud = open3d.geometry.PointCloud()
        landmarks = [[pos[2], -pos[1], pos[0]] for pos in landmarks]
        landmarks_np = np.array(landmarks)
        landmark_pointcloud.points = open3d.utility.Vector3dVector(landmarks_np)
        landmark_pointcloud.paint_uniform_color([0.0, 0.0, 0.0])
        return landmark_pointcloud

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

def _is_same_object(mesh1, mesh2):
    global unique_distance_threshold
    # if two objects are very close to each other, we consider them the same
    center1 = mesh1.get_center()
    center2 = mesh2.get_center()
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    return distance < unique_distance_threshold

def _get_unique_index(mesh, class_name):
    global unique_meshes_by_class
    if class_name not in unique_meshes_by_class:
        unique_meshes_by_class[class_name] = []
    unique_meshes = unique_meshes_by_class[class_name]
    for idx, unique_mesh in enumerate(unique_meshes):
        if _is_same_object(mesh, unique_mesh):
            return idx
    unique_meshes.append(mesh)
    return len(unique_meshes) - 1

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
        
        frame_meshes = []
        meshes_to_draw = []
        
        class_dicts = frame_data['classes']
        for class_dict in class_dicts:
            class_name = class_dict['class_name']
            bb_centers = class_dict['centers']
            bb_dimensions = class_dict['dimensions']
            bb_poses = class_dict['poses']
            meshes = get_box_meshes((bb_centers, bb_dimensions, bb_poses))
            
            # filter out duplicate meshes
            for mesh_idx, mesh in enumerate(meshes):
                if frame_camera_translation and frame_camera_rotation:
                    mesh = adjust_pose_by_camera_pose(mesh, frame_camera_translation, frame_camera_rotation)
                    
                frame_meshes.append(mesh)
                    
                unique_idx = _get_unique_index(mesh, class_name)
                
                # only use new dimension/position/pose if it changes significantly
                global dimension_change_threshold, position_change_threshold, pose_change_threshold
                
                previous_mesh = unique_meshes_by_class[class_name][unique_idx]
                previous_dimension = previous_mesh.get_oriented_bounding_box().extent
                previous_center = previous_mesh.get_center()
                previous_pose = previous_mesh.get_oriented_bounding_box().R
                
                current_dimension = mesh.get_oriented_bounding_box().extent
                current_center = mesh.get_center()
                current_pose = mesh.get_oriented_bounding_box().R
                
                relative_dimension_change = abs(current_dimension - previous_dimension) / (previous_dimension + 1e-6)
                if relative_dimension_change.max() > dimension_change_threshold:
                    new_dimension = current_dimension
                else:
                    new_dimension = previous_dimension
                    
                absolute_position_change = np.linalg.norm(np.array(current_center) - np.array(previous_center))
                if absolute_position_change > position_change_threshold:
                    new_center = current_center
                else:
                    new_center = previous_center
                    
                absolute_pose_change = np.linalg.norm(current_pose - previous_pose)
                if absolute_pose_change > pose_change_threshold:
                    new_pose = current_pose
                else:
                    new_pose = previous_pose
                
                #create open3d oriented bb
                new_mesh = open3d.geometry.OrientedBoundingBox(new_center, new_pose, new_dimension)
                #convert to triangle mesh
                new_mesh = open3d.geometry.TriangleMesh.create_from_oriented_bounding_box(new_mesh)
                
                # new_mesh = get_box_mesh((current_center, new_dimension, current_pose))
                # new_mesh = adjust_pose_by_camera_pose(new_mesh, frame_camera_translation, frame_camera_rotation)
                
                unique_meshes_by_class[class_name][unique_idx] = new_mesh
                
                # unique_meshes_by_class[class_name][unique_idx] = mesh
          
        if ENABLE_FILTERING:
            # render all unique meshes  
            for class_name, unique_meshes in unique_meshes_by_class.items():
                for mesh in unique_meshes:  
                    if class_name in class_colors:
                        mesh.paint_uniform_color(class_colors[class_name])   
                    meshes_to_draw.append(mesh)
        else:
            #render all meshes
            for mesh in frame_meshes:
                if class_name in class_colors:
                        mesh.paint_uniform_color(class_colors[class_name])
                meshes_to_draw.append(mesh)
            
        
        placeholder = get_character_placeholder()
        axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        if frame_camera_translation and frame_camera_rotation:
            placeholder = adjust_pose_by_camera_pose(placeholder, frame_camera_translation, frame_camera_rotation)
            axis = adjust_pose_by_camera_pose(axis, frame_camera_translation, frame_camera_rotation)
            
        meshes_to_draw.append(placeholder)
        meshes_to_draw.append(axis)
        
        global world_landmarks
        landmarks = frame_data.get('landmarks', [])
        world_landmarks.extend(landmarks)
        landmarks_mesh = _point_cloud_from_landmarks(world_landmarks)
        
        vis.clear_geometries()    
        for mesh in meshes_to_draw:
            vis.add_geometry(mesh, reset_bounding_box=frame_idx == 0) 
            
        if landmarks_mesh is not None:
            vis.add_geometry(landmarks_mesh, reset_bounding_box=False) 
        
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