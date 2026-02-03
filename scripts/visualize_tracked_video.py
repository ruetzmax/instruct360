
import argparse
import time
from pathlib import Path
import pickle
import sys
import open3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.util import dict_to_mesh, get_character_placeholder

def do_visualization(object_pkl_path: str):
    with open(object_pkl_path, 'rb') as f:
        frames_data = pickle.load(f)
        
    print(f"Loaded {len(frames_data)} frames")
    print("Controls: 'a' - previous frame, 'd' - next frame, 'q' - quit")
        
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    state = {'frame_idx': 0}
    
    def update_frame():
        frame_idx = state['frame_idx']
        frame_data = frames_data[frame_idx]
        
        frame_meshes = []
        
        class_dicts = frame_data['classes']
        for class_dict in class_dicts:
            if 'meshes' not in class_dict:
                continue
            mesh_dicts = class_dict['meshes']
            for mesh_dict in mesh_dicts:
                mesh = dict_to_mesh(mesh_dict)
                frame_meshes.append(mesh)
        
        placeholder = get_character_placeholder()
        frame_meshes.append(placeholder)
        
        vis.clear_geometries()    
        for mesh in frame_meshes:
            vis.add_geometry(mesh, reset_bounding_box=True) 
        
        vis.poll_events()
        vis.update_renderer()
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
    
    vis.register_key_callback(ord('A'), previous_frame)
    vis.register_key_callback(ord('a'), previous_frame)
    vis.register_key_callback(ord('D'), next_frame)
    vis.register_key_callback(ord('d'), next_frame)
    
    update_frame()
    
    vis.run()
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
    
    args = parser.parse_args()
    do_visualization(args.input)