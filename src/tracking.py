from src.operations2d import get_2d_bounding_boxes, bounding_boxes_to_image_chunks
from src.operations3d import get_3d_bounding_boxes, adjust_rotation_by_chunk_rotation, get_box_meshes
from src.util import read_video_frames, get_color_by_index, mesh_to_dict
from tqdm import tqdm

def get_bounding_boxes_for_class(image, class_name, threshold_2d=0.25, threshold_3d=0.3):
    bounding_boxes_2d = get_2d_bounding_boxes(image, class_name, threshold=threshold_2d)
    
    image_chunks = bounding_boxes_to_image_chunks(image, bounding_boxes_2d)
    
    all_bb_centers = []
    all_bb_dimensions = []
    all_bb_poses = []
    for chunk in image_chunks:
        bb_centers, bb_dimensions, bb_poses = get_3d_bounding_boxes(chunk, class_name, threshold=threshold_3d)
        bb_centers_adjusted, bb_poses_adjusted = adjust_rotation_by_chunk_rotation(bb_centers, bb_poses, chunk)
        all_bb_centers.extend(bb_centers_adjusted)
        all_bb_dimensions.extend(bb_dimensions)
        all_bb_poses.extend(bb_poses_adjusted)
        
    return all_bb_centers, all_bb_dimensions, all_bb_poses

def track_objects_in_video(classes, threshold_2d=0.25, threshold_3d=0.3, export_meshes=False, colors=None, video_path=None, left_video_path=None, right_video_path=None):
    
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
                frame, class_name, threshold_2d=threshold_2d, threshold_3d=threshold_3d
            )
            
            bb_centers, bb_dimensions, bb_poses = bbs
            
            if not bb_centers:
                continue
            
            
            class_result = {
                'class_name': class_name,
                'centers': bb_centers,
                'dimensions': bb_dimensions,
                'poses': bb_poses
            }
            
            if export_meshes:
                color = colors[class_idx] if colors else get_color_by_index(class_idx)
                bb_meshes = get_box_meshes(bbs, color=color)
                bb_mesh_dicts = [mesh_to_dict(mesh) for mesh in bb_meshes]
                class_result['meshes'] = bb_mesh_dicts

            
            frame_result['classes'].append(class_result)
            
        frame_results.append(frame_result)
    
    return frame_results