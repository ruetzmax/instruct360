import subprocess
import os

from matplotlib import image
import msgpack

from src.operations2d import get_2d_bounding_boxes, bounding_boxes_to_image_chunks, get_masks_from_image_chunks, image_chunk_from_undistorted
from src.operations3d import get_3d_bounding_boxes, adjust_bounding_boxes_by_chunk_rotation, get_box_meshes, reconstruct_meshes_for_chunks, adjust_transforms_by_chunk_rotation, apply_mesh_transforms
from src.util import read_video_frames, get_color_by_index, mesh_to_dict
from tqdm import tqdm

def get_bounding_boxes_for_class(
    image,
    class_name,
    threshold_2d=0.25,
    threshold_3d=0.3,
    orientation='vertical',
    video_format='equirectangular',
    use_gpu=False,
):
    bounding_boxes_2d = get_2d_bounding_boxes(
        image,
        class_name,
        threshold=threshold_2d,
        use_gpu=use_gpu,
    )
    
    
    if video_format == 'equirectangular':
        image_chunks = bounding_boxes_to_image_chunks(image, bounding_boxes_2d, orientation=orientation)
        
        all_3d_bb_centers = []
        all_3d_bb_dimensions = []
        all_3d_bb_poses = []
        for chunk in image_chunks:
            bb_3d_centers, bb_3d_dimensions, bb_3d_poses = get_3d_bounding_boxes(chunk, class_name, threshold=threshold_3d)
            bb_3d_centers_adjusted, bb_3d_poses_adjusted = adjust_bounding_boxes_by_chunk_rotation(bb_3d_centers, bb_3d_poses, chunk)
            all_3d_bb_centers.extend(bb_3d_centers_adjusted)
            all_3d_bb_dimensions.extend(bb_3d_dimensions)
            all_3d_bb_poses.extend(bb_3d_poses_adjusted)
    elif video_format == 'undistorted':
        chunk = image_chunk_from_undistorted(image)
        all_3d_bb_centers, all_3d_bb_dimensions, all_3d_bb_poses = get_3d_bounding_boxes(chunk, class_name, threshold=threshold_3d)
        
    return bounding_boxes_2d, all_3d_bb_centers, all_3d_bb_dimensions, all_3d_bb_poses

def track_objects_in_video(
    classes,
    threshold_2d=0.25,
    threshold_3d=0.3,
    export_meshes=False,
    colors=None,
    video_path=None,
    left_video_path=None,
    right_video_path=None,
    orientation='vertical',
    video_format='equirectangular',
    use_gpu=False,
):
    
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
                frame,
                class_name,
                threshold_2d=threshold_2d,
                threshold_3d=threshold_3d,
                orientation=orientation,
                video_format=video_format,
                use_gpu=use_gpu,
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

def track_camera_poses(video_path, video_format='equirectangular'):
    slam_executable = os.path.expanduser("~/lib/stella_vslam_examples/build/run_video_slam")
    config_file = "config/equirectangular.yaml" if video_format == 'equirectangular' else "config/undistorted.yaml"
    slam_command = [
        slam_executable,
        "-v", "config/orb_vocab.fbow",
        "-c", config_file,
        "-m", video_path,
        "--frame-skip", "1",
        "--temporal-mapping",
        "--viewer", "none",
        "--map-db-out", "temp/tracked.msg",
        "--eval-log-dir", "temp",
    ]
    subprocess.run(slam_command, check=True)
    
    camera_poses = []
    
    with open("temp/tracked.msg", "rb") as f:
        map_data = msgpack.unpack(f)
    
    # each row contains: timestamp tx ty tz qx qy qz qw
    with open("temp/frame_trajectory.txt", "r") as f:
        for line in f:
            values = line.strip().split()
            
            frame_translation = [float(values[1]), float(values[2]), float(values[3])]
            frame_rotation = [float(values[4]), float(values[5]), float(values[6]), float(values[7])]
            
            # look for corresponding keyframe and extract landmarks
            landmark_positions = []
            frame_timestamp = str(values[0])
            keyframe_id = None
            keyframes_dict = map_data.get("keyframes", {})
            for idx, keyframe in keyframes_dict.items():
                if str(keyframe["ts"]).startswith(frame_timestamp):
                    keyframe_id = int(idx)
                    break
            if keyframe_id is not None:
                landmarks_dict = map_data.get("landmarks", {})
                for landmark_id, landmark in landmarks_dict.items():
                    if landmark["ref_keyfrm"] == keyframe_id:
                        pos = landmark["pos_w"]
                        landmark_positions.append(pos)
                                                
            frame_dict = {
                'translation': frame_translation,
                'rotation': frame_rotation,
                'landmarks': landmark_positions
            }

            camera_poses.append(frame_dict)
    return camera_poses
        
def reconstruct_meshes_for_class(
    image,
    class_name,
    generate_texture=False,
    use_gpu=True,
):
    bounding_boxes_2d = get_2d_bounding_boxes(
        image,
        class_name,
        use_gpu=use_gpu,
    )
    image_chunks = bounding_boxes_to_image_chunks(image, bounding_boxes_2d, orientation="horizontal")
    masks = get_masks_from_image_chunks(
        image_chunks,
        prompt=class_name,
        use_gpu=use_gpu,
    )
    posed_meshes, unposed_meshes, rotations, translations = reconstruct_meshes_for_chunks(
        image_chunks,
        masks,
        generate_texture=generate_texture,
    )
    
    # Adjust transforms by chunk rotation
    adjusted_meshes = []
    adjusted_rotations_list = []
    adjusted_translations_list = []
    
    for chunk, unposed_mesh, rot, trans in zip(image_chunks, unposed_meshes, rotations, translations):
        adjusted_rot, adjusted_trans = adjust_transforms_by_chunk_rotation([rot], [trans], chunk)
        adjusted_meshes.append(apply_mesh_transforms(unposed_mesh, adjusted_rot[0], adjusted_trans[0]))
        adjusted_rotations_list.append(adjusted_rot[0])
        adjusted_translations_list.append(adjusted_trans[0])
    
    return unposed_meshes, adjusted_meshes, adjusted_rotations_list, adjusted_translations_list
