import subprocess
import os

from matplotlib import image
import msgpack

from src.operations2d import ImageChunk, find_similar_image_chunk, get_2d_bounding_boxes, bounding_boxes_to_image_chunks, get_masks_from_image_chunks, image_chunk_from_undistorted
from src.operations3d import estimate_intrinsics_for_chunk, get_3d_bounding_boxes, adjust_bounding_boxes_by_chunk_rotation, get_box_meshes, get_intrinsics_for_chunk, reconstruct_meshes_for_chunks, adjust_transforms_by_chunk_rotation, apply_mesh_transforms, sam3d_transforms_to_trimesh
from src.util import opencv_to_trimesh_pose, read_trimesh, read_video_frames, get_color_by_index, mesh_to_dict, trimesh_to_opencv
from tqdm import tqdm
import numpy as np
import cv2

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
    posed_meshes, unposed_meshes, scales, rotations, translations = reconstruct_meshes_for_chunks(
        image_chunks,
        masks,
        generate_texture=generate_texture,
    )
    
    # Adjust transforms by chunk rotation
    adjusted_meshes = []
    adjusted_scales_list = []
    adjusted_rotations_list = []
    adjusted_translations_list = []
    chunk_relative_scales_list = []
    chunk_relative_rotations_list = []
    chunk_relative_translations_list = []
    image_chunk_centers = []
    image_chunk_sizes = []
    
    for chunk, unposed_mesh, scale, rot, trans in zip(image_chunks, unposed_meshes, scales, rotations, translations):
        trimesh_scale, trimesh_rot, trimesh_trans = sam3d_transforms_to_trimesh(scale, rot, trans)
        chunk_adjusted_rot, chunk_adjusted_trans = adjust_transforms_by_chunk_rotation([trimesh_rot], [trimesh_trans], chunk)
        adjusted_meshes.append(apply_mesh_transforms(unposed_mesh, chunk_adjusted_rot[0], chunk_adjusted_trans[0], trimesh_scale))
        chunk_relative_scales_list.append(trimesh_scale)
        chunk_relative_rotations_list.append(trimesh_rot)
        chunk_relative_translations_list.append(trimesh_trans)
        adjusted_scales_list.append(trimesh_scale)
        adjusted_rotations_list.append(chunk_adjusted_rot[0])
        adjusted_translations_list.append(chunk_adjusted_trans[0])
        image_chunk_centers.append(chunk.center)
        image_chunk_sizes.append((int(chunk.image.shape[1]), int(chunk.image.shape[0])))

    return (
        unposed_meshes,
        adjusted_meshes,
        adjusted_scales_list,
        adjusted_rotations_list,
        adjusted_translations_list,
        image_chunk_centers,
        image_chunk_sizes,
        chunk_relative_scales_list,
        chunk_relative_rotations_list,
        chunk_relative_translations_list,
    )

def track_object_poses_for_mesh(
        video_path,
        class_name,
        unposed_mesh_path,
        initial_chunk_relative_scale,
        initial_chunk_relative_rotation,
        initial_chunk_relative_translation,
        initial_world_rotation,
        initial_world_translation,
        initial_image_chunk_center,
        initial_image_chunk_size
    ):
    tracked_chunk_relative_scales = [initial_chunk_relative_scale]
    tracked_chunk_relative_rotations = [initial_chunk_relative_rotation]
    tracked_chunk_relative_translations = [initial_chunk_relative_translation]
    tracked_world_rotations = [initial_world_rotation]
    tracked_world_translations = [initial_world_translation]
    tracked_image_chunk_centers = [initial_image_chunk_center]
    tracked_image_chunk_sizes = [initial_image_chunk_size]

    frames = read_video_frames(video_path)
    
    # read and scale mesh
    unposed_mesh = read_trimesh(unposed_mesh_path)
    identity_rotation = np.eye(3, dtype=np.float32)
    zero_translation = np.zeros(3, dtype=np.float32)
    scaled_mesh = apply_mesh_transforms(unposed_mesh, identity_rotation, zero_translation, initial_chunk_relative_scale)
    
    # reconstruct initial chunk
    initial_image_chunk = ImageChunk.from_image_point(frames[0], initial_image_chunk_center, initial_image_chunk_size)
    K = estimate_intrinsics_for_chunk(initial_image_chunk)
    
    previous_image_chunk = initial_image_chunk
    previous_chunk_relative_rotation = initial_chunk_relative_rotation
    previous_chunk_relative_translation = initial_chunk_relative_translation
    for frame_idx in range(1, len(frames)):
        print(f"Tracking mesh in frame {frame_idx}/{len(frames)-1}")
        next_frame = frames[frame_idx]
        
        # get image chunk similar to the previous one
        next_frame_bb2ds = get_2d_bounding_boxes(next_frame, class_name)
        image_chunk_candidates = bounding_boxes_to_image_chunks(next_frame, next_frame_bb2ds, orientation="horizontal")
        next_image_chunk = find_similar_image_chunk(previous_image_chunk, image_chunk_candidates)
        
        # if none can be found, use the previous one
        if next_image_chunk is None:
            next_image_chunk = previous_image_chunk.copy()
        
        # get transforms inside the previous chunk in OpenCV coordinates, so we can project them into the next frame
        cv_vertices, cv_tris, cv_rotation_mat, cv_translation = trimesh_to_opencv(
            scaled_mesh,
            rotation_matrix=previous_chunk_relative_rotation,
            translation_vector=previous_chunk_relative_translation,
        )
        cv_rotation, _ = cv2.Rodrigues(cv_rotation_mat)
        
        # perform a RAPID step to get the new rotation and translation
        _, cv_rotation_new, cv_translation_new, _ = cv2.rapid.rapid(
            img=next_image_chunk.image,
            num=30,
            len=50,
            pts3d=cv_vertices,
            tris=cv_tris,
            K=K,
            rvec=cv_rotation,
            tvec=cv_translation
        )
        
        # convert back to the format used by the renderer
        next_rotation_cv, _ = cv2.Rodrigues(cv_rotation_new)
        next_translation_cv = cv_translation_new.reshape(3)
        next_rotation_chunk, next_translation_chunk = opencv_to_trimesh_pose(
            next_rotation_cv,
            next_translation_cv,
        )

        # convert chunk-relative transforms to world-relative transforms
        next_rotation_world, next_translation_world = adjust_transforms_by_chunk_rotation([next_rotation_chunk], [next_translation_chunk], next_image_chunk)
        next_rotation_world = next_rotation_world[0]
        next_translation_world = next_translation_world[0]
        
        # store results
        tracked_chunk_relative_scales.append(initial_chunk_relative_scale)
        tracked_chunk_relative_rotations.append(next_rotation_chunk)
        tracked_chunk_relative_translations.append(next_translation_chunk)
        tracked_world_rotations.append(next_rotation_world)
        tracked_world_translations.append(next_translation_world)
        tracked_image_chunk_centers.append(next_image_chunk.center)
        tracked_image_chunk_sizes.append(initial_image_chunk_size)
        
        previous_image_chunk = next_image_chunk
        previous_chunk_relative_rotation = next_rotation_chunk
        previous_chunk_relative_translation = next_translation_chunk
        
    return (
        tracked_chunk_relative_scales,
        tracked_chunk_relative_rotations,
        tracked_chunk_relative_translations,
        tracked_world_rotations,
        tracked_world_translations,
        tracked_image_chunk_centers,
        tracked_image_chunk_sizes,
    )