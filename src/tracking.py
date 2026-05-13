import subprocess
import os

import msgpack

from src.filters import init_kalman, do_kalman_step
from src.operations2d import ImageChunk, get_2d_bounding_boxes, bounding_boxes_to_image_chunks, get_masks_from_image_chunks, find_closest_image_chunk
from src.operations3d import estimate_intrinsics_for_chunk, estimate_pose_for_image_chunk, reconstruct_meshes_for_chunks, adjust_transforms_by_chunk_rotation, apply_mesh_transforms, sam3d_transforms_to_trimesh
from src.util import read_trimesh, read_video_frames
import numpy as np

def track_camera_poses(video_path, video_format='equirectangular'):
    os.makedirs("temp", exist_ok=True)
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
    generate_texture=True,
):
    bounding_boxes_2d = get_2d_bounding_boxes(
        image,
        class_name,
        use_persistent_runner=False # shut down model after use to save some VRAM. Sam3d is biig...
    )
    image_chunks = bounding_boxes_to_image_chunks(image, bounding_boxes_2d, orientation="horizontal")
    masks = get_masks_from_image_chunks(
        image_chunks,
        prompt=class_name,
        use_persistent_runner=False 
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
        initial_image_chunk_size,
        frame_camera_translations,
        frame_camera_rotations,
        sam3d_to_metric_scale_factor=1.5,
        fps=24,
    ):
    os.makedirs("temp", exist_ok=True)
    initial_chunk_relative_scale = np.asarray(initial_chunk_relative_scale, dtype=np.float32)
    tracked_chunk_relative_scales = [initial_chunk_relative_scale]
    tracked_chunk_relative_rotations = [initial_chunk_relative_rotation]
    tracked_chunk_relative_translations = [initial_chunk_relative_translation]
    tracked_world_rotations = [initial_world_rotation]
    tracked_world_translations = [initial_world_translation]
    tracked_world_rotations_kalman = []
    tracked_world_translations_kalman = []
    tracked_image_chunk_centers = [initial_image_chunk_center]
    tracked_image_chunk_sizes = [initial_image_chunk_size]

    frames = read_video_frames(video_path)
    
    # read and scale mesh
    unposed_mesh = read_trimesh(unposed_mesh_path)
    identity_rotation = np.eye(3, dtype=np.float32)
    zero_translation = np.zeros(3, dtype=np.float32)
    scaled_mesh = apply_mesh_transforms(
        unposed_mesh,
        identity_rotation,
        zero_translation,
        initial_chunk_relative_scale * sam3d_to_metric_scale_factor,
    )
    scaled_mesh_path = 'temp/scaled.glb'
    scaled_mesh.export(scaled_mesh_path)

    # reconstruct initial chunk
    initial_image_chunk = ImageChunk.from_image_point(frames[0], initial_image_chunk_center, initial_image_chunk_size)
    K = estimate_intrinsics_for_chunk(initial_image_chunk)
    
    kf = init_kalman(1.0 / fps)
    initial_world_rotation_kalman, initial_world_translation_kalman = do_kalman_step(
        kf,
        initial_world_rotation,
        initial_world_translation,
    )
    tracked_world_rotations_kalman.append(initial_world_rotation_kalman)
    tracked_world_translations_kalman.append(initial_world_translation_kalman)
    
    previous_image_chunk = initial_image_chunk
    previous_chunk_relative_rotation = initial_chunk_relative_rotation
    previous_chunk_relative_translation = initial_chunk_relative_translation
    previous_world_rotation = initial_world_rotation_kalman
    previous_world_translation = initial_world_translation_kalman
    depth_debug_dir = os.path.join("temp", "depth_debug")
    foundationpose_debug_dir = os.path.join("temp", "foundationpose_debug")
    os.makedirs(depth_debug_dir, exist_ok=True)
    os.makedirs(foundationpose_debug_dir, exist_ok=True)
    
    for frame_idx in range(1, len(frames)):
        print(f"Tracking mesh in frame {frame_idx}/{len(frames)-1}")
        next_frame = frames[frame_idx]
        
        # look for the object in the whole frame and get most similar image chunk
        next_frame_bb2ds = get_2d_bounding_boxes(next_frame, class_name)
        image_chunk_candidates = bounding_boxes_to_image_chunks(next_frame, next_frame_bb2ds, orientation="horizontal", chunk_size=initial_image_chunk_size)
        next_image_chunk = find_closest_image_chunk(previous_image_chunk, image_chunk_candidates)

        # if no chunk could be found (ie. the object is not visible or too distorted) use last frames values and continue looking next frame
        if next_image_chunk is None:
            next_image_chunk = ImageChunk.from_image_point(next_frame, previous_image_chunk.center, initial_image_chunk_size)
            tracked_chunk_relative_scales.append(tracked_chunk_relative_scales[-1])
            tracked_chunk_relative_rotations.append(tracked_chunk_relative_rotations[-1])
            tracked_chunk_relative_translations.append(tracked_chunk_relative_translations[-1])
            tracked_world_rotations.append(tracked_world_rotations[-1])
            tracked_world_translations.append(tracked_world_translations[-1])
            tracked_world_rotations_kalman.append(tracked_world_rotations_kalman[-1])
            tracked_world_translations_kalman.append(tracked_world_translations_kalman[-1])
            tracked_image_chunk_centers.append(next_image_chunk.center)
            tracked_image_chunk_sizes.append(next_image_chunk.image.shape[:2])
            previous_image_chunk = next_image_chunk
            continue

        # run FoundationPose
        next_rotation_chunk, next_translation_chunk = estimate_pose_for_image_chunk(
            chunk=next_image_chunk,
            unposed_mesh_path=scaled_mesh_path,
            class_name=class_name,
            K=K,
            depth_debug_image_path=os.path.join(depth_debug_dir, "debug_depth_frame.png") if frame_idx==1 else None,
            foundationpose_debug_image_path=os.path.join(
                foundationpose_debug_dir,
                "debug_pose_frame.png",
            ),
            foundationpose_debug_level=3 if frame_idx == 1 else 0,
        )

        # rotate pose by -90 degrees around x
        angle_x = np.deg2rad(90.0)
        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x),  np.cos(angle_x)],
        ])
        next_rotation_chunk = rot_x @ next_rotation_chunk
        next_translation_chunk = rot_x @ next_translation_chunk

        # convert chunk-relative transforms to world-relative transforms
        next_rotation_world, next_translation_world = adjust_transforms_by_chunk_rotation([next_rotation_chunk], [next_translation_chunk], next_image_chunk)
        next_rotation_world = next_rotation_world[0]
        next_translation_world = next_translation_world[0] 
        
        # apply kalman filter to world transforms
        next_rotation_world_kalman, next_translation_world_kalman = do_kalman_step(
            kf,
            next_rotation_world,
            next_translation_world,
        )
        
        # store results
        tracked_chunk_relative_scales.append(initial_chunk_relative_scale)
        tracked_chunk_relative_rotations.append(next_rotation_chunk)
        tracked_chunk_relative_translations.append(next_translation_chunk)
        tracked_world_rotations.append(next_rotation_world)
        tracked_world_translations.append(next_translation_world)
        tracked_world_rotations_kalman.append(next_rotation_world_kalman)
        tracked_world_translations_kalman.append(next_translation_world_kalman)
        tracked_image_chunk_centers.append(next_image_chunk.center)
        tracked_image_chunk_sizes.append(initial_image_chunk_size)
        
        previous_image_chunk = next_image_chunk
        previous_chunk_relative_rotation = next_rotation_chunk
        previous_chunk_relative_translation = next_translation_chunk
        previous_world_rotation = next_rotation_world_kalman
        previous_world_translation = next_translation_world_kalman
        
    return (
        tracked_chunk_relative_scales,
        tracked_chunk_relative_rotations,
        tracked_chunk_relative_translations,
        tracked_world_rotations,
        tracked_world_translations,
        tracked_world_rotations_kalman,
        tracked_world_translations_kalman,
        tracked_image_chunk_centers,
        tracked_image_chunk_sizes,
    )