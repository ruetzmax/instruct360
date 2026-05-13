from pathlib import Path
import pickle
import sys
import argparse


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tracking import track_object_poses_for_mesh


def track_object_poses(
	input_video_path,
	input_pkl_path,
	classes,
	output_pkl_path=None,
	image_chunk_size=(300, 300),
	fps=24,
):
	with open(input_pkl_path, 'rb') as f:
		frames_data = pickle.load(f)

	if not isinstance(frames_data, list) or len(frames_data) == 0:
		raise ValueError("Input pickle must contain a non-empty list of frame data")

	frame_camera_translations = [frame["camera_translation"] for frame in frames_data]
	frame_camera_rotations = [frame["camera_rotation"] for frame in frames_data]

	first_frame_data = frames_data[0]
	first_frame_classes = first_frame_data.get('classes', [])
	class_filter = set(classes)

	tracked_mesh_count = 0
	tracked_results = []

	for class_entry in first_frame_classes:
		class_name = class_entry.get('class_name')
		if class_name not in class_filter:
			continue

		reconstructed_meshes = class_entry.get('reconstructed_meshes', [])
		if not isinstance(reconstructed_meshes, list):
			continue

		for mesh_data in reconstructed_meshes:
			required_fields = [
				'unposed_mesh_path',
				'chunk_relative_scale',
				'chunk_relative_rotation',
				'chunk_relative_translation',
				'rotation',
				'translation',
				'image_chunk_center',
				'image_chunk_size',
			]
			if any(field not in mesh_data for field in required_fields):
				continue

			(
				tracked_chunk_relative_scales,
				tracked_chunk_relative_rotations,
				tracked_chunk_relative_translations,
				tracked_world_rotations,
				tracked_world_translations,
				tracked_world_rotations_kalman,
				tracked_world_translations_kalman,
				tracked_image_chunk_centers,
				tracked_image_chunk_sizes,
			) = track_object_poses_for_mesh(
				video_path=input_video_path,
				class_name=class_name,
				unposed_mesh_path=mesh_data['unposed_mesh_path'],
				initial_chunk_relative_scale=mesh_data['chunk_relative_scale'],
				initial_chunk_relative_rotation=mesh_data['chunk_relative_rotation'],
				initial_chunk_relative_translation=mesh_data['chunk_relative_translation'],
				initial_world_rotation=mesh_data['rotation'],
				initial_world_translation=mesh_data['translation'],
				initial_image_chunk_center=mesh_data['image_chunk_center'],
				initial_image_chunk_size=image_chunk_size,
				frame_camera_translations=frame_camera_translations,
				frame_camera_rotations=frame_camera_rotations,
				fps=fps,
			)

			tracked_results.append({
				'class_name': class_name,
				'unposed_mesh_path': mesh_data['unposed_mesh_path'],
				'tracked_chunk_relative_scales': tracked_chunk_relative_scales,
				'tracked_chunk_relative_rotations': tracked_chunk_relative_rotations,
				'tracked_chunk_relative_translations': tracked_chunk_relative_translations,
				'tracked_world_rotations': tracked_world_rotations,
				'tracked_world_translations': tracked_world_translations,
				'tracked_world_rotations_kalman': tracked_world_rotations_kalman,
				'tracked_world_translations_kalman': tracked_world_translations_kalman,
				'tracked_image_chunk_centers': tracked_image_chunk_centers,
				'tracked_image_chunk_sizes': tracked_image_chunk_sizes,
			})
			tracked_mesh_count += 1

	for tracked_mesh in tracked_results:
		class_name = tracked_mesh['class_name']
		unposed_mesh_path = tracked_mesh['unposed_mesh_path']
		tracked_chunk_relative_scales = tracked_mesh['tracked_chunk_relative_scales']
		tracked_chunk_relative_rotations = tracked_mesh['tracked_chunk_relative_rotations']
		tracked_chunk_relative_translations = tracked_mesh['tracked_chunk_relative_translations']
		tracked_world_rotations = tracked_mesh['tracked_world_rotations']
		tracked_world_translations = tracked_mesh['tracked_world_translations']
		tracked_world_rotations_kalman = tracked_mesh['tracked_world_rotations_kalman']
		tracked_world_translations_kalman = tracked_mesh['tracked_world_translations_kalman']
		tracked_image_chunk_centers = tracked_mesh['tracked_image_chunk_centers']
		tracked_image_chunk_sizes = tracked_mesh['tracked_image_chunk_sizes']

		num_frames = len(tracked_chunk_relative_scales)
		if num_frames > len(frames_data):
			print(
				f"Warning: Tracking produced {num_frames} frames, but input pickle has {len(frames_data)}; "
				"truncating to pickle length."
			)
			num_frames = len(frames_data)

		for frame_idx in range(num_frames):
			frame_data = frames_data[frame_idx]
			frame_classes = frame_data.get('classes')
			if not isinstance(frame_classes, list):
				frame_classes = []
				frame_data['classes'] = frame_classes

			class_entry = None
			for existing_class in frame_classes:
				if isinstance(existing_class, dict) and existing_class.get('class_name') == class_name:
					class_entry = existing_class
					break

			if class_entry is None:
				class_entry = {'class_name': class_name, 'reconstructed_meshes': []}
				frame_classes.append(class_entry)

			reconstructed_meshes = class_entry.get('reconstructed_meshes')
			if not isinstance(reconstructed_meshes, list):
				reconstructed_meshes = []
				class_entry['reconstructed_meshes'] = reconstructed_meshes

			mesh_entry = {
				'unposed_mesh_path': unposed_mesh_path,
				'scale': tracked_chunk_relative_scales[frame_idx],
				'rotation': tracked_world_rotations[frame_idx],
				'translation': tracked_world_translations[frame_idx],
				'rotation_kalman': tracked_world_rotations_kalman[frame_idx],
				'translation_kalman': tracked_world_translations_kalman[frame_idx],
				'chunk_relative_scale': tracked_chunk_relative_scales[frame_idx],
				'chunk_relative_rotation': tracked_chunk_relative_rotations[frame_idx],
				'chunk_relative_translation': tracked_chunk_relative_translations[frame_idx],
				'image_chunk_center': tracked_image_chunk_centers[frame_idx],
				'image_chunk_size': tracked_image_chunk_sizes[frame_idx],
			}

			replaced = False
			for mesh_idx, existing_mesh in enumerate(reconstructed_meshes):
				if isinstance(existing_mesh, dict) and existing_mesh.get('unposed_mesh_path') == unposed_mesh_path:
					reconstructed_meshes[mesh_idx] = mesh_entry
					replaced = True
					break

			if not replaced:
				reconstructed_meshes.append(mesh_entry)

	if output_pkl_path is None:
		output_pkl_path = input_pkl_path

	with open(output_pkl_path, 'wb') as f:
		pickle.dump(frames_data, f)

	print(f"Tracked object poses for {tracked_mesh_count} meshes")
	print(f"Saved output with tracked object poses to: {output_pkl_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Track object poses for reconstructed meshes from the first frame"
	)
	parser.add_argument(
		"--input_video",
		type=str,
		required=True,
		help="Path to the input video file"
	)
	parser.add_argument(
		"--input_pkl",
		type=str,
		required=True,
		help="Path to the input pickle file containing reconstructed first-frame mesh data"
	)
	parser.add_argument(
		"--classes",
		type=str,
		nargs="+",
		required=True,
		help="List of object class names to track (e.g., 'chair' 'table')"
	)
	parser.add_argument(
		"--output_pkl",
		type=str,
		help="Path to save the output pickle file with tracked object poses"
	)
	parser.add_argument(
		"--image_chunk_size",
		type=int,
		nargs=2,
		default=[300, 300],
		help="Image chunk size as two integers: width height"
	)
	parser.add_argument(
		"--fps",
		type=int,
		default=24,
		help="Video"
	)

	args = parser.parse_args()

	track_object_poses(
		args.input_video,
		args.input_pkl,
		args.classes,
		args.output_pkl,
		image_chunk_size=tuple(args.image_chunk_size),
		fps=args.fps,
	)
