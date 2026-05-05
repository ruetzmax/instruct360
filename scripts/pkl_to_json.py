import argparse
import json
import pickle
from pathlib import Path
import numpy as np

def _to_jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _matrix_to_quaternion(rotation_matrix):
    rot = np.asarray(rotation_matrix, dtype=float)
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    return [x, y, z, w]


def _rotation_to_quaternion(rotation):
    if rotation is None:
        return None

    rot_array = np.array(rotation, dtype=float)
    if rot_array.shape in {(4,), (4, 1), (1, 4)}:
        return _to_jsonable(rot_array.reshape(-1).tolist())
    if rot_array.shape == (3, 3):
        quat = _matrix_to_quaternion(rot_array)
        return _to_jsonable(quat)
    if rot_array.size == 9:
        rot_matrix = rot_array.reshape(3, 3)
        quat = _matrix_to_quaternion(rot_matrix)
        return _to_jsonable(quat)
    return _to_jsonable(rotation)


def _simplify_mesh(mesh_dict):
    return {
        'scale': _to_jsonable(mesh_dict.get('scale')),
        'rotation': _rotation_to_quaternion(mesh_dict.get('rotation')),
        'translation': _to_jsonable(mesh_dict.get('translation')),
        'rotation_kalman': _rotation_to_quaternion(mesh_dict.get('rotation_kalman')),
        'translation_kalman': _to_jsonable(mesh_dict.get('translation_kalman')),
        'chunk_relative_rotation': _rotation_to_quaternion(mesh_dict.get('chunk_relative_rotation')),
        'chunk_relative_translation': _to_jsonable(mesh_dict.get('chunk_relative_translation')),
    }


def _simplify_class(class_dict):
    reconstructed_meshes = class_dict.get('reconstructed_meshes', [])
    if not isinstance(reconstructed_meshes, list):
        reconstructed_meshes = []

    return {
        'class_name': class_dict.get('class_name'),
        'reconstructed_meshes': [
            _simplify_mesh(mesh)
            for mesh in reconstructed_meshes
            if isinstance(mesh, dict)
        ],
    }


def _simplify_frame(frame_dict):
    classes = frame_dict.get('classes', [])
    if not isinstance(classes, list):
        classes = []

    return {
        'camera_translation': _to_jsonable(frame_dict.get('camera_translation')),
        'camera_rotation': _rotation_to_quaternion(frame_dict.get('camera_rotation')),
        'classes': [
            _simplify_class(class_dict)
            for class_dict in classes
            if isinstance(class_dict, dict)
        ],
    }


def convert_pkl_to_json(input_path, output_path):
    with open(input_path, 'rb') as handle:
        data = pickle.load(handle)

    if not isinstance(data, list):
        raise TypeError(f'Expected the pickle to contain a list of frames, got {type(data).__name__}')

    frames = [
        _simplify_frame(frame)
        for frame in data
        if isinstance(frame, dict)
    ]

    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(frames, handle, indent=2)
        handle.write('\n')

    
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a tracked pose pickle into a JSON file with camera pose and reconstructed mesh metadata.'
    )
    parser.add_argument('--input', type=Path, help='Path to the input .pkl file')
    parser.add_argument(
        '--output',
        type=Path,
        help='Path to the output .json file.',
    )
    
    args = parser.parse_args()
    convert_pkl_to_json(args.input, args.output)
    print(f'Wrote JSON to {args.output}')