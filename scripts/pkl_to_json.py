import argparse
import json
import pickle
from pathlib import Path

def _to_jsonable(value):
    try:
        import numpy as np
    except Exception:
        np = None

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _simplify_mesh(mesh_dict):
    return {
        'scale': _to_jsonable(mesh_dict.get('scale')),
        'rotation': _to_jsonable(mesh_dict.get('rotation')),
        'translation': _to_jsonable(mesh_dict.get('translation')),
        'chunk_relative_rotation': _to_jsonable(mesh_dict.get('chunk_relative_rotation')),
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
        'camera_rotation': _to_jsonable(frame_dict.get('camera_rotation')),
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