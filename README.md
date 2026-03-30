# About
Reconstruct scene information from omnidirectional video input. Pipeline allows for execution of arbitrary operations (=payloads) on perspective images of objects of interest, taken from the panorama. Multiple payloads are available. Ovmono extracts 3d bounding boxes for arbitrary classes. Sam3D reconstructs object meshes.

# Setup
1. Clone the repo and cd into it
2. `conda create -n instruct360 python=3.11.0 && conda activate instruct360`
3. `pip install -r requirements.txt` 
4. Follow the [Ovmono3D installation instructions](https://github.com/UVA-Computer-Vision-Lab/ovmono3d/tree/main?tab=readme-ov-file#installation-) inside of the root folder. (For CPU execution, use [this fork](https://github.com/ruetzmax/ovmono3d) instead)
5. Follow the [GroundingDINO installation instructions](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) inside of the root folder and inside a conda env called "grounding_dino. 
<!-- 4. Install [cuda toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) -->
6. Install [ov-seg](https://github.com/facebookresearch/ov-seg) and place the pretrained weights inside the ov-seg/checkpoints folder
7. Install [sam3d](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) in a separate conda env "sam3d-objects" but clone it into the project root

OPTIONAL - for SLAM / camera postion tracking
- Install [stella_vslam](https://stella-cv.readthedocs.io/en/latest/installation.html)

Possible Errors:
-  `ImportError: libtiff.so.5: cannot open shared object file...` ensure that libtiff is installed and downgrade PIL `pip install pillow==9.5.0"
-  you may need to installed opencv / additional packages in the respective payload environments

# Inference
To infere 3D object bounding boxes from a video, run:
`python scripts/track_objects_in_video.py --video_path vids/office.mp4 --classes "cupboard" "cup" "chair" --threshold_2d 0.35 --threshold_3d 0.4 --orientation horizontal --export_meshes True --output output/office_objects.pkl`

To visualize the results, run:
`python scripts/visualize_tracked_video.py --input output/office_objects.pkl`
To draw 2d bounding boxes, run:
`python scripts/visualize_2d_bounding_boxes.py --input_video vids/office.mp4 --object_pkl output/office_objects.pkl --output_video output/boxes2d.mp4`
To append camera poses to already tracked data (will be automatically considered in visualization), run: 
`python scripts/append_camera_poses.py --input_video vids/office.mp4 --input_pkl output/office_objects.pkl --output_pkl output/office_objects_with_poses.pkl`
