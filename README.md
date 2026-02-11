# About
Extract 3d bounding boxes for arbitrary classes from a equirectangular panorama images / videos.

# Setup
1. Clone the repo and cd into it
2. `conda create -n instruct360 python=3.8.20 && conda activate instruct360`
3. `pip install -r requirements.txt`
4. Follow the [Ovmono3D installation instructions](https://github.com/UVA-Computer-Vision-Lab/ovmono3d/tree/main?tab=readme-ov-file#installation-) inside of the root folder, but use the instruct360 conda environment. (For CPU execution, use [this fork](https://github.com/ruetzmax/ovmono3d) instead)
5. Follow the [GroundingDINO installation instructions](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install) inside of the root folder
6. Install [stella_vslam](https://stella-cv.readthedocs.io/en/latest/installation.html)

# Inference
To infere 3D object bounding boxes from a video, run:
`python scripts/track_objects_in_video.py --video_path vids/office.mp4 --classes "cupboard" "cup" "chair" --threshold_2d 0.25 --threshold_3d 0.3 --export_meshes True --output output/office_objects.pkl`

To visualize the results, run:
`python scripts/visualize_tracked_video.py --input output/office_objects.pkl`
To draw 2d bounding boxes, run:
`python scripts/visualize_2d_bounding_boxes.py --input_video vids/office.mp4 --object_pkl output/office_objects.pkl --output_video output/boxes2d.mp4`
To append camera poses to already tracked data (will be automatically considered in visualization), run: 
`python scripts/append_camera_poses.py --input_video vids/office.mp4 --input_pkl output/office_objects.pkl --output_pkl output/office_objects_with_poses.pkl`
