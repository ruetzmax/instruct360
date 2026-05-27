# About
Pipeline for object capturing from omnidirectional / 360 degree input videos. Includes camera pose tracking, mesh reconstruction and pose estimation.

https://github.com/user-attachments/assets/f5130f4a-4aa9-4756-a370-16e632fcab64

# Setup
1. Clone the repo and cd into it
2. `conda create -n instruct360 python=3.11.0 && conda activate instruct360`
3. `pip install -r requirements.txt` 
4. Install [stella_vslam](https://stella-cv.readthedocs.io/en/latest/installation.html)
5. Multiple foundation models will be executed during the pipeline. To avoid dependency conflicts, each model has to be installed in a separate virtual environment. Pay attention to the required CUDA versions (can be different for each environment). To use a different CUDA version, install it from the [Cuda Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) using a runfile to a specific folder (`sudo sh cuda_runfile.run --silent --toolkit --toolkitpath=/your/cuda/path`). To tell the conda environment which version to use, set the following env vars: `conda env config vars set CUDA_HOME=/your/cuda/path && conda env config vars set PATH=$CUDA_HOME/bin:$PATH && conda env config vars set LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`. Clone each of the following projects into the root of this repository and install them as per their respective instructions: 
    - [SAM3](https://github.com/facebookresearch/sam3#installation)
    - [SAM3D Objects](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) (for CUDA 12.8+ use [these instructions](https://github.com/facebookresearch/sam-3d-objects/issues/15#issuecomment-3560650855))
    - [Depth Pro](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
    - [FoundationPose](https://github.com/NVlabs/FoundationPose#env-setup-option-2-conda-local) (for CUDA 12.8+, build PyTorch3D from source as detailed [here](https://github.com/NVlabs/FoundationPose/issues/398#issuecomment-3808312479))


# Running the pipeline
Running the pipeline is a multi-step process. The intermediate results will be passed between stages as a .pkl file.
1. Perform ORB SLAM camera pose and landmark tracking: `python scripts/track_camera_poses.py --input_video vids/demo.mp4 --output_pkl output/results.pkl`
2. Perform mesh reconstruction: `python scripts/reconstruct_meshes.py --input_video vids/demo.mp4 --classes "book" "kettle" --output_dir output/reconstructed_meshes --input_pkl output/results.pkl`
3. Perform pose estimation: `python scripts/track_object_poses.py --input_video vids/demo.mp4 --classes "book" "kettle" --input_pkl  output/results.pkl`

# Visualizing the results
For a quick visualization of results in Open3D, run: `python scripts/visualize_tracked_video.py --input output/results.pkl`  
For displaying the objects in the Unity app, a different output format is required. To perform the conversion, run:  `python scripts/pkl_to_json.py --input output/results.pkl --output output/results.json`
