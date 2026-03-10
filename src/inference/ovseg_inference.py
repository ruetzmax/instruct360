import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.append("ov-seg")
from open_vocab_seg.utils import VisualizationDemo
from open_vocab_seg import add_ovseg_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from inference_utils import base64_to_image, image_to_base64, load_inference_input, save_inference_output


input_data = load_inference_input()

# load model
print("Loading OV-Seg model...")
ov_seg_cfg = get_cfg()
add_deeplab_config(ov_seg_cfg)
add_ovseg_config(ov_seg_cfg)
ov_seg_cfg.merge_from_file("ov-seg/configs/ovseg_swinB_vitL_demo.yaml")
ov_seg_cfg.MODEL.WEIGHTS = "ov-seg/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth"
ov_seg_cfg.DATALOADER.NUM_WORKERS = 0
if not torch.cuda.is_available():
    ov_seg_cfg.MODEL.DEVICE = "cpu"

ov_seg_model = VisualizationDemo(ov_seg_cfg)
print("OV-Seg model loaded.")

# process chunks
chunk_images_b64 = input_data["chunk_images_base64"]
prompt = input_data["prompt"]

all_masks_b64 = []

for idx, chunk_image_b64 in enumerate(chunk_images_b64):
    chunk_image = base64_to_image(chunk_image_b64)
    
    predictions, _ = ov_seg_model.run_on_image(chunk_image, [prompt])
    
    if "sem_seg" not in predictions:
        # return empty mask
        rgba_image = np.zeros((chunk_image.shape[0], chunk_image.shape[1], 4), dtype=np.uint8)
        all_masks_b64.append(image_to_base64(rgba_image))
        continue
    
    # construct and return masked image
    sem_seg = predictions["sem_seg"]
    blank_area = (sem_seg[0] == 0)
    
    masked_image = chunk_image.copy()
    rgba_image = np.zeros((masked_image.shape[0], masked_image.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = masked_image
    rgba_image[:, :, 3] = 255 
    rgba_image[blank_area, 3] = 0
    
    mask_b64 = image_to_base64(rgba_image)
    all_masks_b64.append(mask_b64)
    print(f"Processed chunk {idx+1}/{len(chunk_images_b64)}")

output_data = {
    "masks_base64": all_masks_b64
}

save_inference_output(output_data)
print(f"Saved {len(all_masks_b64)} masks")
