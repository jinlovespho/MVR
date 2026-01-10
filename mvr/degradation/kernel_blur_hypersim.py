import os 
import sys
sys.path.append(os.getcwd())
from motionblur.motionblur import Kernel 
import glob 
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image 
from tqdm import tqdm 

import h5py
from PIL import Image
import numpy as np 

# KERNEL_SIZE=100
BLUR_INTENSITY=0.1

# for KERNEL_SIZE in [10]:
for KERNEL_SIZE in [50, 30, 10]:

    # load hypersim clean data
    all_data_folders = sorted(glob.glob(f'/mnt/dataset1/MV_Restoration/hypersim/data/*/images/*final_hdf5*'))

    for data in all_data_folders:
        volume = data.split('/')[-3].split('_')[-2]
        scene = data.split('/')[-3].split('_')[-1]
        camera = data.split('/')[-1].split('_')[-3]
        

        # only filter volumes until 001~010
        if volume not in ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']:
            continue

        
        print(data)
        print(f'Processing data: ai_{volume}_{scene}/cam_{camera}')
        
        # hypersim degradation saving directory
        deg_save_dir = f'/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel{KERNEL_SIZE}_intensity01/ai_{volume}_{scene}/scene_cam_{camera}_final_hdf5/images'
        os.makedirs(deg_save_dir, exist_ok=True)

        gamma = 1.0 / 2.2
        inv_gamma = 1.0 / gamma
        percentile = 90
        brightness_nth_percentile_desired = 0.8
        eps = 0.0001
        
        img_paths = sorted(glob.glob(f'{data}/*color*'))
        for image_path in tqdm(img_paths):
            
            img_id = image_path.split('/')[-1].split('.')[-3]
            # print('img id: ', img_id)

            try:
                with h5py.File(image_path, "r") as f:
                    rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]
                
                entity_path = image_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
                if os.path.exists(entity_path):
                    with h5py.File(entity_path, "r") as f:
                        render_entity_id = f["dataset"][:].astype(np.int32)
                    valid_mask = render_entity_id != -1
                else:
                    valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

            brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]
            brightness_valid = brightness[valid_mask]
            
            if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
                scale = 1.0
            else:
                current_p = np.percentile(brightness_valid, percentile)
                scale = np.power(brightness_nth_percentile_desired, inv_gamma) / current_p

            rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
            rgb_tm = np.clip(rgb_tm, 0.0, 1.0)
            rgb_uint8 = (rgb_tm * 255.0).round().astype(np.uint8)
            img_pil = Image.fromarray(rgb_uint8, mode="RGB")
            
            # init kernel
            kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
            blurred = kernel.applyTo(img_pil, keep_image_dim=True)
            # w, h = blurred.size
            # margin = 128
            # img_cropped = blurred.crop((
            #     margin,          # left
            #     margin,          # top
            #     w - margin,      # right
            #     h - margin       # bottom
            # ))
            blurred.save(f'{deg_save_dir}/frame_{img_id}_color.png')
            


