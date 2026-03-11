import os 
import sys
sys.path.append(os.getcwd())
import glob 
from PIL import Image 
from tqdm import tqdm

import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image 


deg_save_dir = f'/mnt/dataset1/multi_view_restoration/ECCV26_RESULTS/eth3d/deg_bicubic'
os.makedirs(deg_save_dir, exist_ok=True)

scenes = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker','meadow', 'office', 'pipes', 'playground', 'relief','relief_2', 'terrace', 'terrains']
for scene in tqdm(scenes, desc='Scenes'):
    print(f'Applying bicubic, Processing scene: {scene}')
    os.makedirs(f'{deg_save_dir}/bicubic_img/{scene}', exist_ok=True)

    rgb_paths = sorted(glob.glob(f'/mnt/dataset1/multi_view_restoration/eth3d/eth3d_jisang_train/rgb/{scene}/images/dslr_images/*'))
    
    for rgb_path in rgb_paths:
        
        img_id = rgb_path.split('/')[-1].split('.')[0]

        img = Image.open(rgb_path).convert("RGB")
        img_t = to_tensor(img)
        img_t = img_t.unsqueeze(0)
        _, _, H, W = img_t.shape

        lq = F.interpolate(img_t, size=(H//4, W//4), mode='bicubic', align_corners=False)
        lq = F.interpolate(lq, size=(H, W), mode='bicubic', align_corners=False)
        
        save_image(lq, f'{deg_save_dir}/bicubic_img/{scene}/{img_id}.png')
        