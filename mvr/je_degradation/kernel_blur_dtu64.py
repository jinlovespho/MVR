import os 
import sys
sys.path.append(os.getcwd())

from motionblur.motionblur import Kernel 
import glob 
from tqdm import tqdm
from PIL import Image

hq_root_path = f'/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean/dtu64'
scenes = glob.glob(f'{hq_root_path}/*scan*')

BLUR_INTENSITY = 0.1

def resize_long_side(img, target=640):
    w, h = img.size
    scale = target / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.BICUBIC)

for KERNEL_SIZE in [100, 300, 500]:

    print('Applying kernel: ', KERNEL_SIZE)

    kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)

    for scene in tqdm(scenes):
        
        images = glob.glob(f'{scene}/image/*')
        
        deg_scene_save_path = '/'.join(images[0].split('/')[:-1])
        deg_scene_save_path = deg_scene_save_path.replace('clean', f'cam_blur_{KERNEL_SIZE}_resize_640')
        os.makedirs(deg_scene_save_path, exist_ok=True)
        
        for image in images:
            
            img_id = image.split('/')[-1].split('.')
            img_id = '.'.join(img_id[:-1])

            img = Image.open(image).convert('RGB')
            orig_size = img.size

            resized_img = resize_long_side(img, 640)

            tmp_path = "/tmp/tmp_resize3.jpg"
            resized_img.save(tmp_path)

            blurred = kernel.applyTo(tmp_path, keep_image_dim=True)

            # blurred = blurred.resize(orig_size, Image.BICUBIC)

            blurred.save(f'{deg_scene_save_path}/{img_id}.jpg')

print('CAM BLUR FINISH: dtu64')