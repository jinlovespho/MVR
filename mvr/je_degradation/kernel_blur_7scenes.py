import os
import sys
sys.path.append(os.getcwd())

from motionblur.motionblur import Kernel
import glob
from tqdm import tqdm
from PIL import Image

hq_root_path = f'/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean/7scenes/7Scenes'

scenes = glob.glob(f'{hq_root_path}/*')
scenes = [scene for scene in scenes if 'meshes' not in scene]

BLUR_INTENSITY = 0.1

def resize_long_side(img, target=640):
    w, h = img.size
    if max(w, h) == target:
        return img

    scale = target / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.BICUBIC)


for KERNEL_SIZE in [100, 300, 500]:
    print('Applying kernel: ', KERNEL_SIZE)

    for scene in tqdm(scenes):
        images = glob.glob(f'{scene}/*/*color*')
        deg_scene_save_path = '/'.join(images[0].split('/')[:-1])
        deg_scene_save_path = deg_scene_save_path.replace('clean', f'cam_blur_{KERNEL_SIZE}_resize_640')
        os.makedirs(deg_scene_save_path, exist_ok=True)

        img = Image.open(images[0])
        print('Image size: ', img.size)

        for image in images:

            img_id = image.split('/')[-1].split('.')
            img_id = '.'.join(img_id[:-1])

            img = Image.open(image).convert('RGB')
            orig_size = img.size

            resized_img = resize_long_side(img, 640)

            tmp_path = '/tmp/tmp_resize1.jpg'
            resized_img.save(tmp_path)

            kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
            blurred = kernel.applyTo(tmp_path, keep_image_dim=True)

            # blurred = blurred.resize(orig_size, Image.BICUBIC)

            blurred.save(f'{deg_scene_save_path}/{img_id}.jpg')

print('CAM BLUR FINISH: 7scenes')