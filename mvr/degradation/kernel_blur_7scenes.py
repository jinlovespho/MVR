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
apply_same_blur = True 


BLUR_INTENSITY=0.1
for KERNEL_SIZE in [150]:

    print('Applying kernel: ', KERNEL_SIZE)
    for scene in tqdm(scenes):
        
        images = glob.glob(f'{scene}/*/*color*')
        
        deg_scene_save_path = '/'.join(images[0].split('/')[:-1])
        if apply_same_blur:
            deg_scene_save_path = deg_scene_save_path.replace('clean', f'cam_blur_{KERNEL_SIZE}_same_rebuttal')
        else:
            deg_scene_save_path = deg_scene_save_path.replace('clean', f'cam_blur_{KERNEL_SIZE}')
        
        os.makedirs(deg_scene_save_path, exist_ok=True)

        ## open and check image size
        img = Image.open(images[0])
        print('Image size: ', img.size)
                
        if apply_same_blur:
            kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
            for image in images:
                img_id = image.split('/')[-1].split('.')
                img_id = '.'.join(img_id[:-1])
                # init kernel
                blurred = kernel.applyTo(image, keep_image_dim=True)
                blurred.save(f'{deg_scene_save_path}/{img_id}.jpg')
                print(f'Saved blurred image: {deg_scene_save_path}/{img_id}.jpg')

        else:        
            for image in images:
                img_id = image.split('/')[-1].split('.')
                img_id = '.'.join(img_id[:-1])
                # init kernel
                kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
                blurred = kernel.applyTo(image, keep_image_dim=True)
                blurred.save(f'{deg_scene_save_path}/{img_id}.jpg')
                print(f'Saved blurred image: {deg_scene_save_path}/{img_id}.jpg')
    

print('CAM BLUR FINISH: 7scenes')
        