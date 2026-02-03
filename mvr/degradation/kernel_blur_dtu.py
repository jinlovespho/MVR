import os 
import sys
sys.path.append(os.getcwd())
from motionblur.motionblur import Kernel 
import glob 
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image 
from tqdm import tqdm


hq_root_path = f'/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean/dtu'
scenes = glob.glob(f'{hq_root_path}/Rectified/*')


BLUR_INTENSITY=0.1
for KERNEL_SIZE in [50, 100, 300, 500]:

    print('Applying kernel: ', KERNEL_SIZE)
    for scene in tqdm(scenes):
        
        images = glob.glob(f'{scene}/*')
        
        deg_scene_save_path = '/'.join(images[0].split('/')[:-1])
        deg_scene_save_path = deg_scene_save_path.replace('clean', f'cam_blur_{KERNEL_SIZE}')
        os.makedirs(deg_scene_save_path, exist_ok=True)
        
        for image in images:
            
            img_id = image.split('/')[-1].split('.')[0]
            
            # init kernel
            kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
            blurred = kernel.applyTo(image, keep_image_dim=True)
            blurred.save(f'{deg_scene_save_path}/{img_id}.jpg')
            

print('CAM BLUR FINISH: dtu')
        