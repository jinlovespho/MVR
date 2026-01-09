import os
import glob
import h5py
import torch 
import numpy as np
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3


# load model 
# model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1").to("cuda")

# load data 
hq_root_path = "/mnt/dataset1/MV_Restoration/hypersim/data"
hq_imgs = sorted(glob.glob(f'{hq_root_path}/*/images/*final_hdf5*/*color*'))

# set save path 
save_root_dir = f'/mnt/dataset1/MV_Restoration/hypersim/da3_clean_latent/singleview'
os.makedirs(save_root_dir, exist_ok=True)

# hypersim preprocess values
gamma = 1.0 / 2.2
inv_gamma = 1.0 / gamma
percentile = 90
brightness_nth_percentile_desired = 0.8
eps = 0.0001

# forward pass single image for da3 inference
for img_path in hq_imgs:
    volume = img_path.split('/')[-4].split('_')[-2]
    scene = img_path.split('/')[-4].split('_')[-1]
    camera = img_path.split('/')[-2].split('_')[-3]
    img_id = img_path.split('/')[-1].split('.')[-3]
    
    # only filter volumes until 001~010
    if volume not in ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']:
        continue
    
    print(f'Processing data: ai_{volume}_{scene}/cam_{camera} img-{img_id}')
    
    try:
        with h5py.File(img_path, "r") as f:
            rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]
        
        entity_path = img_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
        if os.path.exists(entity_path):
            with h5py.File(entity_path, "r") as f:
                render_entity_id = f["dataset"][:].astype(np.int32)
            valid_mask = render_entity_id != -1
        else:
            valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
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
    rgb_tm = (rgb_tm * 255.0).round().astype(np.uint8)


    # inference
    with torch.no_grad():
        
        # vggt inference 
        if cfg.model.val.mv_3dff.model == 'vggt':
            with torch.cuda.amp.autocast(dtype=dtype):
                imgs = load_and_preprocess_images(imgs_path).to(device)
                pred_vggt = model(imgs)
                pred_depth = pred_vggt['depth']  # [F, 350, 518, 1]
                pred_depth = pred_depth.squeeze(dim=(0,-1))
                # pred_depth = pred_depth.detach().cpu().numpy().astype(np.float32)
                
        # da3 inference 
        elif cfg.model.val.mv_3dff.model == 'da3':
            # pred_da3 = model.inference(imgs_path)
            pred_da3 = model.inference([rgb_tm], export_feat_layers=[19, 27, 33, 39])
            pred_depth = torch.from_numpy(pred_da3.depth).to(device)                 # (F, 336 504)
            save_image(pred_depth, './tmp2.jpg', normalize=True)

    print(img)
    print(volume, scene, camera, img_id)
    breakpoint()
    
    
    
    



prediction = model.inference(imgs)
assert len(imgs) == len(prediction.depth)

for img_path, depth in zip(imgs, prediction.depth):
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    # depth: torch.Tensor or numpy → force numpy float32
    if hasattr(depth, 'detach'):
        depth = depth.detach().cpu().numpy()

    depth = depth.astype(np.float32)

    np.save(f'{save_root_dir}/{img_id}.npy', depth)
    