import os 
import glob
import h5py
import cv2 
import numpy as np 
from torch.utils.data import Dataset 

from depth_anything_3.utils.io.input_processor import InputProcessor



class PhoHypersim(Dataset):
    def __init__(self, data_cfg):
        
        self.data = {}
        
        if data_cfg.hq_root_path is not None:
            self.data['hq_img'] = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/images/*final_hdf5*/*color*'))
            # filter only hypersim volume 001~010

        if data_cfg.hq_latent_root_path is not None:
            self.data['hq_latent'] = sorted(glob.glob(f'{data_cfg.hq_latent_root_path}/*/*/*'))
            # filter only hypersim volume 001~010
            
        if data_cfg.lq_root_path is not None:
            self.data['lq_img'] = sorted(glob.glob(f'{data_cfg.lq_root_path}/*/*/images/*.png'))
            # filter only hypersim volume 001~010
            

        self.view_sel_strategy = data_cfg.view_sel_strategy 
        
        
        # hypersim hdf5 preprocess values 
        self.gamma = 1.0 / 2.2
        self.inv_gamma = 1.0 / self.gamma
        self.percentile = 90
        self.brightness_nth_percentile_desired = 0.8
        self.eps = 0.0001
        
        self.input_processor = InputProcessor()
        
    
    def get_nearby_ids(
        self,
        anchor,
        num_frames,
        expand_ratio=2.0,
    ):
        expand_range = int(num_frames * expand_ratio)
        low = max(0, anchor - expand_range)
        high = min(len(self.data['hq_img']), anchor + expand_range + 1)
        candidates = np.arange(low, high)
        sampled = np.random.choice(
            candidates,
            size=num_frames - 1,
            replace = (len(candidates) < num_frames - 1),
        )
        return np.concatenate([[anchor], sampled])
    

    def convert_imgpath(self, img_path: str) -> np.ndarray:
        """
        Load a TartanAir PNG image and return RGB uint8 numpy array.

        Args:
            img_path: path to *.png image

        Returns:
            img: np.ndarray [H, W, 3], uint8, RGB
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # cv2 loads as BGR uint8
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img_rgb


    def convert_hdf5(self, hdf5_path):
        try:
            with h5py.File(hdf5_path, "r") as f:
                rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]
            entity_path = hdf5_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
            if os.path.exists(entity_path):
                with h5py.File(entity_path, "r") as f:
                    render_entity_id = f["dataset"][:].astype(np.int32)
                valid_mask = render_entity_id != -1
            else:
                valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)
        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            return None
        brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]
        brightness_valid = brightness[valid_mask]
        if len(brightness_valid) == 0 or np.percentile(brightness_valid, self.percentile) < self.eps:
            scale = 1.0
        else:
            current_p = np.percentile(brightness_valid, self.percentile)
            scale = np.power(self.brightness_nth_percentile_desired, self.inv_gamma) / current_p
        rgb_tm = np.power(np.maximum(scale * rgb_color, 0), self.gamma)
        rgb_tm = np.clip(rgb_tm, 0.0, 1.0)
        rgb_tm = (rgb_tm * 255.0).round().astype(np.uint8)
        return rgb_tm
    
    
    def resize(self, img: np.ndarray):
        """
        Resize image using:
        1) longest-side resize to process_res
        2) make divisible by patch_size via small resize

        Args:
            img: np.ndarray [H, W, 3], uint8

        Returns:
            resized_img: np.ndarray [H', W', 3], uint8
        """
        process_res = 504
        patch_size = 14

        # -------------------------
        # 1. resize longest side
        # -------------------------
        h, w = img.shape[:2]
        longest = max(h, w)

        if longest != process_res:
            scale = process_res / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            interpolation = (
                cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            )

            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # -------------------------
        # 2. make divisible by patch_size (resize version)
        # -------------------------
        h, w = img.shape[:2]

        def nearest_multiple(x, p):
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))

        if new_w != w or new_h != h:
            upscale = (new_w > w) or (new_h > h)
            interpolation = (
                cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
            )
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        return img
            
        
        
        
    def __getitem__(self, items):
        idx, num_input_view = items
        # print('hypersim: ', idx, num_input_view)
        frame_ids = self.get_nearby_ids(anchor=idx, num_frames=num_input_view)
        
        hq_views = [self.data['hq_img'][i] for i in frame_ids]
        lq_views = [self.data['lq_img'][i] for i in frame_ids]
        
        hq_view_id=[] 
        hq_view_list=[]
        for hq_view in hq_views:
            volume = hq_view.split('/')[-4].split('_')[-2]
            scene = hq_view.split('/')[-4].split('_')[-1]
            camera = hq_view.split('/')[-2].split('_')[-3]
            view_id = hq_view.split('/')[-1].split('.')[-3]
            hq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
            hq_view_list.append(self.resize(self.convert_hdf5(hq_view)))
            # hq_view_list.append({f'hypersim_{volume}_{scene}_{camera}_{view_id}': self.convert_hdf5(hq_view)})
            # /mnt/dataset1/MV_Restoration/hypersim/deg_data/kernel_50/ai_001_001/scene_cam_00_final_hdf5/frame_0000.png
            
        lq_view_id=[]
        lq_view_list=[]
        for lq_view in lq_views:
            volume = lq_view.split('/')[-4].split('_')[-2]
            scene = lq_view.split('/')[-4].split('_')[-1]
            camera = lq_view.split('/')[-3].split('_')[-3]
            view_id = lq_view.split('/')[-1].split('.')[-2].split('_')[-2]
            lq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
            lq_view_list.append(self.resize(self.convert_imgpath(lq_view)))
            
        return {
            "frame_ids": frame_ids,
            
            "hq_ids": hq_view_id,
            "hq_views": hq_view_list,
            
            'lq_ids': lq_view_id,
            'lq_views': lq_view_list,
        }        

    def __len__(self):
        return len(self.data['hq_img'])
