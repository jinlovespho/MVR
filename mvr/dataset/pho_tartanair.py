import os 
import glob
import cv2 
import numpy as np
from torch.utils.data import Dataset 



class PhoTartanAir(Dataset):
    def __init__(self, data_cfg):
        
        self.data = {}
        
        if data_cfg.hq_root_path is not None:
            self.data['hq_img'] = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/Easy/*/image_left/*.png'))
        
        if data_cfg.hq_latent_root_path is not None:
            self.data['hq_latent'] = sorted(glob.glob(f'{data_cfg.hq_root_path }/*/images/*final_hdf5*/*color*'))
            
        if data_cfg.lq_root_path is not None:
            self.data['lq_img'] = sorted(glob.glob(f'{data_cfg.hq_root_path }/*/images/*final_hdf5*/*color*'))
        
        if data_cfg.lq_latent_root_path is not None:
            self.data['lq_latent'] = sorted(glob.glob(f'{data_cfg.hq_root_path }/*/images/*final_hdf5*/*color*'))
        
        

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
        # print('tartanair: ', idx, num_input_view)
        frame_ids = self.get_nearby_ids(anchor=idx, num_frames=num_input_view,)
        hq_views = [self.data['hq_img'][i] for i in frame_ids]
        
        hq_view_id=[]
        hq_view_list=[]
        for hq_view in hq_views:
            scene = hq_view.split('/')[-5]
            subset = hq_view.split('/')[-3]
            view_id = hq_view.split('/')[-1].split('.')[0]
            
            hq_view_id.append(f'tartanair_{scene}_{subset}_{view_id}')
            hq_view_list.append(self.resize(self.convert_imgpath(hq_view)))
            
            # hq_view_list.append({f'tartanair_{scene}_{subset}_{view_id}': hq_view})
            
        return {
            "frame_ids": frame_ids,
            "hq_ids": hq_view_id,
            "hq_views": hq_view_list
        }        


    def __len__(self):
        return len(self.data['hq_img'])

