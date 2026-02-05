import os 
import csv 
import glob
import h5py
import cv2 
import numpy as np 
import torch 
import random 
from PIL import Image
from torch.utils.data import Dataset 

from depth_anything_3.utils.io.input_processor import InputProcessor

from motionblur.motionblur import Kernel 


class PhoETH3D(Dataset):
    def __init__(self, data_cfg, mode='train'):
        
        self.ds_name = 'eth3d'
        self.data_cfg = data_cfg 
        self.mode = mode 
        
        
        self.data = {}
        
            
        ann_path = data_cfg.ann_path
        lq_root_path = data_cfg.lq_root_path
        hq_root_path = data_cfg.hq_root_path
        gt_depth_root_path = data_cfg.gt_depth_root_path
            
            
        # load data paths 
        lq_paths = sorted(glob.glob(f'{lq_root_path}/image/*/*.png'))[:data_cfg.num_eval_img]
        hq_paths = sorted(glob.glob(f'{hq_root_path}/*/images/dslr_images/*.JPG'))[:data_cfg.num_eval_img]
        depth_paths = sorted(glob.glob(f'{gt_depth_root_path}/*/*/*/*/*.JPG'))[:data_cfg.num_eval_img]
        
        
        # safety check
        assert len(lq_paths) == len(hq_paths) == len(depth_paths)
        assert len(lq_paths) != 0 
        assert len(hq_paths) != 0
        assert len(depth_paths) != 0
            
        
        self.data['hq_img'] = hq_paths 
        self.data['lq_img'] = lq_paths 
        self.data['gt_depth'] = depth_paths 
        

        self.view_sel = data_cfg.view_selection
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
            interpolation = (cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
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
            interpolation = (cv2.INTER_CUBIC if upscale else cv2.INTER_AREA)
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        return img

    def resize_depth(self, depth: np.ndarray):
        process_res = 504
        patch_size = 14

        h, w = depth.shape
        longest = max(h, w)

        if longest != process_res:
            scale = process_res / float(longest)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        h, w = depth.shape

        def nearest_multiple(x, p):
            return int(round(x / p)) * p

        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))

        if new_w != w or new_h != h:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return depth

            
    def load_depth(self, depth_path):
        # eth3d all image sizes
        H, W = 4032, 6048
        depth = np.fromfile(depth_path, dtype=np.float32)
        assert depth.size == H * W, f"Size mismatch: {depth_path}"
        depth = depth.reshape(H, W)
        depth[np.isinf(depth)] = np.nan
        return depth


    def __getitem__(self, items):
        
        idx, num_input_view = items

        # view selection strategy        
        if self.view_sel.strategy == 'near_random':
            frame_ids = self.get_nearby_ids(anchor=idx, num_frames=num_input_view, expand_ratio=self.view_sel.expand_ratio)
        elif self.view_sel_strategy == 'near_random':
            frame_ids = self.get_nearby_ids(anchor=idx, num_frames=num_input_view)
        
        outputs={}
        outputs['frame_ids'] = frame_ids
        
        
        # ----------------------
        #       process hq
        # ----------------------
        hq_view_id=[] 
        hq_view_list=[]
        if 'hq_img' in self.data.keys():
            views = [self.data['hq_img'][i] for i in frame_ids]
            for view in views:
                scene_id = view.split('/')[-4]
                view_id = view.split('/')[-1].split('.')[0]
                hq_view_id.append(f'eth3d_{scene_id}_{view_id}')
                hq_view_list.append(self.resize(self.convert_imgpath(view)))
            outputs['hq_ids'] = hq_view_id
            outputs['hq_views'] = hq_view_list


        # ----------------------
        #       process lq
        # ----------------------
        lq_view_id=[] 
        lq_view_list=[]
        if 'lq_img' in self.data.keys():
            views = [self.data['lq_img'][i] for i in frame_ids]
            for view in views:
                scene_id = view.split('/')[-2]
                view_id = view.split('/')[-1].split('.')[0]
                lq_view_id.append(f'eth3d_{scene_id}_{view_id}')
                lq_view_list.append(self.resize(self.convert_imgpath(view)))
            outputs['lq_ids'] = lq_view_id
            outputs['lq_views'] = lq_view_list



        # -------------------------
        #       process depth
        # -------------------------
        depth_view_id=[]
        depth_view_list=[]
        if 'gt_depth' in self.data.keys():
            views = [self.data['gt_depth'][i] for i in frame_ids]
            for view in views:
                scene_id = view.split('/')[-4]
                view_id = view.split('/')[-1].split('.')[0]
                depth_view_id.append(f'eth3d_{scene_id}_{view_id}')
                depth_view_list.append(self.resize_depth(self.load_depth(view)))
            outputs['gt_depth_ids'] = depth_view_id
            outputs['gt_depths'] = depth_view_list
            

        return outputs
        return {
            "frame_ids": frame_ids,
            
            "hq_ids": hq_view_id,
            "hq_views": hq_view_list,
            
            'hq_latent_ids': hq_latent_view_id,
            'hq_latent_views': hq_latent_view_list,
            
            'lq_ids': lq_view_id,
            'lq_views': lq_view_list,
            
            'gt_depth_ids': depth_view_id,
            'gt_depths': depth_view_list,
        }        

    def __len__(self):
        return len(self.data['hq_img'])
