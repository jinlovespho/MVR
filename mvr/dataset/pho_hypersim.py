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


class PhoHypersim(Dataset):
    def __init__(self, data_cfg, mode='train'):
        
        self.ds_name = 'hypersim'
        self.data_cfg = data_cfg 
        self.mode = mode 
        
        self.data = {}
        
        
        # annotation
        with open(data_cfg.ann_path) as f:
            anns = list(csv.DictReader(f))  # len: 82,900
        train_anns = [ann for ann in anns if ann['split_partition_name']=='train']      # 59,543
        val_anns = [ann for ann in anns if ann['split_partition_name']=='val']          # 7,386
        test_anns = [ann for ann in anns if ann['split_partition_name']=='test']        # 7,690
        
        
        # HQ 
        train_hq_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_final_hdf5/frame.{int(ann['frame_id']):04}.color.hdf5" for ann in train_anns])
        val_hq_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_final_hdf5/frame.{int(ann['frame_id']):04}.color.hdf5" for ann in val_anns])
        test_hq_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_final_hdf5/frame.{int(ann['frame_id']):04}.color.hdf5" for ann in test_anns])


        # depth 
        train_depth_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_geometry_hdf5/frame.{int(ann['frame_id']):04}.depth_meters.hdf5" for ann in train_anns])
        val_depth_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_geometry_hdf5/frame.{int(ann['frame_id']):04}.depth_meters.hdf5" for ann in val_anns])
        test_depth_paths = sorted([f"{data_cfg.hq_root_path}/{ann['scene_name']}/images/scene_{ann['camera_name']}_geometry_hdf5/frame.{int(ann['frame_id']):04}.depth_meters.hdf5" for ann in test_anns])
        

        if mode == 'train':
            self.data['hq_img'] = sorted([path for path in train_hq_paths if os.path.exists(path)])
            self.data['gt_depth'] = sorted([path for path in train_depth_paths if os.path.exists(path)])
        elif mode == 'val':
            # self.data['hq_img'] = sorted([path for path in val_hq_paths if os.path.exists(path)])[:data_cfg.num_eval_img]
            # self.data['gt_depth'] = sorted([path for path in val_depth_paths if os.path.exists(path)])[:data_cfg.num_eval_img]

            hq_imgs = sorted([p for p in val_hq_paths if os.path.exists(p)])
            gt_depths = sorted([p for p in val_depth_paths if os.path.exists(p)])
            assert len(hq_imgs) == len(gt_depths), "HQ and GT depth count mismatch"
            idx = list(range(len(hq_imgs)))
            random.seed(97)
            random.shuffle(idx)
            idx = idx[:data_cfg.num_eval_img]
            self.data['hq_img'] = [hq_imgs[i] for i in idx]
            self.data['gt_depth'] = [gt_depths[i] for i in idx]
            
            
            # fix validation lq images
            self.lq_ids = []
            self.lq_imgs = []
            for hq_view in self.data['hq_img']:
                volume = hq_view.split('/')[-4].split('_')[-2]
                scene = hq_view.split('/')[-4].split('_')[-1]
                camera = hq_view.split('/')[-2].split('_')[-3]
                view_id = hq_view.split('/')[-1].split('.')[-3]
                self.lq_ids.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
                img_pil = self.convert_hdf5_img(hq_view)
                KERNEL_SIZE=50
                BLUR_INTENSITY=0.1
                kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
                blurred = kernel.applyTo(img_pil, keep_image_dim=True)
                blurred = np.array(blurred)
                self.lq_imgs.append(self.resize(blurred))
            
        elif mode == 'test':
            self.data['hq_img'] = sorted([path for path in test_hq_paths if os.path.exists(path)])
            self.data['gt_depth'] = sorted([path for path in test_depth_paths if os.path.exists(path)])

        
        # keep_volumes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
        # if data_cfg.hq_root_path is not None:
        #     self.data['hq_img'] = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/images/*final_hdf5*/*color*'))
        #     # filter only hypersim volume 001~010
        #     self.data['hq_img'] = sorted(data for data in self.data['hq_img'] if data.split('/')[-4].split('_')[-2] in keep_volumes)

        # # if data_cfg.hq_latent_root_path is not None:
        # #     self.data['hq_latent'] = sorted(glob.glob(f'{data_cfg.hq_latent_root_path}/*/*/*'))
        # #     # filter only hypersim volume 001~010
        
            
        # if data_cfg.lq_root_path is not None:
        #     self.data['lq_img'] = sorted(glob.glob(f'{data_cfg.lq_root_path}/*/*/images/*'))
        #     # filter only hypersim volume 001~010


        # if data_cfg.depth_path is not None:
        #     keep_volumes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
        #     self.data['gt_depth'] = sorted(glob.glob(f"{data_cfg.depth_path}/*/images/*geometry_hdf5*/*depth_meters*"))
        #     # filter only hypersim volume 001~010
        #     self.data['gt_depth'] = sorted(data for data in self.data['gt_depth'] if data.split('/')[-4].split('_')[-2] in keep_volumes)

          
        self.view_sel_strategy = data_cfg.view_sel_strategy 
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


    def convert_hdf5_img(self, hdf5_path):
        gamma = 1.0 / 2.2
        inv_gamma = 1.0 / gamma
        percentile = 90
        brightness_nth_percentile_desired = 0.8
        eps = 0.0001
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
        if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
            scale = 1.0
        else:
            current_p = np.percentile(brightness_valid, percentile)
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / current_p
        rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
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
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = depth.shape
        def nearest_multiple(x, p):
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down
        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))
        if new_w != w or new_h != h:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return depth

    def convert_hdf5_depth(self, hdf5_path):
        W, H, FOCAL = 1024, 768, 886.81
        with h5py.File(hdf5_path, "r") as f:
            dist = f["dataset"][:]
        x = np.linspace(-W / 2 + 0.5, W / 2 - 0.5, W)
        y = np.linspace(-H / 2 + 0.5, H / 2 - 0.5, H)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, FOCAL)
        plane_norm = np.sqrt(X**2 + Y**2 + Z**2)
        gt_depth = dist / plane_norm * FOCAL
        gt_depth = np.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
        return gt_depth.astype(np.float32)
        


    def __getitem__(self, items):
        idx, num_input_view = items
        frame_ids = self.get_nearby_ids(anchor=idx, num_frames=num_input_view)
        
        outputs={}
        outputs['frame_ids'] = frame_ids
        
        
        # ----------------------
        #       process hq
        # ----------------------
        hq_view_id=[] 
        hq_view_list=[]
        if 'hq_img' in self.data.keys():
            hq_views = [self.data['hq_img'][i] for i in frame_ids]
            for hq_view in hq_views:
                volume = hq_view.split('/')[-4].split('_')[-2]
                scene = hq_view.split('/')[-4].split('_')[-1]
                camera = hq_view.split('/')[-2].split('_')[-3]
                view_id = hq_view.split('/')[-1].split('.')[-3]
                hq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
                hq_view_list.append(self.resize(self.convert_hdf5_img(hq_view)))
                # hq_view_list.append({f'hypersim_{volume}_{scene}_{camera}_{view_id}': self.convert_hdf5(hq_view)})
            outputs['hq_ids'] = hq_view_id
            outputs['hq_views'] = hq_view_list


        
        # # ----------------------------------
        # #       process hq latent 
        # # ----------------------------------
        # hq_latent_view_id=[]
        # hq_latent_view_list=[]
        # if 'hq_latent' in self.data.keys():
        #     hq_latent_views = [self.data['hq_latent'][i] for i in frame_ids]
        #     for hq_latent_view in hq_latent_views:
        #         volume = hq_latent_view.split('/')[-3].split('_')[-2]
        #         scene = hq_latent_view.split('/')[-3].split('_')[-1]
        #         camera = hq_latent_view.split('/')[-2].split('_')[-1]
        #         view_id = hq_latent_view.split('/')[-1].split('.')[-2]
        #         hq_latent_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
        #         hq_latent_view_list.append(torch.load(hq_latent_view))  # torch.Size([972, 3072])
        #         outputs['hq_latent_ids'] = hq_latent_view_id
        #         outputs['hq_latent_views'] = hq_latent_view_list
            
            
            
            
        # # ----------------------
        # #       process lq
        # # ----------------------
        # lq_view_id=[]
        # lq_view_list=[]
        # if 'lq_img' in self.data.keys():
        #     lq_views = [self.data['lq_img'][i] for i in frame_ids]
        #     for lq_view in lq_views:
        #         volume = lq_view.split('/')[-4].split('_')[-2]
        #         scene = lq_view.split('/')[-4].split('_')[-1]
        #         camera = lq_view.split('/')[-3].split('_')[-3]
        #         view_id = lq_view.split('/')[-1].split('.')[-2].split('_')[-2]
        #         lq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
        #         lq_view_list.append(self.resize(self.convert_imgpath(lq_view)))
        #         outputs['lq_ids'] = lq_view_id 
        #         outputs['lq_views'] = lq_view_list
                




        # ----------------------------------
        #       get lq on the fly
        # ----------------------------------
        lq_view_id=[]
        lq_view_list=[]
        
        if self.mode == 'train':
            hq_views = [self.data['hq_img'][i] for i in frame_ids]
            for hq_view in hq_views:
                volume = hq_view.split('/')[-4].split('_')[-2]
                scene = hq_view.split('/')[-4].split('_')[-1]
                camera = hq_view.split('/')[-2].split('_')[-3]
                view_id = hq_view.split('/')[-1].split('.')[-3]
                lq_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
                img_pil = self.convert_hdf5_img(hq_view)
                
                
                KERNEL_SIZE=50
                BLUR_INTENSITY=0.1
                kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
                blurred = kernel.applyTo(img_pil, keep_image_dim=True)
                blurred = np.array(blurred)
                lq_view_list.append(self.resize(blurred))
            outputs['lq_ids'] = lq_view_id 
            outputs['lq_views'] = lq_view_list

        elif self.mode == 'val':
            outputs['lq_ids'] = [self.lq_ids[i] for i in frame_ids]
            outputs['lq_views'] = [self.lq_imgs[i] for i in frame_ids]

        
        # -------------------------
        #       process depth
        # -------------------------
        depth_view_id=[]
        depth_view_list=[]
        if 'gt_depth' in self.data.keys():
            depth_views = [self.data['gt_depth'][i] for i in frame_ids]
            for depth_view in depth_views:
                volume = depth_view.split('/')[-4].split('_')[-2]
                scene = depth_view.split('/')[-4].split('_')[-1]
                camera = depth_view.split('/')[-2].split('_')[-3]
                view_id = depth_view.split('/')[-1].split('.')[-3]
                depth_view_id.append(f'hypersim_{volume}_{scene}_{camera}_{view_id}')
                depth_view_list.append(self.resize_depth(self.convert_hdf5_depth(depth_view)))
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
