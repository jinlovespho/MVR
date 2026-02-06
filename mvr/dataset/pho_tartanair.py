import os 
import glob
import cv2 
import numpy as np
from torch.utils.data import Dataset 
from depth_anything_3.utils.io.input_processor import InputProcessor
from motionblur.motionblur import Kernel 




class PhoTartanAir(Dataset):
    def __init__(self, data_cfg, mode='train'):
        
        self.ds_name = 'tartanair'
        self.data_cfg = data_cfg 
        self.mode = mode 
        
        self.data = {}
        
        # load data paths 
        hq_paths = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/Easy/*/image_left/*.png'))
        depth_paths = sorted(glob.glob(f'{data_cfg.hq_root_path}/*/Easy/*/depth_left/*.npy'))
        
        
        # safety check 
        assert len(hq_paths) == len(depth_paths)
        assert len(hq_paths) != 0
        assert len(depth_paths) != 0
        
        
        self.data['hq_img'] = hq_paths 
        self.data['gt_depth'] = depth_paths 
        

        self.view_sel = data_cfg.view_selection
        self.input_processor = InputProcessor()



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
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = depth.shape

        # def nearest_multiple(x, p):
        #     return int(round(x / p)) * p
        def nearest_multiple(x, p):
            return (x // p) * p

        new_w = max(1, nearest_multiple(w, patch_size))
        new_h = max(1, nearest_multiple(h, patch_size))

        if new_w != w or new_h != h:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return depth

            
    def load_depth(self, depth_path):
        try:
            depth = np.load(depth_path)
            # depth[np.isinf(depth)] = 0
            # depth[depth > 1000] = 0
            return depth
        except Exception as e:
            print(f"Failed to load {depth_path}: {e}")
            return None
        


    def depth2vis(self, depth, maxthresh = 50):
        depthvis = np.clip(depth,0,maxthresh)
        depthvis = depthvis/maxthresh*255
        depthvis = depthvis.astype(np.uint8)
        depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))
        return depthvis


    def get_random_ids(self, anchor, num_frames):
        """
        Global random sampling baseline (TartanAir).
        """
        if num_frames == 1:
            return np.array([anchor], dtype=np.int64)
        N = len(self.data['hq_img'])
        candidates = np.arange(N)
        candidates = np.delete(candidates, anchor)
        K = num_frames - 1
        sampled = np.random.choice(candidates, size=K, replace=(len(candidates) < K),)
        return np.concatenate([[anchor], sampled]).astype(np.int64)
    
    
    def get_nearby_ids_random(
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



    def __getitem__(self, items):
        
        idx, num_input_view = items
        # print(f'tartanair - {num_input_view}')


        # view selection strategy        
        if self.view_sel.strategy == 'random':
            frame_ids = self.get_random_ids(anchor=idx, num_frames=num_input_view)
        elif self.view_sel.strategy == 'near_random':
            frame_ids = self.get_nearby_ids_random(anchor=idx, num_frames=num_input_view, expand_ratio=self.view_sel.expand_ratio)
        
        
        outputs={}
        outputs['frame_ids'] = frame_ids
        
        
        # ----------------------
        #       process hq
        # ----------------------
        hq_view_id=[] 
        hq_view_list=[]
        if 'hq_img' in self.data.keys():
            views = sorted([self.data['hq_img'][i] for i in frame_ids])
            for view in views:
                scene_id = view.split('/')[-5]
                view_id = view.split('/')[-1].split('.')[0]
                hq_view_id.append(f'tartanair_{scene_id}_{view_id}')
                hq_view_list.append(self.resize(self.convert_imgpath(view)))
            outputs['hq_ids'] = hq_view_id
            outputs['hq_views'] = hq_view_list





        # ----------------------------------
        #       get lq on the fly
        # ----------------------------------
        lq_view_id=[]
        lq_view_list=[]
        
        if self.mode == 'train':
            views = sorted([self.data['hq_img'][i] for i in frame_ids])
            for view in views:
                scene_id = view.split('/')[-5]
                view_id = view.split('/')[-1].split('.')[0]
                lq_view_id.append(f'tartanair_{scene_id}_{view_id}')
                img_pil = self.convert_imgpath(view)
                KERNEL_SIZE = self.data_cfg.lq_kernel_size
                BLUR_INTENSITY=0.1
                kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
                blurred = kernel.applyTo(img_pil, keep_image_dim=True)
                blurred = np.array(blurred)
                lq_view_list.append(self.resize(blurred))
            outputs['lq_ids'] = lq_view_id 
            outputs['lq_views'] = lq_view_list



        # -------------------------
        #       process depth
        # -------------------------
        depth_view_id=[]
        depth_view_list=[]
        # depth_vis_view_list=[]
        if 'gt_depth' in self.data.keys():
            views = sorted([self.data['gt_depth'][i] for i in frame_ids])
            for view in views:
                depth_data = self.load_depth(view)
                if depth_data is None:
                    depth_data = np.zeros((480, 640), dtype=np.float32)  # placeholder
                scene_id = view.split('/')[-5]
                view_id = view.split('/')[-1].split('.')[0]
                depth_view_id.append(f'tartanair_{scene_id}_{view_id}')
                depth_view_list.append(self.resize_depth(depth_data))

            
            

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
