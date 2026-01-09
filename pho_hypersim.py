import os 
import glob
import numpy as np 
from torch.utils.data import Dataset 



class PhoHypersim(Dataset):
    def __init__(self, data_cfg):
        self.hq_root_path = data_cfg.hq_root_path 
        self.hq_imgs = sorted(glob.glob(f'{self.hq_root_path}/*/images/*final_hdf5*/*color*'))
        

    # ---- Meta-style temporal expansion ----
    def get_nearby_ids(
        self,
        anchor,
        num_frames,
        expand_ratio=2.0,
    ):
        expand_range = int(num_frames * expand_ratio)

        low = max(0, anchor - expand_range)
        high = min(len(self.hq_imgs), anchor + expand_range + 1)

        candidates = np.arange(low, high)

        sampled = np.random.choice(
            candidates,
            size=num_frames - 1,
            replace=True,
        )

        return np.concatenate([[anchor], sampled])


    # ---- dataset entry point ----
    def __getitem__(self, items):
        
        idx, image_num, aspect_ratio = items
        print('hypersim: ', idx, image_num, aspect_ratio)

        # breakpoint()
        frame_ids = self.get_nearby_ids(anchor=idx, num_frames=image_num)

        imgs = [self.hq_imgs[i] for i in frame_ids]

        return {
            "frame_ids": frame_ids,
            "images": imgs,
            "aspect_ratio": aspect_ratio,
        }        


    def __len__(self):
        return len(self.hq_imgs)

