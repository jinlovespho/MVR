import torch 
import torchvision.transforms as T 
from torch.utils.data import Dataset 
import bisect 


class PhoConcatDataset(Dataset):
    """
    Meta-style composed dataset.
    Routes (idx, image_num, aspect_ratio) to the correct sub-dataset.
    """

    def __init__(self, datasets, cfg):

        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes()
        
        self.num_input_view = cfg.data.train.num_input_view 

    def _compute_cumulative_sizes(self):
        sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _get_dataset_index(self, global_idx):
        """
        Convert global idx -> (dataset_id, local_idx)
        """
        dataset_id = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev_size = 0 if dataset_id == 0 else self.cumulative_sizes[dataset_id - 1]
        local_idx = global_idx - prev_size
        return dataset_id, local_idx


    def __getitem__(self, items):
        """
        items = (global_idx, image_num, aspect_ratio)
        """
        
        # global_idx, image_num, aspect_ratio = items
        global_idx = items 
        # print('concat dataset (global): ', global_idx, image_num, aspect_ratio)

        dataset_id, local_idx = self._get_dataset_index(global_idx)
        # print('concat dataset (local): ', dataset_id, local_idx)

        # breakpoint()
        # return self.datasets[dataset_id][(local_idx, image_num, aspect_ratio)]
        return self.datasets[dataset_id][(local_idx, self.num_input_view)]


IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def to_tensor(img):
    """
    img: np.ndarray (H,W,3) uint8 RGB
    returns: torch.FloatTensor (3,H,W) ImageNet-normalized
    """
    x = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .contiguous()
        .float() / 255.0
    )
    return x
    # return IMAGENET_NORMALIZE(x)
    

# def multiview_collate_fn(batch):
#     # batch: List[dict], length = B

#     # ---- frame ids ----
#     frame_ids = torch.stack(
#         [torch.as_tensor(b["frame_ids"]) for b in batch],
#         dim=0
#     )  # [B, V]

#     # ---- ids (keep as Python) ----
#     hq_ids = [b["hq_ids"] for b in batch]   # [B][V]
#     hq_latent_ids = [b["hq_latent_ids"] for b in batch]   # [B][V]
#     lq_ids = [b["lq_ids"] for b in batch]   # [B][V]
#     gt_depth_ids = [b["gt_depth_ids"] for b in batch]   # [B][V]


#     # ---- hq images ----
#     hq_views = []
#     for b in batch:                 # over B
#         v_imgs = []
#         for img in b["hq_views"]:   # over V
#             v_imgs.append(to_tensor(img))
#         hq_views.append(torch.stack(v_imgs, dim=0))  # [V,C,H,W]
#     hq_views = torch.stack(hq_views, dim=0)  # [B,V,C,H,W]

#     # ---- lq images ----
#     lq_views = []
#     for b in batch:                 # over B
#         v_imgs = []
#         for img in b["lq_views"]:   # over V
#             v_imgs.append(to_tensor(img))
#         lq_views.append(torch.stack(v_imgs, dim=0))  # [V,C,H,W]
#     lq_views = torch.stack(lq_views, dim=0)  # [B,V,C,H,W]


#     # ---- depth ----
#     gt_depth = []
#     for b in batch:
#         v_depths = []
#         for d in b["gt_depths"]:
#             v_depths.append(torch.from_numpy(d).unsqueeze(0))  # [1,H,W]
#         gt_depth.append(torch.stack(v_depths, dim=0))  # [V,1,H,W]
#     gt_depth = torch.stack(gt_depth, dim=0)  # [B,V,1,H,W]

#     return {
#         "frame_ids": frame_ids,
        
#         "hq_ids": hq_ids,
#         "hq_views": hq_views,
        
#         'lq_ids': lq_ids,
#         'lq_views': lq_views,
        
#         'gt_depth_ids': gt_depth_ids,
#         'gt_depths': gt_depth,
#     }

def multiview_collate_fn(batch):
    """
    batch: List[dict], length = B
    """

    # ------------------------
    # frame ids (tensor)
    # ------------------------
    frame_ids = torch.stack(
        [torch.as_tensor(b["frame_ids"]) for b in batch],
        dim=0
    )  # (B, V)

    # ------------------------
    # ids (keep Python lists)
    # ------------------------
    hq_ids = [b["hq_ids"] for b in batch]                      # [B][V]
    hq_latent_ids = [b["hq_latent_ids"] for b in batch]        # [B][V]
    lq_ids = [b["lq_ids"] for b in batch]                      # [B][V]
    gt_depth_ids = [b["gt_depth_ids"] for b in batch]          # [B][V]

    # ------------------------
    # HQ images
    # ------------------------
    hq_views = []
    for b in batch:
        v_imgs = [to_tensor(img) for img in b["hq_views"]]     # [V,3,H,W]
        hq_views.append(torch.stack(v_imgs, dim=0))
    hq_views = torch.stack(hq_views, dim=0)                    # (B,V,3,H,W)

    # ------------------------
    # HQ latents
    # ------------------------
    hq_latent_views = []
    for b in batch:
        # each element already torch.Tensor (972, 3072)
        v_latents = [latent for latent in b["hq_latent_views"]]
        hq_latent_views.append(torch.stack(v_latents, dim=0))  # (V,972,3072)
    hq_latent_views = torch.stack(hq_latent_views, dim=0)      # (B,V,972,3072)

    # ------------------------
    # LQ images
    # ------------------------
    lq_views = []
    for b in batch:
        v_imgs = [to_tensor(img) for img in b["lq_views"]]     # [V,3,H,W]
        lq_views.append(torch.stack(v_imgs, dim=0))
    lq_views = torch.stack(lq_views, dim=0)                    # (B,V,3,H,W)

    # ------------------------
    # Depth
    # ------------------------
    gt_depths = []
    for b in batch:
        v_depths = [
            torch.from_numpy(d).unsqueeze(0)                   # (1,H,W)
            for d in b["gt_depths"]
        ]
        gt_depths.append(torch.stack(v_depths, dim=0))         # (V,1,H,W)
    gt_depths = torch.stack(gt_depths, dim=0)                  # (B,V,1,H,W)

    # ------------------------
    # return
    # ------------------------
    return {
        "frame_ids": frame_ids,

        "hq_ids": hq_ids,
        "hq_views": hq_views,

        "hq_latent_ids": hq_latent_ids,
        "hq_latent_views": hq_latent_views,

        "lq_ids": lq_ids,
        "lq_views": lq_views,

        "gt_depth_ids": gt_depth_ids,
        "gt_depths": gt_depths,
    }
