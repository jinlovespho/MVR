import os
import glob
import numpy as np
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3


# init model
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")

data_root_path = f'assets/demo_lq'

# Save directory
save_dir = f'./result_val/da3'
os.makedirs(save_dir, exist_ok=True)


imgs = sorted(glob.glob(f'{data_root_path}/*'))

prediction = model.inference(imgs)
assert len(imgs) == len(prediction.depth)

for img_path, depth in zip(imgs, prediction.depth):
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    # depth: torch.Tensor or numpy → force numpy float32
    if hasattr(depth, 'detach'):
        depth = depth.detach().cpu().numpy()

    depth = depth.astype(np.float32)

    np.save(f'{save_dir}/{img_id}.npy', depth)
    