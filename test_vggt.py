import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# ============================================================
# Setup
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

# Initialize model
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

data_root_path = f'assets/demo_lq'
    
# Save directory
save_dir = f'./result_val/vggt'
os.makedirs(save_dir, exist_ok=True)


image_names = sorted(glob.glob(f'{data_root_path}/*'))


# Load images (B=1, N, C, H, W internally handled by VGGT utils)
images = load_and_preprocess_images(image_names).to(device)
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    with torch.cuda.amp(dtype=dtype):
        predictions = model(images)

# Safety check
depth_pred = predictions['depth']  # [1, N, H, W, 1]
assert depth_pred.shape[1] == len(image_names)

# ========================================================
# Save per-image depth
# ========================================================
for idx, img_path in enumerate(image_names):
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    # Extract depth
    depth = depth_pred[0, idx, :, :, 0]  # [H, W]

    # Convert to numpy float32
    depth = depth.detach().cpu().numpy().astype(np.float32)

    # Save
    np.save(os.path.join(save_dir, f'{img_id}.npy'), depth)
