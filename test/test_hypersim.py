import os
import sys
sys.path.append(os.getcwd())

import cv2 
import glob
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T 

from test_utils import depth_to_colormap, depth_error_to_colormap, depth_error_to_colormap_thresholded

############################################
# Utils
############################################

def safe_mean(x):
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return None
    return x.mean().item()


def compute_depth_metrics_nan_safe(pred, gt, valid):
    eps = 1e-6

    pred = pred[valid]
    gt = gt[valid]

    finite = torch.isfinite(pred) & torch.isfinite(gt)
    pred = pred[finite]
    gt = gt[finite]

    if pred.numel() == 0:
        return None

    # AbsRel
    abs_rel = torch.abs(gt - pred) / (gt + eps)
    abs_rel = abs_rel[torch.isfinite(abs_rel)]

    # SqRel
    sq_rel = (gt - pred) ** 2 / (gt + eps)
    sq_rel = sq_rel[torch.isfinite(sq_rel)]

    # RMSE
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))

    # RMSE log
    log_gt = torch.log(gt + eps)
    log_pred = torch.log(pred + eps)
    finite_log = torch.isfinite(log_gt) & torch.isfinite(log_pred)
    rmse_log = torch.sqrt(
        torch.mean((log_gt[finite_log] - log_pred[finite_log]) ** 2)
    )

    # Delta metrics
    ratio = torch.max(gt / (pred + eps), pred / (gt + eps))
    ratio = ratio[torch.isfinite(ratio)]

    d1 = torch.mean((ratio < 1.25).float()).item()
    d2 = torch.mean((ratio < 1.25 ** 2).float()).item()
    d3 = torch.mean((ratio < 1.25 ** 3).float()).item()

    abs_rel_m = safe_mean(abs_rel)
    sq_rel_m = safe_mean(sq_rel)

    if abs_rel_m is None or sq_rel_m is None:
        return None

    return (
        abs_rel_m,
        sq_rel_m,
        rmse.item(),
        rmse_log.item(),
        d1,
        d2,
        d3,
        int(pred.numel()),
    )


IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    up = down + p
    return up if abs(up - x) <= abs(x - down) else down


def preprocess_image(
    img_path: str,
    process_res: int = 504,
    patch_size: int = 14,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns:
        img_t: torch.Tensor of shape (1, 1, 3, H, W), ImageNet normalized
    """
    # --- Load (BGR → RGB) ---
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Resize longest side to process_res ---
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest != process_res:
        scale = process_res / longest
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # --- Make divisible by patch_size ---
    h, w = img.shape[:2]
    new_w = max(1, nearest_multiple(w, patch_size))
    new_h = max(1, nearest_multiple(h, patch_size))
    if (new_w, new_h) != (w, h):
        upscale = new_w > w or new_h > h
        interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # --- To tensor + normalize ---
    img_t = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
    )
    img_t = IMAGENET_NORMALIZE(img_t)

    return img_t.unsqueeze(0).unsqueeze(0).to(device)




def fmt(x, w=9):
    return f"{x:{w}.4f}"


def fmt_int(x, w=10):
    return f"{x:{w}d}"


def write_scene_header(f, name):
    f.write("\n" + "=" * 90 + "\n")
    f.write(f"SCENE: {name}\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{'IMAGE':<16}"
        f"{'AbsRel':>9}{'SqRel':>9}{'RMSE':>9}{'RMSElog':>10}"
        f"{'δ1':>9}{'δ2':>9}{'δ3':>9}{'Valid':>10}\n"
    )
    f.write("-" * 90 + "\n")


############################################
# Config
############################################


# clean 
# SAVE_ROOT_PATH = "/mnt/dataset1/MV_Restoration/hypersim/da3/clean/input_singleview"
SAVE_ROOT_PATH = "TMPTMP"
IMGS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/data/*/images/*final_preview*/*tonemap*"))


# # deg blur (kernel200)
# SAVE_ROOT_PATH = "/mnt/dataset1/MV_Restoration/hypersim/da3/deg_blur_kernel200/input_singleview"
# IMGS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel200_intensity01/*/*final_hdf5*/images/*"))


# # deg blur (kernel100)
# SAVE_ROOT_PATH = "/mnt/dataset1/MV_Restoration/hypersim/da3/deg_blur_kernel100/input_singleview"
# IMGS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel100_intensity01/*/*final_hdf5*/images/*"))


# # deg blur (kernel50)
# SAVE_ROOT_PATH = "/mnt/dataset1/MV_Restoration/hypersim/da3/deg_blur_kernel50/input_singleview"
# IMGS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel50_intensity01/*/*final_hdf5*/images/*"))


# # deg blur (kernel30)
# SAVE_ROOT_PATH = "/mnt/dataset1/MV_Restoration/hypersim/da3/deg_blur_kernel30/input_singleview"
# IMGS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel30_intensity01/*/*final_hdf5*/images/*"))


# # deg blur (kernel10)
# SAVE_ROOT_PATH = "/mnt/dataset1/MV_Restoration/hypersim/da3/deg_blur_kernel10/input_singleview"
# IMGS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel10_intensity01/*/*final_hdf5*/images/*"))


# gt depth 
DEPTHS_PATH = sorted(glob.glob("/mnt/dataset1/MV_Restoration/hypersim/data/*/images/*geometry_hdf5*/*depth_meters*"))
VIS_DEPTH_PATH = os.path.join(SAVE_ROOT_PATH, "vis_depth")
os.makedirs(SAVE_ROOT_PATH, exist_ok=True)
os.makedirs(VIS_DEPTH_PATH, exist_ok=True)
assert len(IMGS_PATH) == len(DEPTHS_PATH)



############################################
# Main
############################################

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.io.output_processor import OutputProcessor
    model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1").to(device)
    model_output_processor = OutputProcessor()
    model.eval()

    # Hypersim intrinsics
    W, H, FOCAL = 1024, 768, 886.81

    out_txt = open(os.path.join(SAVE_ROOT_PATH, "hypersim_depth_metrics_nan_safe.txt"),"w",encoding="utf-8",)

    all_metrics = []
    scene_metrics = []
    current_scene_cam = None

    for img_path, depth_path in tqdm(zip(IMGS_PATH, DEPTHS_PATH), total=len(IMGS_PATH)):

        # -------------------------------
        # Parse IDs
        # -------------------------------
        scene = img_path.split("/")[-4]
        cam_dir = img_path.split("/")[-2]
        camera = "_".join(cam_dir.split("_")[:3])
        # frame = img_path.split("/")[-1].split(".")[-3]
        frame = depth_path.split('/')[-1].split('.')[-3]
        

        img_id = f"{camera}_{frame}"
        scene_cam = f"{scene}/{camera}"

        if scene_cam != current_scene_cam:
            if len(scene_metrics) > 0:
                m = np.mean(scene_metrics, axis=0)
                out_txt.write("-" * 90 + "\n")
                out_txt.write(
                    f"{'MEAN':<16}"
                    f"{fmt(m[0])}{fmt(m[1])}{fmt(m[2],9)}{fmt(m[3],10)}"
                    f"{fmt(m[4])}{fmt(m[5])}{fmt(m[6])}"
                    f"{fmt_int(len(scene_metrics))}\n"
                )

            write_scene_header(out_txt, scene_cam)
            current_scene_cam = scene_cam
            scene_metrics = []

        # -------------------------------
        # GT depth
        # -------------------------------
        with h5py.File(depth_path, "r") as f:
            dist = f["dataset"][:]

        x = np.linspace(-W / 2 + 0.5, W / 2 - 0.5, W)
        y = np.linspace(-H / 2 + 0.5, H / 2 - 0.5, H)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, FOCAL)
        plane = np.stack([X, Y, Z], axis=-1).astype(np.float32)

        gt_depth = dist / np.linalg.norm(plane, axis=2) * FOCAL
        gt_depth = torch.from_numpy(gt_depth).to(device)
        gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

        # -------------------------------
        # Prediction
        # -------------------------------
        with torch.no_grad():
            img_t = preprocess_image(img_path)
            
            breakpoint()
            
            model_out, mvrm_out = model(img_t, export_feat_layers=[])
            pred = model_output_processor(model_out).depth                  # 768 1024
                    
        pred = torch.from_numpy(pred).to(device)

        pred = F.interpolate(
            pred.unsqueeze(1),
            size=gt_depth.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # -------------------------------
        # Valid mask + median scaling
        # -------------------------------
        valid = (
            (gt_depth > 0.1)
            & (gt_depth < 10.0)
            & torch.isfinite(gt_depth)
            & torch.isfinite(pred)
        )

        if valid.sum() == 0:
            continue

        gt_med = torch.median(gt_depth[valid])
        pred_med = torch.median(pred[valid])

        if not torch.isfinite(gt_med) or pred_med <= 1e-6:
            continue

        pred = pred * (gt_med / pred_med)
        
        
        # -------------------------------
        # Visualization
        # -------------------------------
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)    

        if img_bgr is not None:
            H, W = gt_depth.shape
            img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)   # 768 1024 3

            # Tensor → NumPy
            pred_np = pred.detach().cpu().numpy()       # 768 1024
            gt_np = gt_depth.detach().cpu().numpy()     # 768 1024 

            breakpoint()
            # Depth visualizations
            pred_vis = depth_to_colormap(pred_np)   # 768 1024 3
            gt_vis   = depth_to_colormap(gt_np)

            # Error visualization (AbsRel, log-scaled, TURBO)
            # err_vis = depth_error_to_colormap(gt_np, pred_np)
            err_vis = depth_error_to_colormap_thresholded(gt_np, pred_np, thr=0.1)  # 768 1024 3

            # Concatenate: [RGB | Pred | GT | Error]
            concat = np.concatenate(
                [img_bgr, pred_vis, gt_vis, err_vis],
                axis=1
            )


            # Output path
            viz_dir = os.path.join(VIS_DEPTH_PATH, scene, camera)
            os.makedirs(viz_dir, exist_ok=True)

            viz_path = os.path.join(viz_dir, f"{frame}.png")
            cv2.imwrite(viz_path, concat)

        # -------------------------------
        # Metrics
        # -------------------------------
        metrics = compute_depth_metrics_nan_safe(pred, gt_depth, valid)
        if metrics is None:
            continue

        scene_metrics.append(metrics[:-1])
        all_metrics.append(metrics[:-1])

        out_txt.write(
            f"{img_id:<16}"
            f"{fmt(metrics[0])}{fmt(metrics[1])}"
            f"{fmt(metrics[2],9)}{fmt(metrics[3],10)}"
            f"{fmt(metrics[4])}{fmt(metrics[5])}{fmt(metrics[6])}"
            f"{fmt_int(metrics[7])}\n"
        )

    # -------------------------------
    # Final scene
    # -------------------------------
    if len(scene_metrics) > 0:
        m = np.mean(scene_metrics, axis=0)
        out_txt.write("-" * 90 + "\n")
        out_txt.write(
            f"{'MEAN':<16}"
            f"{fmt(m[0])}{fmt(m[1])}{fmt(m[2],9)}{fmt(m[3],10)}"
            f"{fmt(m[4])}{fmt(m[5])}{fmt(m[6])}"
            f"{fmt_int(len(scene_metrics))}\n"
        )

    # -------------------------------
    # Global mean
    # -------------------------------
    if len(all_metrics) > 0:
        g = np.mean(all_metrics, axis=0)
        out_txt.write("\n" + "=" * 90 + "\n")
        out_txt.write("GLOBAL MEAN (NaN-SAFE)\n")
        out_txt.write("-" * 90 + "\n")
        out_txt.write(
            f"{'ALL':<16}"
            f"{fmt(g[0])}{fmt(g[1])}{fmt(g[2],9)}{fmt(g[3],10)}"
            f"{fmt(g[4])}{fmt(g[5])}{fmt(g[6])}"
            f"{fmt_int(len(all_metrics))}\n"
        )

    out_txt.close()
    print("\n NaN-safe Hypersim depth evaluation finished!")


if __name__ == "__main__":
    main()
