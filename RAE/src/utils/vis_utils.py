import cv2 
import torch 
import numpy as np

def load_depth(depth_path, H, W):
    depth = np.fromfile(depth_path, dtype=np.float32)
    assert depth.size == H * W, f"Size mismatch: {depth_path}"
    depth = depth.reshape(H, W)
    depth[np.isinf(depth)] = np.nan
    return depth


@torch.no_grad()
def align_scale_median(gt, pred, eps=1e-6):
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > eps) & (pred > eps)
    if valid.sum() == 0:
        return pred
    scale = torch.median(gt[valid]) / (torch.median(pred[valid]) + eps)
    return pred * scale


@torch.no_grad()
def compute_depth_metrics(gt, pred, eps=1e-6):
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > eps) & (pred > eps)
    if valid.sum() == 0:
        return [float("nan")] * 7

    gt = torch.clamp(gt[valid], min=eps)
    pred = torch.clamp(pred[valid], min=eps)

    thresh = torch.maximum(gt / pred, pred / gt)

    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    return [
        abs_rel.item(),
        sq_rel.item(),
        rmse.item(),
        rmse_log.item(),
        d1.item(),
        d2.item(),
        d3.item(),
    ]


def depth_to_colormap(depth, invalid_color=(0, 0, 0), resize=None):
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vis = depth.copy()
    vmin, vmax = np.percentile(vis[valid], [2, 98])
    vis = np.clip(vis, vmin, vmax)
    vis = (vis - vmin) / (vmax - vmin + 1e-8)
    vis = (vis * 255).astype(np.uint8)

    # color = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    
    color[~valid] = invalid_color
    
    if resize is not None:
        h, w = resize
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return color


def depth_error_to_colormap(gt, pred, invalid_color=(0, 0, 0), resize=None):
    """
    gt, pred: (H, W) numpy arrays
    resize: (h, w) or None
    """
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        h, w = gt.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        if resize is not None:
            vis = cv2.resize(vis, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
        return vis

    error = np.zeros_like(gt, dtype=np.float32)
    error[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]
    error[valid] = np.log10(error[valid] + 1e-4)

    vmin, vmax = np.percentile(error[valid], [2, 98])
    error = np.clip(error, vmin, vmax)
    error = (error - vmin) / (vmax - vmin + 1e-8)
    error = (error * 255).astype(np.uint8)

    vis = cv2.applyColorMap(error, cv2.COLORMAP_TURBO)
    vis[~valid] = invalid_color

    if resize is not None:
        h, w = resize
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)

    return vis


def depth_error_to_colormap_thresholded(
    gt,
    pred,
    thr=0.5,                 # AbsRel threshold
    invalid_color=(0, 0, 0),
    resize=None,
):
    """
    Visualize ONLY high-error regions using TURBO.
    Low-error pixels are blacked out.

    gt, pred: (H, W) numpy arrays
    thr: AbsRel threshold
    """

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        h, w = gt.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        if resize is not None:
            vis = cv2.resize(vis, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
        return vis

    # AbsRel
    absrel = np.zeros_like(gt, dtype=np.float32)
    absrel[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]

    # High-error mask
    high = valid & (absrel >= thr)
    if high.sum() == 0:
        return np.zeros((*gt.shape, 3), dtype=np.uint8)

    # Optional log compression (recommended)
    err = np.zeros_like(gt, dtype=np.float32)
    err[high] = np.log10(absrel[high] + 1e-4)

    # Normalize only on high-error pixels
    vmin, vmax = np.percentile(err[high], [5, 95])
    err = np.clip(err, vmin, vmax)
    err = (err - vmin) / (vmax - vmin + 1e-8)
    err = (err * 255).astype(np.uint8)

    vis = cv2.applyColorMap(err, cv2.COLORMAP_TURBO)

    # Mask everything else
    vis[~high] = invalid_color

    if resize is not None:
        h, w = resize
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)

    return vis


# ============================================================
# Formatting helpers
# ============================================================
def fmt(val, width=8, prec=4):
    return f"{val:{width}.{prec}f}"

def fmt_int(val, width=12):
    return f"{val:{width},}"

def write_scene_header(f, scene):
    f.write("\n" + "=" * 90 + "\n")
    f.write(f"Scene: {scene}\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{'Image':<12}"
        f"{'AbsRel':>8}"
        f"{'SqRel':>8}"
        f"{'RMSE':>9}"
        f"{'RMSElog':>10}"
        f"{'δ1':>8}"
        f"{'δ2':>8}"
        f"{'δ3':>8}"
        f"{'Pixels':>12}\n"
    )
    f.write("-" * 90 + "\n")