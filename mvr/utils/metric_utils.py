import os
import torch
import numpy as np
from typing import Dict

from depth_anything_3.bench.utils import compute_pose



# -----------------------------------------------------------
# Helper: convert (V,3,4) → (V,4,4)
# -----------------------------------------------------------
def to_4x4(se3_3x4: torch.Tensor):
    if se3_3x4.shape[-2:] == (4, 4):
        return se3_3x4
    V = se3_3x4.shape[0]
    bottom = torch.tensor([0,0,0,1], device=se3_3x4.device).view(1,1,4).repeat(V,1,1)
    return torch.cat([se3_3x4, bottom], dim=1)


# -----------------------------------------------------------
# Depth Metrics
# -----------------------------------------------------------
def compute_depth_metrics(pred, gt, eps=1e-6):
    """
    pred, gt: (V,1,H,W) or (V,H,W)
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    if gt.ndim == 4:
        gt = gt.squeeze(1)

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = torch.max(gt / (pred + eps), pred / (gt + eps))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25**2).float().mean()
    d3 = (thresh < 1.25**3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / (gt + eps))
    sq_rel = torch.mean((gt - pred)**2 / (gt + eps))
    rmse = torch.sqrt(torch.mean((gt - pred)**2))

    return {
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
    }


# -----------------------------------------------------------
# Main Metric Function
# -----------------------------------------------------------
def metric_all(metric_save_root, scene, poses, depths):
    """
    poses = (hq_pose, lq_pose, res_pose)
        each: (V,3,4) or (V,4,4)

    depths = (hq_depth, lq_depth, res_depth)
        each: (1,V,1,H,W) or (V,1,H,W)
    """

    os.makedirs(metric_save_root, exist_ok=True)
    scene = scene.replace("/", "_")
    save_path = os.path.join(metric_save_root, f"{scene}_metrics.txt")

    hq_pose, lq_pose, res_pose = poses
    hq_depth, lq_depth, res_depth = depths

    # --------------------------
    # Pose Metrics
    # --------------------------
    hq_pose = to_4x4(hq_pose)
    lq_pose = to_4x4(lq_pose)
    res_pose = to_4x4(res_pose)

    pose_metric_lq = compute_pose(lq_pose, hq_pose)
    pose_metric_res = compute_pose(res_pose, hq_pose)

    # --------------------------
    # Depth Metrics
    # --------------------------
    # Remove batch dimension if exists
    if hq_depth.ndim == 5:
        hq_depth = hq_depth[0]
        lq_depth = lq_depth[0]
        res_depth = res_depth[0]

    depth_metric_lq = compute_depth_metrics(lq_depth, hq_depth)
    depth_metric_res = compute_depth_metrics(res_depth, hq_depth)

    # --------------------------
    # Save Results
    # --------------------------
    with open(save_path, "w") as f:
        f.write(f"Scene: {scene}\n\n")

        f.write("=== Pose Metrics (AUC) ===\n")
        f.write("LQ vs HQ:\n")
        f.write(f"  AUC30: {pose_metric_lq.auc30:.4f}\n")
        f.write(f"  AUC15: {pose_metric_lq.auc15:.4f}\n")
        f.write(f"  AUC05: {pose_metric_lq.auc05:.4f}\n")
        f.write(f"  AUC03: {pose_metric_lq.auc03:.4f}\n\n")

        f.write("RES vs HQ:\n")
        f.write(f"  AUC30: {pose_metric_res.auc30:.4f}\n")
        f.write(f"  AUC15: {pose_metric_res.auc15:.4f}\n")
        f.write(f"  AUC05: {pose_metric_res.auc05:.4f}\n")
        f.write(f"  AUC03: {pose_metric_res.auc03:.4f}\n\n")

        f.write("=== Depth Metrics ===\n")
        f.write("LQ vs HQ:\n")
        for k, v in depth_metric_lq.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\n")

        f.write("RES vs HQ:\n")
        for k, v in depth_metric_res.items():
            f.write(f"  {k}: {v:.6f}\n")

    print(f"[Saved metrics] {save_path}")
    
    
    