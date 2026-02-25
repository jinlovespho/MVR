

import torch
import torch.nn.functional as F

def velocity_direction_loss(pred_vel, gt_vel, eps=1e-8):
    """
    Cosine direction alignment loss for velocity field.

    Args:
        pred_vel: predicted velocity from model
                  shape: (b, v, n, d)
        gt_vel: ground-truth velocity ut from transport
                shape: (b, v, n, d)
        eps: small constant for numerical stability

    Returns:
        scalar loss
    """

    b, v, n, d = pred_vel.shape

    # Flatten spatial dimensions
    pred_flat = pred_vel.reshape(b * v, -1)
    gt_flat   = gt_vel.reshape(b * v, -1)

    # Normalize vectors (safe normalization)
    pred_norm = pred_flat / (pred_flat.norm(dim=1, keepdim=True) + eps)
    gt_norm   = gt_flat   / (gt_flat.norm(dim=1, keepdim=True) + eps)

    cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1, eps=eps)
    loss = (1 - cos_sim).mean()

    return loss



def directional_alignment_loss(restored_latent, hq_latent, lq_latent):
    # all shape: (b, v, n, d)

    # compute direction vectors
    direction_pred = restored_latent - lq_latent
    direction_gt   = hq_latent - lq_latent

    # flatten spatial dims for cosine
    b, v, n, d = direction_pred.shape
    direction_pred = direction_pred.view(b*v, -1)
    direction_gt   = direction_gt.view(b*v, -1)

    cos_sim = F.cosine_similarity(direction_pred, direction_gt, dim=-1)

    # maximize cosine similarity → minimize (1 - cosine)
    loss = (1 - cos_sim).mean()

    return loss



def cross_view_latent_consistency(restored_latent):
    # restored_latent: (b, v, n, d)

    b, v, n, d = restored_latent.shape

    # pairwise difference between adjacent views
    loss = 0.0
    count = 0

    for i in range(v - 1):
        zi = restored_latent[:, i]     # (b, n, d)
        zj = restored_latent[:, i+1]

        loss += ((zi - zj) ** 2).mean()
        count += 1

    return loss / count



