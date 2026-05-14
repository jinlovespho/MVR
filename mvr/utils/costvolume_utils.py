import csv
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATCH_SIZE = 14
PCK_THRESHOLDS = [1, 3, 5, 10]
N_VIS_QUERIES = 9  # 3x3 grid of query patches for visualization

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _tensor_to_rgb(img_chw):
    """(3, H, W) ImageNet-normalized tensor → (H, W, 3) float32 in [0, 1]"""
    img = img_chw.permute(1, 2, 0).float().cpu().numpy()
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)


def _inv_view_order(ref_b_idx, v):
    """
    Return the inverse permutation that restores model-order features to original order.

    The backbone's reorder_by_reference moves the reference view to position 0:
        model order: [ref, 0, 1, ..., ref-1, ref+1, ..., v-1]

    aux features are NOT restored before export (the restore call is commented out),
    so they arrive in model order.  depth / extrinsics / intrinsics go through the DPT
    head which DOES restore order, so they are in original order.

    This function returns inv such that  features[:, inv, :]  has view i at position i
    (original order), making features consistent with depth/poses and with hq_imgs.

    Also returns ref_v: the index of the reference view in original order.
    """
    if ref_b_idx is None:
        return list(range(v)), 0

    ref = int(ref_b_idx[0].item())
    # model_order[new_pos] = orig_pos
    model_order = [ref] + list(range(0, ref)) + list(range(ref + 1, v))
    # inv[orig_pos] = new_pos  (where in model order is each original view)
    inv = [0] * v
    for new_pos, orig_pos in enumerate(model_order):
        inv[orig_pos] = new_pos
    return inv, ref


@torch.no_grad()
def _project_pts_to_patches(pts_world, w2c, K, Ph, Pw):
    """
    Project world-space points into a target view's patch grid.

    Args:
        pts_world : (b, N, 3)  world-space source points
        w2c       : (b, 3, 4)  world-to-camera of target view
        K         : (b, 3, 3)  pixel-space intrinsics of target view
        Ph, Pw    : patch grid height and width

    Returns:
        flat_gt : (b, N) long — flat patch index in target view, -1 = invisible
    """
    R = w2c[:, :3, :3]                                              # (b, 3, 3)
    t = w2c[:, :3, 3]                                               # (b, 3)

    pts_cam = torch.einsum('bij,bnj->bni', R, pts_world) + t.unsqueeze(1)  # (b, N, 3)
    Z = pts_cam[:, :, 2]                                            # (b, N)

    fx = K[:, 0, 0].unsqueeze(1)   # (b, 1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    u = pts_cam[:, :, 0] * fx / Z.clamp(min=1e-8) + cx             # (b, N)
    v = pts_cam[:, :, 1] * fy / Z.clamp(min=1e-8) + cy

    pi_col = (u / PATCH_SIZE).long()
    pi_row = (v / PATCH_SIZE).long()

    valid   = (Z > 0) & (pi_row >= 0) & (pi_row < Ph) & (pi_col >= 0) & (pi_col < Pw)
    flat_gt = pi_row * Pw + pi_col
    flat_gt[~valid] = -1
    return flat_gt                                                   # (b, N)


@torch.no_grad()
def _build_gt_correspondences(depth, intrinsics, extrinsics_w2c, Ph, Pw):
    """
    Build geometry-based GT patch correspondences for all cross-view pairs.

    Args:
        depth          : (b, v, H, W)
        intrinsics     : (b, v, 3, 3)
        extrinsics_w2c : (b, v, 3, 4) world-to-camera
        Ph, Pw         : patch grid shape

    Returns:
        gt_flat : (b, v, v, N) flat GT patch index in target view, -1 = invisible
    """
    b, v, H, W = depth.shape
    N      = Ph * Pw
    device = depth.device

    # Patch-averaged depth
    depth_patch = depth.reshape(b, v, Ph, PATCH_SIZE, Pw, PATCH_SIZE).mean(dim=(3, 5))  # (b,v,Ph,Pw)

    # Patch-center pixel coords
    rows = (torch.arange(Ph, device=device, dtype=torch.float32) + 0.5) * PATCH_SIZE
    cols = (torch.arange(Pw, device=device, dtype=torch.float32) + 0.5) * PATCH_SIZE
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')      # (Ph, Pw)

    fx = intrinsics[:, :, 0, 0]   # (b, v)
    fy = intrinsics[:, :, 1, 1]
    cx = intrinsics[:, :, 0, 2]
    cy = intrinsics[:, :, 1, 2]

    # Unproject patch centres to camera space then world space
    Z = depth_patch                                                  # (b, v, Ph, Pw)
    X = (grid_c[None, None] - cx[:, :, None, None]) * Z / fx[:, :, None, None]
    Y = (grid_r[None, None] - cy[:, :, None, None]) * Z / fy[:, :, None, None]
    pts_cam = torch.stack([X, Y, Z], dim=-1).reshape(b, v, N, 3)    # (b, v, N, 3)

    # c2w from w2c: R_c2w = R^T,  t_c2w = -R^T @ t
    R_w2c = extrinsics_w2c[:, :, :3, :3]                            # (b, v, 3, 3)
    t_w2c = extrinsics_w2c[:, :, :3, 3]                             # (b, v, 3)
    R_c2w = R_w2c.transpose(-1, -2)
    t_c2w = -torch.einsum('bvij,bvj->bvi', R_c2w, t_w2c)           # (b, v, 3)

    pts_world = (torch.einsum('bvij,bvnj->bvni', R_c2w, pts_cam)
                 + t_c2w.unsqueeze(2))                              # (b, v, N, 3)

    # Project each source view's world points into every target view
    gt_flat = torch.full((b, v, v, N), -1, dtype=torch.long, device=device)
    for va in range(v):
        for vb in range(v):
            if va == vb:
                gt_flat[:, va, vb] = torch.arange(N, device=device).unsqueeze(0).expand(b, -1)
            else:
                gt_flat[:, va, vb] = _project_pts_to_patches(
                    pts_world[:, va],
                    extrinsics_w2c[:, vb],
                    intrinsics[:, vb],
                    Ph, Pw,
                )
    return gt_flat                                                   # (b, v, v, N)


@torch.no_grad()
def _compute_pck(patches, gt_flat, Ph, Pw, thresholds=None, center_feats=True):
    """
    Compute PCK for patch-token features using geometry-based GT correspondences.

    For each cross-view pair (va, vb), the predicted match is the argmax of the
    cosine-similarity cost volume; Chebyshev distance to the GT patch is used.

    Args:
        patches      : (b, v, N, D) unnormalized patch features
        gt_flat      : (b, v, v, N) GT flat patch index in target, -1 = invalid
        Ph, Pw       : patch grid dims
        thresholds   : list of int patch-distance thresholds (default PCK_THRESHOLDS)
        center_feats : subtract per-view mean patch token before normalizing (default True).
                       Removes the shared background component so cosine similarity
                       reflects spatial discrimination rather than global image content.

    Returns:
        {t: float PCK@t} averaged over all valid cross-view pairs
    """
    if thresholds is None:
        thresholds = PCK_THRESHOLDS
    b, v, N, _ = patches.shape
    p = patches.float()
    if center_feats:
        p = p - p.mean(dim=2, keepdim=True)          # (b, v, N, D) - mean over N patches
    patches_norm = F.normalize(p, dim=-1)             # (b, v, N, D)

    counts = {t: [] for t in thresholds}

    for va in range(v):
        for vb in range(v):
            if va == vb:
                continue
            sim       = torch.einsum('bnd,bmd->bnm',
                                     patches_norm[:, va],
                                     patches_norm[:, vb])           # (b, N, N)
            pred_flat = sim.argmax(dim=-1)                          # (b, N)
            pred_row  = pred_flat // Pw
            pred_col  = pred_flat % Pw

            gt_f  = gt_flat[:, va, vb]                             # (b, N)
            valid = gt_f >= 0
            if not valid.any():
                continue

            gt_r = gt_f.clamp(min=0) // Pw
            gt_c = gt_f.clamp(min=0) % Pw
            dist = torch.max((pred_row - gt_r).abs(),
                             (pred_col - gt_c).abs()).float()

            for t in thresholds:
                correct = (dist <= t) & valid
                if valid.sum() > 0:
                    counts[t].append(
                        (correct.sum().float() / valid.sum().float()).item()
                    )

    return {t: float(np.mean(counts[t])) if counts[t] else 0.0 for t in thresholds}


def _attn_heatmap(sim, temperature=0.07, topk=None, sharpen_gamma=1.0):
    """
    Convert raw cosine-similarity scores (N,) into a sharpened attention map (N,).

    Three independent sharpening stages, applied in order:
      1. softmax(sim / temperature)  — lower T concentrates mass at the peak
      2. top-k masking               — zero all but the k highest-attention patches
      3. power sharpening (gamma)    — raise to gamma>1 to further suppress background

    Args:
        sim           : (N,) cosine-similarity tensor
        temperature   : softmax temperature; try 0.07 → 0.03 → 0.01 for sharper results
        topk          : if set, keep only the top-k patches and zero the rest
        sharpen_gamma : raise attention weights to this power after softmax (1 = no-op)

    Returns:
        heat : (N,) float32 numpy array, values in [0, 1], peak = 1
    """
    attn = torch.softmax(sim.float() / temperature, dim=0)   # (N,) sums to 1

    if topk is not None and topk < attn.numel():
        # Zero all but the top-k patches; renormalise so they still sum to 1
        vals, idx = torch.topk(attn, k=topk)
        mask = torch.zeros_like(attn)
        mask.scatter_(0, idx, vals)
        attn = mask / (mask.sum() + 1e-8)

    if sharpen_gamma != 1.0:
        attn = attn ** sharpen_gamma
        attn = attn / (attn.sum() + 1e-8)

    heat = attn.cpu().numpy()
    heat = heat / (heat.max() + 1e-8)   # peak always at 1 for consistent colormap
    return heat


def _save_costvolume_vis(hq_patches, lq_patches, res_patches,
                         hq_imgs, gt_flat, Ph, Pw, save_path,
                         lq_imgs=None,
                         ref_v=0, temperature=0.07, topk=None, sharpen_gamma=1.0,
                         center_feats=True):
    """
    Visualize cost-volume attention maps for a 3×3 grid of query patches.

    All inputs (patches, hq_imgs, gt_flat) must already be in original view order.
    Pass ref_v = ref_b_idx so the reference view is used as the query source.

    Saves three files derived from save_path stem:
      cv_vis_hq.png   — HQ feature attention overlaid on HQ images
      cv_vis_lq.png   — LQ feature attention overlaid on LQ images
      cv_vis_res.png  — Restored feature attention overlaid on LQ images

    Layout per file:
      Rows : one per query patch  (N_VIS_QUERIES = 9, 3×3 grid)
      Cols : reference view (query marked) | view_0 attn | view_1 attn | …  (all non-ref views)

    Args:
        hq_patches    : (b, v, N, D)  — features in original view order
        lq_patches    : (b, v, N, D)
        res_patches   : (b, v, N, D)
        hq_imgs       : (v, 3, H, W) ImageNet-normalized tensor  (original order)
        gt_flat       : (b, v, v, N)  — GT correspondences in original view order (unused, kept for API compat)
        Ph, Pw        : patch grid dims
        save_path     : base path; stem is reused with _hq/_lq/_res suffixes
        lq_imgs       : (v, 3, H, W) LQ image tensor; falls back to hq_imgs if None
        ref_v         : index of the reference view in original order (default 0)
        center_feats  : subtract per-view mean patch token before normalizing (default True)
        temperature   : softmax temperature (default 0.07)
        topk          : top-k masking after softmax; None = no masking
        sharpen_gamma : power sharpening exponent (default 1.0 = off)
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    v = hq_patches.shape[1]
    if v < 2:
        return

    def _prep(patches_bvnd):
        p = patches_bvnd[0].float()                        # (v, N, D)
        if center_feats:
            p = p - p.mean(dim=1, keepdim=True)            # subtract mean over N patches
        return F.normalize(p, dim=-1)

    tgt_views = [i for i in range(v) if i != ref_v]       # all non-reference views

    nq_side  = int(round(N_VIS_QUERIES ** 0.5))
    r_idxs   = np.linspace(Ph // 6, Ph - Ph // 6 - 1, nq_side, dtype=int)
    c_idxs   = np.linspace(Pw // 6, Pw - Pw // 6 - 1, nq_side, dtype=int)
    query_rc = [(r, c) for r in r_idxs for c in c_idxs]

    _lq_imgs = lq_imgs if lq_imgs is not None else hq_imgs

    hq_ref_img  = _tensor_to_rgb(hq_imgs[ref_v])
    lq_ref_img  = _tensor_to_rgb(_lq_imgs[ref_v])
    hq_tgt_imgs = [_tensor_to_rgb(hq_imgs[tv]) for tv in tgt_views]
    lq_tgt_imgs = [_tensor_to_rgb(_lq_imgs[tv]) for tv in tgt_views]
    H_img, W_img = hq_ref_img.shape[:2]
    extents  = [[0, W_img, H_img, 0]] * len(tgt_views)

    cmap     = plt.cm.jet
    n_rows   = len(query_rc)
    n_cols   = 1 + len(tgt_views)                         # ref col + one col per target view
    stem     = os.path.splitext(save_path)[0]             # drop .png

    conditions = [
        ('hq',  _prep(hq_patches),  'HQ',       hq_ref_img, hq_tgt_imgs),
        ('lq',  _prep(lq_patches),  'LQ',       lq_ref_img, lq_tgt_imgs),
        ('res', _prep(res_patches), 'Restored',  lq_ref_img, lq_tgt_imgs),
    ]

    for cond_key, feat_n, cond_label, c_ref_img, c_tgt_imgs in conditions:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.5, n_rows * 3.2),
                                 gridspec_kw={'wspace': 0, 'hspace': 0})
        if n_rows == 1:
            axes = axes[None]
        if n_cols == 1:
            axes = axes[:, None]

        # Column headers
        axes[0, 0].set_title('Reference (query)', fontsize=9, fontweight='bold')
        for j, tv in enumerate(tgt_views):
            axes[0, j + 1].set_title(f'{cond_label}  —  view {tv}',
                                     fontsize=9, fontweight='bold')

        for i, (q_r, q_c) in enumerate(query_rc):
            q_flat = q_r * Pw + q_c
            q_px_x = (q_c + 0.5) * PATCH_SIZE
            q_px_y = (q_r + 0.5) * PATCH_SIZE

            # Col 0: reference image with query dot
            ax = axes[i, 0]
            ax.imshow(c_ref_img)
            ax.plot(q_px_x, q_px_y, 'ro',
                    markersize=7, markeredgewidth=1.5, markeredgecolor='w')
            ax.set_xlim(0, W_img)
            ax.set_ylim(H_img, 0)
            ax.axis('off')

            # Cols 1…: attention heatmap overlaid on each target view
            for j, tv in enumerate(tgt_views):
                sim  = (feat_n[ref_v, q_flat] * feat_n[tv]).sum(dim=-1)   # (N,)
                heat = _attn_heatmap(sim, temperature=temperature,
                                     topk=topk, sharpen_gamma=sharpen_gamma)
                heat = heat.reshape(Ph, Pw)

                ax = axes[i, j + 1]
                ax.imshow(c_tgt_imgs[j])
                ax.imshow(heat,
                          extent=extents[j],
                          interpolation='bilinear',
                          cmap=cmap,
                          alpha=0.55,
                          vmin=0, vmax=1.0)
                ax.set_xlim(0, W_img)
                ax.set_ylim(H_img, 0)
                ax.axis('off')

        out = f"{stem}_{cond_key}.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved cost-volume vis] {out}")


def _save_pck_chart(pck_log, save_path, title_suffix=''):
    """
    Save a paper-quality two-panel PCK figure plus one PNG per sub-panel.

    Combined figure (save_path):
      Panel A (top row)  : Line plots — PCK vs layer depth, one subplot per threshold.
      Panel B (bottom)   : PCK threshold curve — PCK vs threshold at the deepest layer.

    Individual figures saved next to save_path:
      pck_layer_t{T}.png         — one per threshold T  (PCK vs layer)
      pck_threshold_curve.png    — PCK vs threshold at deepest layer

    Args:
        pck_log      : {layer_str → {'hq': {t: val}, 'lq': ..., 'res': ...}}
        save_path    : path for the combined PNG  (e.g. …/pck_chart.png)
        title_suffix : appended to suptitle (e.g. '  —  dataset avg (12 scenes)')
    """
    import matplotlib.ticker as ticker

    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    layers_sorted = sorted(pck_log.keys(), key=lambda x: int(x))
    if not layers_sorted:
        return
    thresholds    = PCK_THRESHOLDS
    deepest_layer = layers_sorted[-1]
    x_vals        = [int(l) for l in layers_sorted]

    # ── Shared style ────────────────────────────────────────────────────
    COLORS  = {'hq': '#2E7D32', 'lq': '#C62828', 'res': '#1565C0'}
    MARKERS = {'hq': 'o',       'lq': 's',        'res': '^'}
    LABELS  = {'hq': 'HQ',      'lq': 'LQ',       'res': 'Restored'}
    MS, LW  = 5, 1.8

    RC_COMBINED = {
        'font.family':       'DejaVu Sans',
        'font.size':         9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'axes.grid':         True,
        'grid.alpha':        0.35,
        'grid.linestyle':    '--',
        'grid.linewidth':    0.6,
        'legend.framealpha': 0.92,
        'legend.edgecolor':  '#cccccc',
        'legend.fontsize':   8,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
    }
    # Individual panels get larger fonts for standalone readability
    RC_SOLO = {**RC_COMBINED,
               'font.size':      13,
               'legend.fontsize': 12,
               'xtick.labelsize': 11,
               'ytick.labelsize': 11,
               'axes.linewidth':  1.0,
               'grid.linewidth':  0.8,
               'lines.linewidth': 2.2,
               }

    def _draw_lines(ax, y_by_cond, x_data, ms=MS, lw=LW):
        for cond in ['hq', 'lq', 'res']:
            y = y_by_cond[cond]
            ax.plot(x_data, y,
                    color=COLORS[cond], marker=MARKERS[cond],
                    markersize=ms, linewidth=lw, label=LABELS[cond], zorder=3)

    def _style_layer_ax(ax, t, x_vals, show_legend=False, panel_label=None):
        ax.set_title(f'PCK @ {t} patch{"es" if t > 1 else ""}',
                     fontweight='bold', pad=6)
        ax.set_xlabel('Layer index')
        ax.set_ylabel('PCK')
        ax.set_ylim(-0.02, 1.08)
        if len(x_vals) > 8:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        else:
            ax.set_xticks(x_vals)
        if show_legend:
            ax.legend(loc='lower right')
        if panel_label:
            ax.text(-0.14, 1.06, panel_label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top')

    def _style_threshold_ax(ax, deepest_layer, thresholds, show_legend=True, panel_label=None):
        ax.set_title(f'PCK vs Threshold  (Layer {deepest_layer})',
                     fontweight='bold', pad=6)
        ax.set_xlabel('Patch-distance threshold τ')
        ax.set_ylabel('PCK')
        ax.set_ylim(-0.02, 1.08)
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f'τ={t}' for t in thresholds])
        if show_legend:
            ax.legend(loc='lower right')
        if panel_label:
            ax.text(-0.12, 1.06, panel_label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top')

    n_thresh = len(thresholds)

    # ── 1. Combined figure ───────────────────────────────────────────────
    with plt.rc_context(RC_COMBINED):
        fig = plt.figure(figsize=(3.5 * n_thresh, 7.5))
        gs  = fig.add_gridspec(2, n_thresh, hspace=0.52, wspace=0.40,
                               height_ratios=[1, 1])

        for i, t in enumerate(thresholds):
            ax = fig.add_subplot(gs[0, i])
            y_by_cond = {cond: [pck_log[l][cond].get(t, 0.0) for l in layers_sorted]
                         for cond in ['hq', 'lq', 'res']}
            _draw_lines(ax, y_by_cond, x_vals)
            _style_layer_ax(ax, t, x_vals,
                            show_legend=(i == 0),
                            panel_label=f'({chr(ord("A") + i)})')

        col_start = 1 if n_thresh >= 4 else 0
        col_end   = n_thresh - 1 if n_thresh >= 4 else n_thresh
        ax_b = fig.add_subplot(gs[1, col_start:col_end])
        y_by_cond = {cond: [pck_log[deepest_layer][cond].get(t, 0.0) for t in thresholds]
                     for cond in ['hq', 'lq', 'res']}
        _draw_lines(ax_b, y_by_cond, thresholds)
        _style_threshold_ax(ax_b, deepest_layer, thresholds,
                            panel_label=f'({chr(ord("A") + n_thresh)})')

        fig.suptitle(f'Cross-View Correspondence Quality (PCK){title_suffix}',
                     fontsize=12, fontweight='bold', y=1.02)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    print(f"[Saved PCK chart] {save_path}")

    # ── 2. Individual: one line-plot per threshold ───────────────────────
    stem = os.path.splitext(save_path)[0]   # strip .png so we can append suffixes
    with plt.rc_context(RC_SOLO):
        for t in thresholds:
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
            y_by_cond = {cond: [pck_log[l][cond].get(t, 0.0) for l in layers_sorted]
                         for cond in ['hq', 'lq', 'res']}
            _draw_lines(ax, y_by_cond, x_vals, ms=7, lw=2.2)
            _style_layer_ax(ax, t, x_vals, show_legend=True)
            plt.tight_layout(pad=1.5)
            out = f"{stem}_layer_t{t}.png"
            plt.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"[Saved PCK chart] {out}")

    # ── 3. Individual: PCK threshold curve ──────────────────────────────
    with plt.rc_context(RC_SOLO):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        y_by_cond = {cond: [pck_log[deepest_layer][cond].get(t, 0.0) for t in thresholds]
                     for cond in ['hq', 'lq', 'res']}
        _draw_lines(ax, y_by_cond, thresholds, ms=8, lw=2.2)
        _style_threshold_ax(ax, deepest_layer, thresholds)
        plt.tight_layout(pad=1.5)
        out = f"{stem}_threshold_curve.png"
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved PCK chart] {out}")


@torch.no_grad()
def costvolume_pck(hq_encoder_out, lq_encoder_out, raw_output,
                   hq_imgs=None, lq_imgs=None,
                   save_root=None, scene='scene',
                   ref_b_idx=None,
                   center_feats=True,
                   temperature=0.07, topk=None, sharpen_gamma=1.0,
                   vis_layers=None):
    """
    Build cross-view cost volumes from patch tokens and compute PCK.

    Uses HQ encoder depth + camera parameters as ground-truth geometry.
    Compares HQ, LQ, and restored (RES) features for correspondence quality
    across all exported feature layers.

    View-order note
    ---------------
    The backbone reorders views so the reference view is first (model order).
    Aux features are exported in model order (the restore call is commented out
    in vision_transformer.py).  Depth / extrinsics / intrinsics go through the
    DPT head which DOES restore original order.  hq_imgs is also in original
    order.  Passing ref_b_idx lets us invert the permutation and align everything
    to original order before computing PCK and rendering visualizations.

    Outputs (when save_root is provided):
      {scene}_pck.txt              — per-layer PCK table
      {scene}_pck_chart.png        — grouped bar chart
      {scene}_layer{K}_cv_vis.png  — heatmap visualization at deepest layer

    Args:
        hq_encoder_out : model output for HQ images (.aux, .depth, extrinsics, intrinsics)
        lq_encoder_out : model output for LQ images
        raw_output     : model output after MVRM restoration
        hq_imgs        : (v, 3, H, W) HQ image tensor (ImageNet-normalized, original order)
        lq_imgs        : (v, 3, H, W) LQ image tensor (reserved for future use)
        save_root      : directory for saved outputs; no files written if None
        scene          : scene name used as filename prefix
        ref_b_idx      : reference-view index tensor from encoder_out.ref_b_idx (shape (1,))
        center_feats   : subtract per-view mean patch token before similarity (default True).
                         Single biggest factor for sharp heatmaps — removes shared background.
        temperature    : softmax temperature for visualization (default 0.07; try 0.03/0.01)
        topk           : top-k masking after softmax; None = no masking (e.g. topk=5)
        sharpen_gamma  : power sharpening exponent (default 1.0 = off; try 2.0)
        vis_layers     : list of layer indices (int) to save cv_vis for, e.g. [12, 24, 39].
                         None = save for all exported layers (slow).

    Returns:
        pck_log : dict[layer_str → {'hq': {t: pck}, 'lq': ..., 'res': ...}]
    """
    pck_log = {}

    assert hq_encoder_out.aux.keys() == lq_encoder_out.aux.keys() == raw_output.aux.keys()

    # GT geometry from HQ encoder output — already in original view order
    depth        = hq_encoder_out.depth.float()          # (b, v, H, W)
    extrinsics   = hq_encoder_out['extrinsics'].float()  # (b, v, 3, 4) w2c
    intrinsics_K = hq_encoder_out['intrinsics'].float()  # (b, v, 3, 3)
    b, v, H, W   = depth.shape
    Ph, Pw       = H // PATCH_SIZE, W // PATCH_SIZE

    # Inverse permutation: restores aux features from model order to original order
    inv_order, ref_v = _inv_view_order(ref_b_idx, v)

    # Build GT correspondences once — shared across all feature layers (original order)
    gt_flat = _build_gt_correspondences(depth, intrinsics_K, extrinsics, Ph, Pw)  # (b,v,v,N)

    feat_layer_keys  = list(hq_encoder_out.aux.keys())
    all_layer_indices = [int(k.split('_')[-1]) for k in feat_layer_keys]

    # Resolve vis_layers: None = all, negative indices count from end (like Python lists)
    if vis_layers is None:
        _vis_set = set(all_layer_indices)
    else:
        _vis_set = set()
        for idx in vis_layers:
            if idx < 0:
                _vis_set.add(all_layer_indices[idx])
            else:
                _vis_set.add(idx)

    for feat_layer_key in feat_layer_keys:
        layer_idx = feat_layer_key.split('_')[-1]

        hq_feat  = hq_encoder_out.aux[feat_layer_key]   # (b, v, 1+N, D)  model order
        lq_feat  = lq_encoder_out.aux[feat_layer_key]
        res_feat = raw_output.aux[feat_layer_key]

        # Skip cls/cam token, then restore to original view order so features align
        # with depth/extrinsics/intrinsics (original order) and hq_imgs (original order)
        hq_patches  = hq_feat[:, inv_order, 1:, :]   # (b, v, N, D)
        lq_patches  = lq_feat[:, inv_order, 1:, :]
        res_patches = res_feat[:, inv_order, 1:, :]

        hq_pck  = _compute_pck(hq_patches,  gt_flat, Ph, Pw, center_feats=center_feats)
        lq_pck  = _compute_pck(lq_patches,  gt_flat, Ph, Pw, center_feats=center_feats)
        res_pck = _compute_pck(res_patches, gt_flat, Ph, Pw, center_feats=center_feats)

        pck_log[layer_idx] = {'hq': hq_pck, 'lq': lq_pck, 'res': res_pck}

        # Save heatmap visualization for selected layers only
        if (save_root is not None
                and hq_imgs is not None
                and int(layer_idx) in _vis_set):
            scene_dir = os.path.join(save_root, scene)
            os.makedirs(scene_dir, exist_ok=True)
            _save_costvolume_vis(
                hq_patches, lq_patches, res_patches,
                hq_imgs, gt_flat, Ph, Pw,
                save_path=os.path.join(scene_dir, f"layer{layer_idx}_cv_vis.png"),
                lq_imgs=lq_imgs,
                ref_v=ref_v,
                center_feats=center_feats,
                temperature=temperature,
                topk=topk,
                sharpen_gamma=sharpen_gamma,
            )

    if save_root is not None:
        scene_dir = os.path.join(save_root, scene)
        os.makedirs(scene_dir, exist_ok=True)

        # Text log
        txt_path = os.path.join(scene_dir, 'pck.txt')
        with open(txt_path, 'w') as f:
            col_w = 10
            header = f"{'Layer':<8}" + ''.join(
                f"{'HQ@'+str(t):<{col_w}}{'LQ@'+str(t):<{col_w}}{'RES@'+str(t):<{col_w}}"
                for t in PCK_THRESHOLDS
            )
            f.write(header + '\n' + '-' * len(header) + '\n')
            for lid in sorted(pck_log.keys(), key=lambda x: int(x)):
                row = f"{lid:<8}"
                for t in PCK_THRESHOLDS:
                    row += (f"{pck_log[lid]['hq'].get(t, 0):<{col_w}.4f}"
                            f"{pck_log[lid]['lq'].get(t, 0):<{col_w}.4f}"
                            f"{pck_log[lid]['res'].get(t, 0):<{col_w}.4f}")
                f.write(row + '\n')
        print(f"[Saved PCK log] {txt_path}")

        # JSON
        json_path = os.path.join(scene_dir, 'pck.json')
        with open(json_path, 'w') as f:
            json_log = {
                lid: {cond: {str(t): v for t, v in pck.items()}
                      for cond, pck in entry.items()}
                for lid, entry in pck_log.items()
            }
            json.dump(json_log, f, indent=2)
        print(f"[Saved PCK json] {json_path}")

        # CSV  (one row per layer × condition combination)
        csv_path = os.path.join(scene_dir, 'pck.csv')
        fieldnames = ['scene', 'layer', 'condition'] + [f'pck@{t}' for t in PCK_THRESHOLDS]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for lid in sorted(pck_log.keys(), key=lambda x: int(x)):
                for cond in ['hq', 'lq', 'res']:
                    row = {'scene': scene, 'layer': lid, 'condition': cond}
                    row.update({f'pck@{t}': pck_log[lid][cond].get(t, 0.0)
                                for t in PCK_THRESHOLDS})
                    writer.writerow(row)
        print(f"[Saved PCK csv] {csv_path}")

        _save_pck_chart(pck_log, save_path=os.path.join(scene_dir, 'pck_chart.png'))

        # Re-aggregate all scenes so the dataset summary is always up-to-date.
        aggregate_pck_results(save_root)

    return pck_log


def aggregate_pck_results(save_root):
    """
    Read all per-scene *_pck.csv files in save_root and write dataset-level averages.

    Called automatically at the end of every costvolume_pck call, so by the time
    the last scene finishes the aggregate file is complete.

    Outputs written to save_root:
      _DATASET_avg_pck.csv   — one row per (layer, condition), values averaged over scenes
      _DATASET_avg_pck.json  — same data as nested dict
      _DATASET_avg_pck.txt   — human-readable table

    Args:
        save_root : directory that contains per-scene *_pck.csv files
    """
    import glob

    scene_csvs = sorted(glob.glob(os.path.join(save_root, '*/pck.csv')))
    if not scene_csvs:
        return

    # Accumulate: {(layer, condition) → {pck@t: [values]}}
    accum = {}
    n_scenes = 0
    for csv_path in scene_csvs:
        try:
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['layer'], row['condition'])
                    if key not in accum:
                        accum[key] = {col: [] for col in reader.fieldnames
                                      if col.startswith('pck@')}
                    for col in accum[key]:
                        try:
                            accum[key][col].append(float(row[col]))
                        except (KeyError, ValueError):
                            pass
        except Exception:
            continue
        n_scenes += 1

    if not accum or n_scenes == 0:
        return

    pck_cols = sorted({col for key in accum for col in accum[key]},
                      key=lambda c: int(c.split('@')[1]))

    # Average
    avg = {key: {col: float(np.mean(vals)) if vals else 0.0
                 for col, vals in cols.items()}
           for key, cols in accum.items()}

    # --- CSV ---
    out_csv = os.path.join(save_root, '_DATASET_avg_pck.csv')
    fieldnames = ['n_scenes', 'layer', 'condition'] + pck_cols
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        layers = sorted({k[0] for k in avg}, key=lambda x: int(x))
        for lid in layers:
            for cond in ['hq', 'lq', 'res']:
                key = (lid, cond)
                if key not in avg:
                    continue
                row = {'n_scenes': n_scenes, 'layer': lid, 'condition': cond}
                row.update(avg[key])
                writer.writerow(row)

    # --- JSON ---
    out_json = os.path.join(save_root, '_DATASET_avg_pck.json')
    json_out = {'n_scenes': n_scenes, 'layers': {}}
    layers = sorted({k[0] for k in avg}, key=lambda x: int(x))
    for lid in layers:
        json_out['layers'][lid] = {}
        for cond in ['hq', 'lq', 'res']:
            key = (lid, cond)
            if key in avg:
                json_out['layers'][lid][cond] = avg[key]
    with open(out_json, 'w') as f:
        json.dump(json_out, f, indent=2)

    # --- TXT ---
    out_txt = os.path.join(save_root, '_DATASET_avg_pck.txt')
    col_w = 10
    thresholds = [int(c.split('@')[1]) for c in pck_cols]
    header = f"{'Layer':<8}" + ''.join(
        f"{'HQ@'+str(t):<{col_w}}{'LQ@'+str(t):<{col_w}}{'RES@'+str(t):<{col_w}}"
        for t in thresholds
    )
    with open(out_txt, 'w') as f:
        f.write(f"# Dataset average over {n_scenes} scenes\n")
        f.write(header + '\n' + '-' * len(header) + '\n')
        for lid in layers:
            row = f"{lid:<8}"
            for t in thresholds:
                col = f'pck@{t}'
                row += (f"{avg.get((lid,'hq'),{}).get(col,0):<{col_w}.4f}"
                        f"{avg.get((lid,'lq'),{}).get(col,0):<{col_w}.4f}"
                        f"{avg.get((lid,'res'),{}).get(col,0):<{col_w}.4f}")
            f.write(row + '\n')

    print(f"[Saved dataset avg PCK ({n_scenes} scenes)] {out_txt}")

    # --- Chart (same two-panel format as per-scene) ---
    avg_pck_log = {}
    for lid in layers:
        avg_pck_log[lid] = {}
        for cond in ['hq', 'lq', 'res']:
            key = (lid, cond)
            if key in avg:
                avg_pck_log[lid][cond] = {int(c.split('@')[1]): v
                                          for c, v in avg[key].items()}
            else:
                avg_pck_log[lid][cond] = {t: 0.0 for t in PCK_THRESHOLDS}

    _save_pck_chart(
        avg_pck_log,
        save_path=os.path.join(save_root, '_DATASET_avg_pck_chart.png'),
        title_suffix=f'  —  dataset avg ({n_scenes} scenes)',
    )
