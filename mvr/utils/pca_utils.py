import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_pca2_analysis_per_view(
    hq_out,
    lq_out,
    res_out,
    save_root,
    scene,
    patch_grid=(27, 36),
    n_components=2,
    device='cuda'
):

    os.makedirs(save_root, exist_ok=True)
    layers = list(hq_out.aux.keys())

    B = 1
    V = hq_out.aux[layers[0]].shape[1]
    L = len(layers)

    fig, axs = plt.subplots(
        L,
        V * 3,
        figsize=(3 * 3 * V, 3 * L)
    )

    if L == 1:
        axs = np.expand_dims(axs, 0)

    for row_idx, layer in enumerate(layers):
        print(f"PCA 2D analysis for layer {layer} ...")

        def flatten_features(feat, grid_h=27, grid_w=36):
            B, V, N, D = feat.shape
            feat = feat[:, :, 1:, :]
            feat = feat.reshape(B, V, grid_h, grid_w, D)
            return feat.to(device)

        hq_feat = flatten_features(hq_out.aux[layer], *patch_grid)
        lq_feat = flatten_features(lq_out.aux[layer], *patch_grid)
        res_feat = flatten_features(res_out.aux[layer], *patch_grid)

        B, V, H, W, D = hq_feat.shape

        combined = torch.cat([
            hq_feat.reshape(-1, D),
            lq_feat.reshape(-1, D),
            res_feat.reshape(-1, D)
        ], dim=0)

        combined_mean = combined.mean(dim=0, keepdim=True)
        combined_centered = combined - combined_mean
        U, S, Vh = torch.linalg.svd(combined_centered, full_matrices=False)
        pcs = Vh[:n_components].T

        hq_proj = ((hq_feat.reshape(-1, D) - combined_mean) @ pcs).cpu().numpy()
        lq_proj = ((lq_feat.reshape(-1, D) - combined_mean) @ pcs).cpu().numpy()
        res_proj = ((res_feat.reshape(-1, D) - combined_mean) @ pcs).cpu().numpy()

        for v in range(V):
            start = v * H * W
            end = (v + 1) * H * W

            titles = ["HQ", "LQ", "RES"]
            data = [hq_proj, lq_proj, res_proj]
            colors = ["blue", "red", "green"]

            for i in range(3):
                ax = axs[row_idx, v*3 + i]
                ax.scatter(
                    data[i][start:end, 0],
                    data[i][start:end, 1],
                    s=2, alpha=0.5, c=colors[i]
                )

                ax.set_title(f"{titles[i]}\nLayer:{layer} | V{v}", fontsize=8)
                ax.axis("off")

        # Add layer label on left side
        axs[row_idx, 0].set_ylabel(f"Layer: {layer}", fontsize=12)

    plt.suptitle(f"{scene} - 2D PCA Scatter (Top {n_components} PCs)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{scene}_PCA2_all_views.png"))
    plt.close()

def plot_pca3_analysis_per_view(
    hq_out,
    lq_out,
    res_out,
    save_root,
    scene,
    patch_grid=(27, 36),
    device='cuda'
):

    os.makedirs(save_root, exist_ok=True)
    layers = list(hq_out.aux.keys())

    B = 1
    V = hq_out.aux[layers[0]].shape[1]
    L = len(layers)
    n_components = 3

    fig, axs = plt.subplots(
        L,
        V * 3,
        figsize=(3 * 3 * V, 3 * L)
    )

    if L == 1:
        axs = np.expand_dims(axs, 0)

    for row_idx, layer in enumerate(layers):
        

        def flatten_features(feat, grid_h=27, grid_w=36):
            B, V, N, D = feat.shape
            feat = feat[:, :, 1:, :]
            feat = feat.reshape(B, V, grid_h, grid_w, D)
            return feat.to(device)

        hq_feat = flatten_features(hq_out.aux[layer], *patch_grid)
        lq_feat = flatten_features(lq_out.aux[layer], *patch_grid)
        res_feat = flatten_features(res_out.aux[layer], *patch_grid)

        B, V, H, W, D = hq_feat.shape

        combined = torch.cat([
            hq_feat.reshape(-1, D),
            lq_feat.reshape(-1, D),
            res_feat.reshape(-1, D)
        ], dim=0)

        combined_mean = combined.mean(dim=0, keepdim=True)
        combined_centered = combined - combined_mean
        U, S, Vh = torch.linalg.svd(combined_centered, full_matrices=False)
        pcs = Vh[:n_components].T

        def project_to_rgb(feat):
            proj = (feat.reshape(-1, D) - combined_mean) @ pcs
            proj = proj.reshape(B, V, H, W, 3)
            min_val = proj.amin(dim=(0,1,2,3), keepdim=True)
            max_val = proj.amax(dim=(0,1,2,3), keepdim=True)
            return (proj - min_val) / (max_val - min_val + 1e-8)

        hq_rgb = project_to_rgb(hq_feat).cpu().numpy()
        lq_rgb = project_to_rgb(lq_feat).cpu().numpy()
        res_rgb = project_to_rgb(res_feat).cpu().numpy()

        titles = ["HQ", "LQ", "RES"]
        data = [hq_rgb, lq_rgb, res_rgb]

        for v in range(V):
            for i in range(3):
                ax = axs[row_idx, v*3 + i]
                ax.imshow(data[i][0, v])
                ax.set_title(f"{titles[i]}\nLayer:{layer} | V{v}", fontsize=8)
                ax.axis("off")

        # Row label
        axs[row_idx, 0].set_ylabel(f"Layer: {layer}", fontsize=12)

    plt.suptitle(f"{scene} - PCA RGB Colormap (Top 3 PCs)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, f"{scene}_PCA_RGB_all_views.png"))
    plt.close()
    print(f"PCA RGB analysis done for all layers. Saved to {save_root}")