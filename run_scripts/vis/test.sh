python Depth-Anything-3/src/depth_anything_3/utils/reconstruct_npz_scene.py \
    --input /mnt/dataset1/MV_Restoration/NIPS26_RESULTS_RE/jae_val/da3_giant/w_mvrm_FRONT/hyp1tar1__pcloss1__da3-pc_temp001_cycle15-to-mvrm-dec-9__lq2hq-wnoise03__cond/h100_ep20/filtered_cam_blur_150/ray_saddle_maxview10_cfg1/model_results/7scenes/chess/unposed/exports/mini_npz/results.npz \
    --output-root ./results/viz \
    --camera-source pred \
    --sim3 \
    --export glb \
    --export-view-glb \
    --camera-size 0.05 \
    --view-plane-interval 5