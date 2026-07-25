"""
Qualitative-only MVRM restoration on a self-collected, GT-less real-world video.

Why this exists (not just `python -m depth_anything_3.bench.evaluator`):
`eval_deblur_bench` (used by ALL_IN_ONE_MVRM_PHO_giant.yaml) hardcodes its
dataset list to RealBlur_J/R (see evaluator.py `deblur_bench_root`) and every
metric it computes (PSNR/SSIM/LPIPS, pose/depth-vs-pseudo-GT) requires a
paired sharp ground-truth frame per view. A single handheld video has no such
pairing, so none of those numbers are computable here — this script only
produces visual outputs (restored RGB, depth, camera trajectory).

It reuses the existing Evaluator (for model/denoiser/rgb-decoder loading,
identical to what the CLI entrypoint does) and the public `api.inference()`
call already used by the deblur-bench per-sequence loop
(evaluator.py:344-368) — no framework files are modified.

Usage:
    CUDA_VISIBLE_DEVICES=6 python run_scripts/DA3/val/infer_self_collected_qualitative.py
"""

import glob
import os

import numpy as np
import torch
from addict import Dict

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.cfg import load_config

# ---------------------------------------------------------------------------
CONFIG_PATH = "run_configs/DA3/val/ALL_IN_ONE_MVRM_PHO_giant.yaml"
FRAMES_DIR = "self_collected_motionblur_video/frames/kaist_mydesk"
SCENE_NAME = "kaist_mydesk"
WORK_DIR = "/mnt/dataset1/MV_Restoration/self_collected_qualitative"
NUM_VIEWS = 55  # evenly sampled across the clip (see note below)
# ---------------------------------------------------------------------------


def main():
    config = load_config(CONFIG_PATH, argv=[])
    config.workspace.work_dir = WORK_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model loading: identical to evaluator.py's __main__ + top of infer() ---
    evaluator = Evaluator(
        work_dir=WORK_DIR,
        datas=[],  # deblur-style inference never touches MV_REGISTRY datasets
        modes=["pose"],  # unused by this path, but must be a valid mode
        ref_view_strategy=config.eval.ref_view_strategy,
        max_frames=config.eval.max_frames,
        gpu_id=0,
        total_gpus=1,
        full_cfg=config,
    )

    api = DepthAnything3.from_pretrained(config.model.path).to(device)
    if torch.__version__ >= "2.0":
        api = torch.compile(api)

    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(42)

    # Mirrors evaluator.py Evaluator.infer(), lines ~260-289
    if config.MVRM_EVAL.eval_method == "w_mvrm":
        evaluator.denoiser = evaluator.denoiser.to(device)
        evaluator.denoiser2 = None
    else:
        raise NotImplementedError(
            f"This script only mirrors the w_mvrm setup path "
            f"(config has eval_method={config.MVRM_EVAL.eval_method})"
        )

    if evaluator.rgb_dec_cfg is not None:
        evaluator.rgb_decoder = evaluator.rae.mae_decoder.to(device)
        adapter = getattr(evaluator.rae, "adapter", None)
        evaluator.proj_adapter = adapter.to(device) if adapter is not None else None
    else:
        evaluator.rgb_decoder = None
        evaluator.proj_adapter = None

    # --- Frame selection ---
    # Evenly sample across the whole clip rather than taking `max_frames`
    # consecutive frames: at 24fps, 10 consecutive frames span ~0.4s (barely
    # any camera baseline), which starves the multi-view pose/depth heads of
    # parallax. Spreading the sample across the full ~2.3s clip gives more
    # usable viewpoint diversity.
    all_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    if len(all_frames) > NUM_VIEWS:
        idx = np.linspace(0, len(all_frames) - 1, NUM_VIEWS).round().astype(int)
        frames = [all_frames[i] for i in sorted(set(idx.tolist()))]
    else:
        frames = all_frames
    print(f"[INFO] Using {len(frames)}/{len(all_frames)} frames: {[os.path.basename(f) for f in frames]}")

    n = len(frames)
    dummy_ext = np.eye(4, dtype=np.float32)[None].repeat(n, axis=0)
    dummy_ixt = np.eye(3, dtype=np.float32)[None].repeat(n, axis=0)

    scene_data = Dict()
    scene_data.lq_image_files = frames  # actual model input (the real blurry frames)
    scene_data.image_files = frames  # no clean GT exists; dummy so preprocessing shapes line up
    scene_data.extrinsics = dummy_ext
    scene_data.intrinsics = dummy_ixt
    scene_data.aux = Dict()

    export_dir = os.path.join(WORK_DIR, "model_results", SCENE_NAME, "unposed")
    api.inference(
        scene_data,
        export_dir=export_dir,
        export_format="npz",
        ref_view_strategy=evaluator.ref_view_strategy,
        eval_sampler=evaluator.eval_sampler,
        denoiser=evaluator.denoiser,
        denoiser2=evaluator.denoiser2,
        noise_generator=noise_generator,
        cfg=config,
        scene_info=(SCENE_NAME, "take01"),
        use_pose=False,
        use_ray_pose=config.model.use_ray_pose,
        rgb_decoder=evaluator.rgb_decoder,
        proj_adapter=evaluator.proj_adapter,
    )

    print(f"[DONE] Outputs under {WORK_DIR}:")
    print(f"  pho_vis_results/{SCENE_NAME}/unposed/take01.png        (LQ/HQ-echo/RES panel)")
    print(f"  pho_rgb_recon_results/{SCENE_NAME}/unposed/take01.png  (HQ-echo/LQ/RES rows)")
    print(f"  pho_cam_traj_results/{SCENE_NAME}/unposed/fair_take01.png")
    print(f"  model_results/{SCENE_NAME}/take01/unposed/exports/npz/results.npz  (restored RGB + depth + pose)")


if __name__ == "__main__":
    main()
