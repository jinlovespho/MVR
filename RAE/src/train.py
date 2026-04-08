# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import argparse
import math
import os
from collections import defaultdict
import cv2 
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm 
import argparse
from pathlib import Path
import math
from omegaconf import OmegaConf

from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors   
from depth_anything_3.utils.geometry import unproject_depth, affine_inverse, as_homogeneous


##### model imports

from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *
from utils.vis_utils import *
from utils.loss_utils import velocity_direction_loss, camera_loss_single

import torch.nn.functional as F 
from torchvision.utils import save_image 
from utils.vis_utils import depth_to_colormap, depth_error_to_colormap_thresholded, tensor_to_uint8_image
import torchvision

from einops import rearrange
from RAE.src import initialize
from motionblur.motionblur import Kernel 
import matplotlib.pyplot as plt



from mvr.utils.featsim_utils import *
from mvr.utils.metric_utils import *
from mvr.utils.freq_utils import *
from mvr.utils.pca_utils import *

from RAE.src.utils.vis_utils import vis_all
from RAE.src.vis_cam_pose import plot_cam_trajectory, plot_cam_trajectory_fair, plot_all_cam_trajectory_fair


# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    args = parser.parse_args()
    return args


def main():
    
    # # NAN DEBUG
    # torch.autograd.set_detect_anomaly(True)
    
    
    # set up ddp setting
    rank, world_size, device = setup_distributed()
    
    
    # load configs
    args = parse_args()
    full_cfg = OmegaConf.load(args.config)
    training_cfg = full_cfg.training 
    
    
    # set logger and directories
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(full_cfg, rank)



    time_dist_shift = math.sqrt(full_cfg.misc.time_dist_shift_dim / full_cfg.misc.time_dist_shift_base)




    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    log_interval = int(training_cfg.get("log_interval", 100))
    # sample_every = int(training_cfg.get("sample_every", 2500)) 
    # checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ckpt_step_interval = int(training_cfg.get('ckpt_step_interval', 25000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    global_seed = int(training_cfg.get("global_seed", 0))
    

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    
    
    # load encoder and denoiser 
    models, processors = initialize.load_model(full_cfg, rank, device)
    

    # load training and validation data 
    train_loader, train_sampler = initialize.load_train_data(full_cfg, micro_batch_size, rank, world_size)
    loader_batches = len(train_loader)
    steps_per_epoch = math.ceil(loader_batches / grad_accum_steps)
    
    if rank==0 and len(full_cfg.data.val.list) != 0:
        val_loader, val_sampler = initialize.load_val_data(full_cfg, 1, rank, world_size)


    # load optimizer
    optimizer, optim_msg = build_optimizer([p for p in models['denoiser'].parameters() if p.requires_grad], training_cfg)


    # load scheduler 
    if full_cfg.get('scheduler'):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    else:
        scheduler=None
        sched_msg=None
    
    # load Transport 
    transport = create_transport(**full_cfg.transport.params, time_dist_shift=time_dist_shift,)
    transport_sampler = Sampler(transport)


    # load sampler 
    eval_sampler = initialize.load_sampler(full_cfg, transport_sampler)
    
    
    
    ema_model_fn = models['ema_denoiser'].forward

    
    val_noise_generator = torch.Generator(device=device)
    val_noise_generator.manual_seed(global_seed)  # any fixed seed you like

    
    ### Resuming and checkpointing
    start_epoch = 0
    global_train_step = 0
    optimizer_step = 0 
    running_loss = 0.0

    
    # maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    maybe_resume_ckpt_path = full_cfg.stage_2.ckpt
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_train_step = initialize.load_checkpoint(
                ckpt_path,
                models['ddp_denoiser'],
                models['ema_denoiser'],
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_train_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
            
            
            
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in models['encoder'].parameters())
        logger.info(f"Stage-1 Encoder parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in models['denoiser'].parameters() if p.requires_grad)
        logger.info(f"Stage-2 Denoiser parameters: {num_params/1e6:.2f}M")
        logger.info(f"Clipping gradients to max norm {clip_grad}.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU. grad accum {full_cfg.training.grad_accum_steps} per GPU")
        logger.info(f"Dataset contains total {len(train_loader.dataset)} training samples, {steps_per_epoch} steps per epoch.")
        for train_ds in train_loader.dataset.datasets:
            logger.info(f'  - {train_ds.ds_name}: {len(train_ds)}')
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    IMAGENET_NORMALIZE = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
    

    dist.barrier() 
    for epoch in range(start_epoch, num_epochs):
        models['ddp_denoiser'].train()
        train_sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)


        # train loop
        for train_step, batch in enumerate(train_loader):


            # load batch data
            # train_frame_id = batch['frame_ids']               # b v
            train_hq_id = batch['hq_ids']                     # len(hq_id) = b, len(hq_id[i]) = v
            # train_gt_depth = batch['gt_depths'].to(device)    # b v 1 378 504
            train_hq_views = batch['hq_views'].to(device)     # b v 3 378 504
            train_lq_views = batch['lq_views'].to(device)     # b v 3 378 504


            # apply imagenet normalization
            train_b, train_v, train_c, train_h, train_w = train_hq_views.shape 
            train_hq_views = IMAGENET_NORMALIZE(train_hq_views.view(train_b*train_v, train_c, train_h, train_w)).view(train_b, train_v, train_c, train_h, train_w)
            train_lq_views = IMAGENET_NORMALIZE(train_lq_views.view(train_b*train_v, train_c, train_h, train_w)).view(train_b, train_v, train_c, train_h, train_w)


            print(train_hq_views.shape)

            
            # lq view forward pass
            with torch.no_grad():
                lq_encoder_out, lq_mvrm_out = models['encoder'](
                                                    image=train_lq_views, 
                                                    # export_feat_layers=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39], 
                                                    export_feat_layers=[], 
                                                    mvrm_cfg=full_cfg.mvrm.train, 
                                                    mode='train'
                                                    )
            lq_pred_pose_enc = lq_encoder_out.pose_enc
            lq_pred_pose = lq_encoder_out['extrinsics'] # b v 3 4
            lq_ref_b_idx = lq_encoder_out.ref_b_idx
            lq_encoder_out = processors['encoder_output_processor'](lq_encoder_out)
            train_lq_pred_depth_np = lq_encoder_out.depth                  # b v 378 504
            train_lq_pred_depth = torch.from_numpy(train_lq_pred_depth_np).to(device) 
            # lq_latent = lq_mvrm_out['extract_feat']      # b v 973 3072
            lq_latent = lq_mvrm_out[('extract_feat', full_cfg.mvrm.train.extract_feat_layers[0])]
                        

            # hq forward pass
            with torch.no_grad():
                hq_encoder_out, hq_mvrm_out = models['encoder'](
                                                    image=train_hq_views, 
                                                    # export_feat_layers=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39], 
                                                    export_feat_layers=[], 
                                                    mvrm_cfg=full_cfg.mvrm.train, 
                                                    mode='train',
                                                    ref_b_idx=lq_ref_b_idx,
                                                    # ref_b_idx=None
                                                    analysis = full_cfg.analysis_HQ
                                                    )
            hq_pred_pose_enc = hq_encoder_out.pose_enc
            hq_pred_intrinsics = hq_encoder_out['intrinsics']  # (b, v, 3, 3) tensor
            hq_pred_pose = hq_encoder_out['extrinsics'] # b v 3 4
            hq_encoder_out = processors['encoder_output_processor'](hq_encoder_out)
            train_hq_pred_depth_np = hq_encoder_out.depth                  # b v 378 504
            train_hq_pred_depth = torch.from_numpy(train_hq_pred_depth_np).to(device) 
            # hq_latent = hq_mvrm_out['extract_feat']
            hq_latent = hq_mvrm_out[('extract_feat', full_cfg.mvrm.train.extract_feat_layers[0])]
            assert lq_latent.shape == hq_latent.shape 
            

            # HQ attn_map extraction 
            hq_maps = {} 
            if full_cfg.analysis_HQ.vis_map and len(full_cfg.analysis_HQ.da3_attn_map.extract_idx) != 0:
                for layer_idx in full_cfg.analysis_HQ.da3_attn_map.extract_idx:
                    print('HQ DA3 ATTENTION MAP EXTRACTION - LAYER ', layer_idx)
                    attn_idx, attn_type, attn_map = models['encoder'].model.backbone.pretrained.blocks[layer_idx].attn.attn_map
                    assert layer_idx==attn_idx 
                    # hq_maps[('da3', attn_idx, attn_type)] = attn_map  # 1 head num_view*(n+1) num_view*(n+1) 
                    hq_maps[('da3', attn_idx, attn_type)] = attn_map.mean(dim=1)  # 1 head num_view*(n+1) num_view*(n+1)  -> 1 num_view*(n+1) num_view*(n+1)
                # to_vis_imgs_list.append(('hq_imgs', imgs))
                # to_vis_attn_maps_list.append(('hq_maps', hq_maps))
                
            

                                    
            # if full_cfg.analysis_HQ.get('vis_pointcloud', False):
            #     print("HQ POINT_CLOUD EXTRACTION")
            #     from depth_anything_3.utils.geometry import unproject_depth, affine_inverse, as_homogeneous

            #     # train_hq_pred_depth: (b, v, H, W) already on device
            #     # hq_pred_pose:        (b, v, 3, 4) w2c already on device
            #     # hq_pred_intrinsics:  (b, v, 3, 3) saved above

            #     depth_t = train_hq_pred_depth.unsqueeze(-1).float()   # (b, v, H, W, 1)
            #     c2w_t   = affine_inverse(as_homogeneous(hq_pred_pose)) # (b, v, 4, 4)

            #     pts = unproject_depth(depth_t, hq_pred_intrinsics, c2w_t)  # (b, v, H, W, 3)

            #     # colors from input images: (b, v, 3, H, W) -> (b, v, H, W, 3)
            #     colors = train_hq_views.permute(0, 1, 3, 4, 2).float()
                
                







            # # POSE DEBUG
            # val_noise_generator.manual_seed(global_seed)
            # pure_noise = torch.randn(lq_latent.shape, generator=val_noise_generator, device=device, dtype=torch.float32)
            # if full_cfg.mvrm.lq_latent_cond == 'addition':
            #     xt = pure_noise + lq_latent            
            # model_kwargs={
            #     'model_img_size': (train_h, train_w)
            # }
            # with torch.no_grad():
            #     restored_samples = eval_sampler(xt, ema_model_fn, **model_kwargs)[-1]     # b v n d
            # mvrm_result={}
            # mvrm_result['restored_latent'] = restored_samples
            # with torch.no_grad():
            #     encoder_out, val_mvrm_out = models['encoder'](
            #                                                 image=train_lq_views, 
            #                                                 export_feat_layers=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39], 
            #                                                 mvrm_cfg=full_cfg.mvrm.val, 
            #                                                 mvrm_result=mvrm_result,
            #                                                 mode='val'
            #                                                 )
            # encoder_out = processors['encoder_output_processor'](encoder_out)
            # pred_depth_np = encoder_out.depth   # num_view h w
            # pred_depth = torch.from_numpy(pred_depth_np).to(device)
            
            
            
            
            # sim_log = {}  # layer -> dict of metrics
            # assert hq_encoder_out.aux.keys() == lq_encoder_out.aux.keys() == encoder_out.aux.keys()
            # for feat_layer_key in hq_encoder_out.aux.keys():
            #     layer_idx = feat_layer_key.split('_')[-1]

            #     hq_feat = hq_encoder_out.aux[feat_layer_key]  # (V, 1+N, D)
            #     lq_feat = lq_encoder_out.aux[feat_layer_key]  # (V, 1+N, D)
            #     res_feat = encoder_out.aux[feat_layer_key]    # (V, 1+N, D)

            #     # tokens
            #     cam_tkn_hq = hq_feat[:, 0:1]    # (V, 1, D)
            #     cam_tkn_lq = lq_feat[:, 0:1]
            #     cam_tkn_res = res_feat[:, 0:1]

            #     patch_tkn_hq = hq_feat[:, 1:]   # (V, N, D)
            #     patch_tkn_lq = lq_feat[:, 1:]
            #     patch_tkn_res = res_feat[:, 1:]

            #     # --- overall (all tokens)
            #     all_hq_lq = cosine_sim_mean(hq_feat, lq_feat)
            #     all_hq_res = cosine_sim_mean(hq_feat, res_feat)

            #     # --- camera token similarity (mean over V; the token axis is size 1 anyway)
            #     cam_hq_lq = cosine_sim_mean(cam_tkn_hq, cam_tkn_lq)
            #     cam_hq_res = cosine_sim_mean(cam_tkn_hq, cam_tkn_res)

            #     # --- patch token similarity stats (over V*N tokens)
            #     patch_hq_lq_stats = cosine_sim_stats(patch_tkn_hq, patch_tkn_lq)
            #     patch_hq_res_stats = cosine_sim_stats(patch_tkn_hq, patch_tkn_res)

            #     sim_log[layer_idx] = {
            #         "all_tokens": {
            #             "hq_vs_lq_mean": all_hq_lq,
            #             "hq_vs_res_mean": all_hq_res,
            #         },
            #         "camera_token": {
            #             "hq_vs_lq_mean": cam_hq_lq,
            #             "hq_vs_res_mean": cam_hq_res,
            #         },
            #         "patch_tokens": {
            #             "hq_vs_lq": patch_hq_lq_stats,
            #             "hq_vs_res": patch_hq_res_stats,
            #         }
            #     }

            #     # optional: quick print
            #     print(
            #         f"[layer {layer_idx}] "
            #         f"all(hq,lq)={all_hq_lq:.4f} all(hq,res)={all_hq_res:.4f} | "
            #         f"cam(hq,lq)={cam_hq_lq:.4f} cam(hq,res)={cam_hq_res:.4f} | "
            #         f"patch_mean(hq,lq)={patch_hq_lq_stats['mean']:.4f} patch_mean(hq,res)={patch_hq_res_stats['mean']:.4f}"
            #     )




            # # plot_save_root = f'plots_rere/wo_mvrm/{train_step}'
            # # plot_save_root = f'plots_rere/w_mvrm_JIHYE2_lqkernel100/{train_step}'
            # plot_save_root = f'plots_rere/w_mvrm_JIHYE2_lqkernel200/{train_step}'






            # plot_two_similarity_curves(
            #     sim_log,
            #     key1="all_tokens",
            #     key2_a="hq_vs_lq_mean",
            #     key2_b="hq_vs_res_mean",
            #     label_a="HQ vs LQ",
            #     label_b="HQ vs Restored",
            #     title="All Tokens Similarity Across Layers",
            #     save_path=f"{plot_save_root}/sim_all_tokens_combined.png"
            # )




            # plot_two_similarity_curves(
            #     sim_log,
            #     key1="camera_token",
            #     key2_a="hq_vs_lq_mean",
            #     key2_b="hq_vs_res_mean",
            #     label_a="HQ vs LQ",
            #     label_b="HQ vs Restored",
            #     title="Camera Token Similarity Across Layers",
            #     save_path=f"{plot_save_root}/sim_camera_token_combined.png"
            # )

            
            # plot_two_similarity_curves(
            #     sim_log,
            #     key1="patch_tokens",
            #     key2_a="hq_vs_lq",
            #     key2_b="hq_vs_res",
            #     label_a="HQ vs LQ",
            #     label_b="HQ vs Restored",
            #     title="Patch Token Similarity Across Layers",
            #     save_path=f"{plot_save_root}/sim_patch_tokens_combined.png"
            # )
            
            # save_image(train_lq_views.squeeze(0), f"{plot_save_root}/img_lq.png", normalize=True)
            # save_image(train_hq_views.squeeze(0), f"{plot_save_root}/img_hq.png", normalize=True)
            # save_image(train_hq_pred_depth.unsqueeze(1), f"{plot_save_root}/img_hq_depth.png", normalize=True)
            # save_image(train_lq_pred_depth.unsqueeze(1), f"{plot_save_root}/img_lq_depth.png", normalize=True)
            # save_image(pred_depth.unsqueeze(1), f"{plot_save_root}/img_res_depth.png", normalize=True)


            
            
            # print(lq_pred_pose_enc.requires_grad)
            
            # # VISUALIZE CAMERA POSES 
            # from vis_cam_pose import plot_cam_trajectory
            # gt_pose = batch['poses']     # b v 4 4 
            # save_image(batch['hq_views'].squeeze(0), 'TMP/hq.jpg')
            # save_image(batch['lq_views'].squeeze(0), 'TMP/lq.jpg')
            # plot_cam_trajectory(gt_pose[0], hq_pred_pose[0], lq_pred_pose[0], save_path="TMP/cam_traj.png")
            # print(batch['hq_ids'])
        
            
            if train_b==1 and len(train_hq_pred_depth_np.shape)<4 and len(train_lq_pred_depth_np.shape)<4:
                train_hq_pred_depth_np = np.expand_dims(train_hq_pred_depth_np, axis=0)
                train_lq_pred_depth_np = np.expand_dims(train_lq_pred_depth_np, axis=0)                
                # pred_depth_np = np.expand_dims(pred_depth_np, axis=0)                


            if rank == 0 and training_cfg.vis.train_depth_every > 0 and global_train_step % training_cfg.vis.train_depth_every == 0:
                logger.info(f"Train sample shape: {train_hq_views.shape}")

                vis_hq_rgb = []
                vis_lq_rgb = []
                vis_res_rgb = []

                vis_hq_depth = []
                vis_lq_depth = []
                vis_res_depth = []

                for view_idx in range(train_v):

                    # ---------------- RGB ----------------
                    hq_rgb = tensor_to_uint8_image(train_hq_views[0, view_idx])
                    lq_rgb = tensor_to_uint8_image(train_lq_views[0, view_idx])

                    vis_hq_rgb.append(hq_rgb)
                    vis_lq_rgb.append(lq_rgb)
                    vis_res_rgb.append(lq_rgb)  # placeholder (no restored RGB yet)

                    # ---------------- Depth ----------------
                    vis_hq_depth.append(
                        depth_to_colormap(train_hq_pred_depth_np[0, view_idx])
                    )
                    vis_lq_depth.append(
                        depth_to_colormap(train_lq_pred_depth_np[0, view_idx])
                    )
                    # vis_res_depth.append(
                    #     depth_to_colormap(pred_depth_np[0, view_idx])
                    # )

                # ------------------------------------------------
                # Concatenate multi-view horizontally
                # ------------------------------------------------
                vis_hq_rgb    = np.concatenate(vis_hq_rgb, axis=1)
                vis_lq_rgb    = np.concatenate(vis_lq_rgb, axis=1)
                # vis_res_rgb   = np.concatenate(vis_res_rgb, axis=1)

                vis_hq_depth  = np.concatenate(vis_hq_depth, axis=1)
                vis_lq_depth  = np.concatenate(vis_lq_depth, axis=1)
                # vis_res_depth = np.concatenate(vis_res_depth, axis=1)

                # ------------------------------------------------
                # Build 3-column rows
                # ------------------------------------------------

                # Row 1: RGB comparison
                row_rgb = np.concatenate(
                    [
                        vis_hq_rgb,
                        vis_lq_rgb,
                        # vis_res_rgb
                    ],
                    axis=1
                )

                # Row 2: Depth comparison
                row_depth = np.concatenate(
                    [
                        vis_hq_depth,
                        vis_lq_depth,
                        # vis_res_depth
                    ],
                    axis=1
                )

                # Stack rows vertically
                vis_train_all = np.concatenate(
                    [
                        row_rgb[:, :, ::-1],  # RGB -> BGR for cv2
                        row_depth
                    ],
                    axis=0
                )

                # ------------------------------------------------
                # Save
                # ------------------------------------------------
                vis_train_depth_save_dir = f"{experiment_dir}/vis_train_depth"
                os.makedirs(vis_train_depth_save_dir, exist_ok=True)

                vis_id = "-".join(train_hq_id[0])

                cv2.imwrite(
                    f"{vis_train_depth_save_dir}/step{global_train_step:07}_{vis_id}.jpg",
                    vis_train_all
                )


            # # ------------------------------------------------
            # # VISUALIZE TRAIN (only first batch)
            # # ------------------------------------------------
            # if rank == 0 and training_cfg.vis.train_depth_every > 0 and global_train_step % training_cfg.vis.train_depth_every == 0:
            #     logger.info(f"Train sample shape: {train_hq_views.shape}")
            #     vis_train_hq_rgb = []
            #     vis_train_lq_rgb = []
            #     vis_train_hq_depth = []
            #     vis_train_lq_depth = []
            #     vis_train_res_depth = []
            #     for view_idx in range(train_v):
            #         # RGB (unnormalized!)
            #         vis_train_hq_rgb.append(tensor_to_uint8_image(train_hq_views[0, view_idx]))
            #         vis_train_lq_rgb.append(tensor_to_uint8_image(train_lq_views[0, view_idx]))
            #         # Depths
            #         vis_train_hq_depth.append(depth_to_colormap(train_hq_pred_depth_np[0, view_idx]))
            #         vis_train_lq_depth.append(depth_to_colormap(train_lq_pred_depth_np[0, view_idx]))
            #         vis_train_res_depth.append(depth_to_colormap(pred_depth_np[0, view_idx]))
            #     # concatenate views along width
            #     vis_train_hq_rgb    = np.concatenate(vis_train_hq_rgb, axis=1)
            #     vis_train_lq_rgb    = np.concatenate(vis_train_lq_rgb, axis=1)
            #     vis_train_hq_depth  = np.concatenate(vis_train_hq_depth, axis=1)
            #     vis_train_lq_depth  = np.concatenate(vis_train_lq_depth, axis=1)
            #     vis_train_res_depth = np.concatenate(vis_train_res_depth, axis=1)
            #     # stack rows (modalities)
            #     vis_train_top = np.concatenate([vis_train_hq_rgb, vis_train_lq_rgb], axis=1)
            #     vis_train_bot = np.concatenate([vis_train_hq_depth, vis_train_lq_depth], axis=1)
            #     vis_train_all = np.concatenate([vis_train_top[:,:,::-1], vis_train_bot], axis=0)
            #     # save
            #     vis_train_depth_save_dir = f"{experiment_dir}/vis_train_depth"
            #     os.makedirs(vis_train_depth_save_dir, exist_ok=True)
            #     vis_id = "-".join(train_hq_id[0])
            #     cv2.imwrite(f"{vis_train_depth_save_dir}/step{global_train_step:07}_{vis_id}.jpg",vis_train_all)



            # compute loss (per microbatch)
            transport_output = transport.training_losses_mvrm(
                model=models['ddp_denoiser'],
                x1=hq_latent,
                xcond=lq_latent,
                model_img_size=(train_h, train_w),
                cfg=full_cfg
            )
            
            
            mvrm_maps = None 
            if full_cfg.mvrm.analysis.vis_attn_map:
                mvrm_maps = transport_output.get('mvrm_maps', None)
                
                            
            # flow matching velocity loss
            transport_loss = transport_output["loss"].mean()
            loss = transport_loss
                        
                        
                    
            def cross_entropy_attn(pred, target, row_mask=None, eps=1e-8):
                """CAMEO-style cross-entropy between attention probability distributions.
                row_mask: (b, v*n) bool tensor; if given, averages only over True rows."""
                per_row = -(target * (pred + eps).log()).sum(dim=-1)  # (b, v*n)
                if row_mask is not None:
                    return per_row[row_mask].mean()
                return per_row.mean()


            # Attention alignment loss (CAMEO-style)
            attn_loss = torch.tensor(0.0, device=device)
            if full_cfg.mvrm.loss.attn_align.use and mvrm_maps is not None:
                lambda_attn = full_cfg.mvrm.loss.attn_align.lambda_coeff
                mvrm_key = ('mvrm', full_cfg.mvrm.loss.attn_align.mvrm_layer_idx, 'global')
                pred_map = mvrm_maps[mvrm_key]  # (1, v*(n+1), v*(n+1))


                if full_cfg.mvrm.loss.attn_align.da3_attn_map.use and hq_maps:
                    print('attention alignment - HQ attention map')
                    # target: HQ DA3 attention map
                    da3_key    = ('da3', full_cfg.mvrm.loss.attn_align.da3_attn_map.da3_layer_idx, 'global')
                    target_map = hq_maps[da3_key].to(device)      # (1, N, N)
                    attn_loss  = cross_entropy_attn(pred_map, target_map)
                    loss += lambda_attn * attn_loss

                elif full_cfg.mvrm.loss.attn_align.use and full_cfg.mvrm.loss.attn_align.da3_point_cloud.use:
                    print(f'attention alignment - HQ point cloud correspondence map - temperature {full_cfg.mvrm.loss.attn_align.da3_point_cloud.get("vis_pc_temperature", "None")}')
                    # target: geometric correspondence from HQ point cloud
                    depth_t = train_hq_pred_depth.unsqueeze(-1).float()                 # (b, v, H, W, 1)
                    c2w_t   = affine_inverse(as_homogeneous(hq_pred_pose.float()))      # (b, v, 4, 4)
                    pts = unproject_depth(depth_t, hq_pred_intrinsics.float(), c2w_t)  # (b, v, H, W, 3)

                    PATCH_SIZE = 14
                    b, v, H, W, _ = pts.shape
                    Ph, Pw = H // PATCH_SIZE, W // PATCH_SIZE
                    n = Ph * Pw  # spatial tokens per view

                    # pool to patch resolution: (b, v, Ph, Pw, 3)
                    pts_patch  = pts.reshape(b, v, Ph, PATCH_SIZE, Pw, PATCH_SIZE, 3).mean(dim=(3, 5))
                    # flatten all views' patches: (b, v*n, 3)
                    pts_flat   = pts_patch.reshape(b, v * n, 3)

                    # reorder view blocks to match pred_map (ref view first)
                    if lq_ref_b_idx is None:
                        lq_ref_b_idx = torch.tensor([0], device=device)  # default to first view as reference if not provided by encoder
                    ref_v      = int(lq_ref_b_idx[0].item())
                    view_order = [ref_v] + [vi for vi in range(v) if vi != ref_v]
                    perm       = torch.cat([torch.arange(vi * n, vi * n + n, device=pts_flat.device) for vi in view_order])
                    pts_flat   = pts_flat[:, perm, :]  # (b, v*n, 3) reordered

                    # pairwise neg L2: (b, v*n, v*n)
                    diff       = pts_flat.unsqueeze(1) - pts_flat.unsqueeze(2)  # (b, v*n, v*n, 3)
                    neg_l2     = -torch.norm(diff, dim=-1)                                    # (b, v*n, v*n)
                    T          = full_cfg.mvrm.loss.attn_align.da3_point_cloud.get('vis_pc_temperature', 1.0)
                    if T == -1:  # hard one-hot assignment (nearest neighbour per view block)
                        print('hard assignment for point cloud')
                        _blocked   = neg_l2.reshape(b, v * n, v, n)
                        geo_target = torch.zeros_like(_blocked).scatter_(-1, _blocked.argmax(dim=-1, keepdim=True), 1.0)
                        geo_target = geo_target.reshape(b, v * n, v * n)
                    else:
                        print('soft assignment for point cloud')
                        geo_target = (neg_l2 / T).softmax(dim=-1)                       # (b, v*n, v*n)

                    # --- Visibility mask (optional) ---
                    # Controls which (query patch, target view) pairs contribute to the loss.
                    # Unmasked geo_target rows that fail visibility are zeroed and renormalized.
                    pc_cfg       = full_cfg.mvrm.loss.attn_align.da3_point_cloud
                    vis_mask_type = pc_cfg.get('visibility_mask', 'none')  # [none, cycle_consistency, reprojection]

                    if vis_mask_type != 'none':
                        vis_mask = torch.zeros(b, v * n, v * n, dtype=torch.bool, device=device)

                        if vis_mask_type == 'cycle_consistency':
                            print(f'Using cycle consistency visibility mask - cycle threshold {pc_cfg.get("vis_pc_cycle_threshold", "None")}')
                            ref_idx = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)  # (b, n)
                            for va in range(v):
                                for vb in range(v):
                                    va_s, va_e = va * n, (va + 1) * n
                                    vb_s, vb_e = vb * n, (vb + 1) * n
                                    if va == vb:
                                        vis_mask[:, va_s:va_e, vb_s:vb_e] = True
                                        continue
                                    corr_ab = geo_target[:, va_s:va_e, vb_s:vb_e]   # (b, n, n)
                                    corr_ba = geo_target[:, vb_s:vb_e, va_s:va_e]   # (b, n, n)
                                    fwd       = corr_ab.argmax(dim=-1)                 # (b, n)
                                    bwd       = corr_ba.argmax(dim=-1)                 # (b, n)
                                    roundtrip = torch.gather(bwd, 1, fwd)              # (b, n)
                                    cycle_thresh = pc_cfg.get('vis_pc_cycle_threshold', 0)
                                    if cycle_thresh == 0:
                                        visible = (roundtrip == ref_idx)
                                    else:
                                        rt_r,  rt_c  = roundtrip // Pw, roundtrip % Pw
                                        ref_r, ref_c = ref_idx   // Pw, ref_idx   % Pw
                                        distance    = torch.max((rt_r - ref_r).abs(), (rt_c - ref_c).abs())
                                        visible = distance <= cycle_thresh
                                    vis_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n)

                        elif vis_mask_type == 'reprojection':
                            print(f'using reprojection visibility mask - depth threshold {pc_cfg.get("reprojection_depth_threshold", "None")}')
                            depth_thr  = pc_cfg.get('reprojection_depth_threshold', 0.1)
                            view_order_t = torch.tensor(view_order, device=device)

                            # Reorder intrinsics, w2c, depth to match view_order
                            K_ord     = hq_pred_intrinsics.float()[:, view_order_t]              # (b, v, 3, 3)
                            w2c_ord   = as_homogeneous(hq_pred_pose.float())[:, view_order_t]    # (b, v, 4, 4) w2c
                            depth_ord = depth_t.squeeze(-1)[:, view_order_t]                     # (b, v, H, W)
                            depth_patch_ord = depth_ord.reshape(
                                b, v, Ph, PATCH_SIZE, Pw, PATCH_SIZE).mean(dim=(3, 5))           # (b, v, Ph, Pw)

                            # pts_flat is already reordered → (b, v, n, 3)
                            pts_v = pts_flat.reshape(b, v, n, 3)

                            for va in range(v):
                                for vb in range(v):
                                    va_s, va_e = va * n, (va + 1) * n
                                    vb_s, vb_e = vb * n, (vb + 1) * n
                                    if va == vb:
                                        vis_mask[:, va_s:va_e, vb_s:vb_e] = True
                                        continue

                                    P_world = pts_v[:, va]                        # (b, n, 3)
                                    R_vb    = w2c_ord[:, vb, :3, :3]             # (b, 3, 3)
                                    t_vb    = w2c_ord[:, vb, :3, 3]              # (b, 3)
                                    P_cam   = torch.bmm(P_world, R_vb.transpose(1, 2)) + t_vb.unsqueeze(1)  # (b, n, 3)

                                    z_proj  = P_cam[:, :, 2]                     # (b, n)
                                    fx = K_ord[:, vb, 0, 0].unsqueeze(1)
                                    fy = K_ord[:, vb, 1, 1].unsqueeze(1)
                                    cx = K_ord[:, vb, 0, 2].unsqueeze(1)
                                    cy = K_ord[:, vb, 1, 2].unsqueeze(1)

                                    u_px   = fx * P_cam[:, :, 0] / z_proj.clamp(min=1e-8) + cx  # (b, n) pixel x
                                    v_px   = fy * P_cam[:, :, 1] / z_proj.clamp(min=1e-8) + cy  # (b, n) pixel y
                                    pi_col = (u_px / PATCH_SIZE).long()          # patch column in vb
                                    pi_row = (v_px / PATCH_SIZE).long()          # patch row in vb

                                    in_bounds = (z_proj > 0) & (pi_row >= 0) & (pi_row < Ph) & (pi_col >= 0) & (pi_col < Pw)

                                    # Depth consistency: compare projected depth vs actual patch depth in vb
                                    pi_row_c = pi_row.clamp(0, Ph - 1)
                                    pi_col_c = pi_col.clamp(0, Pw - 1)
                                    flat_idx = (pi_row_c * Pw + pi_col_c)        # (b, n)
                                    depth_vb_flat = depth_patch_ord[:, vb].reshape(b, -1)        # (b, Ph*Pw)
                                    depth_at_proj = torch.gather(depth_vb_flat, 1, flat_idx)     # (b, n)
                                    rel_err  = (z_proj - depth_at_proj).abs() / (depth_at_proj.abs() + 1e-8)
                                    depth_ok = rel_err < depth_thr

                                    visible  = in_bounds & depth_ok
                                    vis_mask[:, va_s:va_e, vb_s:vb_e] = visible.unsqueeze(-1).expand(-1, -1, n)

                        # Zero out non-visible correspondences and renormalize rows
                        geo_target = geo_target * vis_mask.float()
                        row_sum    = geo_target.sum(dim=-1, keepdim=True)          # (b, v*n, 1)
                        valid_rows = (row_sum.squeeze(-1) > 0)                     # (b, v*n) rows with mass
                        geo_target = geo_target / row_sum.clamp(min=1e-8)

                    # slice CLS tokens out of pred_map: positions 0, n+1, 2*(n+1), ...
                    N_total = pred_map.shape[-1]
                    spatial_mask = torch.ones(N_total, dtype=torch.bool, device=device)
                    for vi in range(v):
                        spatial_mask[vi * (n + 1)] = False
                    pred_map_spatial = pred_map[:, spatial_mask, :][:, :, spatial_mask]  # (1, v*n, v*n)

                    row_mask  = valid_rows if vis_mask_type != 'none' else None
                    attn_loss = cross_entropy_attn(pred_map_spatial, geo_target, row_mask=row_mask)
                    loss += lambda_attn * attn_loss
                    
                                          
                        
            if full_cfg.mvrm.loss.velocity_direction.use:
                # velocity direction regularization
                pred_vel = transport_output["pred"]
                gt_vel   = transport_output["target_velocity"]
                lambda_vel_dir = full_cfg.mvrm.loss.velocity_direction.lambda_coeff
                loss_vel_dir = velocity_direction_loss(pred_vel, gt_vel.detach())
                loss_vel_dir_scaled = lambda_vel_dir * loss_vel_dir
                # loss += loss_vel_dir_scaled
            else:
                loss_vel_dir = torch.tensor(0.0, device=device)
                loss_vel_dir_scaled = torch.tensor(0.0, device=device)
                
                
            # camera pose loss
            if full_cfg.mvrm.loss.camera_pose.use:
                lambda_cam = full_cfg.mvrm.loss.camera_pose.lambda_coeff
                weight_trans=1.0       # weight for translation loss
                weight_rot=1.0         # weight for rotation loss
                weight_focal=0.5       # weight for focal length loss
                # --------------------------------------------------
                # Flatten (B, V, 9) → (B*V, 9)
                # --------------------------------------------------
                B, V, D = hq_pred_pose_enc.shape
                hq_pose_flat = hq_pred_pose_enc.reshape(B * V, D)
                lq_pose_flat = lq_pred_pose_enc.reshape(B * V, D)
                # --------------------------------------------------
                # Compute camera loss
                # --------------------------------------------------
                loss_T, loss_R, loss_FL = camera_loss_single(
                    pred_pose_enc=lq_pose_flat,
                    gt_pose_enc=hq_pose_flat.detach(),   # detach GT branch
                    loss_type="l1"
                )
                # Total camera loss
                loss_cam = weight_trans*loss_T + weight_rot*loss_R + weight_focal*loss_FL
                loss_cam_scaled = lambda_cam * loss_cam
                # Add to total loss
                # loss += loss_cam_scaled
            else:
                loss_cam = torch.tensor(0.0, device=device)
                loss_cam_scaled = torch.tensor(0.0, device=device)

                
            # ---------------------------
            # Backward
            # ---------------------------
            loss.backward()


            if clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    models['ddp_denoiser'].parameters(),
                    clip_grad
                )
                

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            update_ema(
                models['ema_denoiser'],
                models['ddp_denoiser'].module,
                decay=ema_decay,
            )

            optimizer_step += 1

            # ---------------------------
            # Logging
            # ---------------------------
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()

            if rank == 0 and global_train_step % log_interval == 0:

                avg_loss = running_loss / log_interval

                stats = {
                    "train/loss_interval_avg": avg_loss,
                    "train/loss_transport": transport_loss.item(),
                    "train/loss_cam_token": transport_output["cam_tkn_loss"].mean().item(),
                    "train/loss_patch_token": transport_output["patch_tkn_loss"].mean().item(),
                    
                    
                    "train_attn/loss_attn": attn_loss.item(),
                    
                                        
                    "train_cam/loss_cam_T": loss_T.item() if full_cfg.mvrm.loss.camera_pose.use else 0.0,
                    "train_cam/loss_cam_R": loss_R.item() if full_cfg.mvrm.loss.camera_pose.use else 0.0,
                    "train_cam/loss_cam_FL": loss_FL.item() if full_cfg.mvrm.loss.camera_pose.use else 0.0,
                    "train_cam/loss_cam": loss_cam.item() if full_cfg.mvrm.loss.camera_pose.use else 0.0,
                    "train_cam/loss_cam_scaled": loss_cam_scaled.item() if full_cfg.mvrm.loss.camera_pose.use else 0.0,
                    
                    'train_reg/loss_vel_dir': loss_vel_dir.item(),
                    'train_reg/loss_vel_dir_scaled': loss_vel_dir_scaled.item(),
                    
                    "train_etc/lq_drop_prob": full_cfg.training.guidance.lq_drop,
                    "train_etc/lr": optimizer.param_groups[0]["lr"],
                }

                logger.info(
                    f"[Epoch {epoch} | Step {global_train_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )

                if full_cfg.log.tracker.name == 'wandb':
                    wandb_utils.log(stats, step=global_train_step)

                running_loss = 0.0

        
            # ckpt saving
            if rank==0 and global_train_step > 0 and global_train_step % ckpt_step_interval == 0:
                logger.info(f"Saving checkpoint at global_train_step {global_train_step}...")
                ckpt_path = f"{checkpoint_dir}/ep-{global_train_step:07d}.pt" 
                initialize.save_checkpoint(
                    ckpt_path,
                    global_train_step,
                    epoch,
                    models['ddp_denoiser'],
                    models['ema_denoiser'],
                    optimizer,
                    scheduler,
                )                        
            

            # validation
            if rank==0 and (len(full_cfg.data.val.list) != 0) and (training_cfg.vis.val_depth_every > 0)and (global_train_step % training_cfg.vis.val_depth_every) == 0:

                val_featsim_metrics = {}

                val_lq_metrics = {
                    "pose_auc30": [],
                    "pose_auc15": [],
                    "pose_auc05": [],
                    "pose_auc03": [],
                    
                    "depth_abs_rel": [],
                    "depth_sq_rel": [],
                    "depth_rmse": [],
                    "depth_d1": [],
                    "depth_d2": [],
                    "depth_d3": [],
                }
                
                val_res_metrics = {
                    "pose_auc30": [],
                    "pose_auc15": [],
                    "pose_auc05": [],
                    "pose_auc03": [],
                    
                    "depth_abs_rel": [],
                    "depth_sq_rel": [],
                    "depth_rmse": [],
                    "depth_d1": [],
                    "depth_d2": [],
                    "depth_d3": [],
                }

                models['ddp_denoiser'].eval()
                export_feat_layers=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39]
                
                # val loop
                for val_step, val_batch in enumerate(tqdm(val_loader)):
                    
                    logger.info(f'Validating Samples: {val_step+1}/{len(val_loader)}')    
                                    
                    

                    # load val batch 
                    val_frame_id = val_batch['frame_ids']               # b v
                    val_hq_id = val_batch['hq_ids']                     # len(hq_id) = b, len(hq_id[i]) = v
                    # val_gt_depth = val_batch['gt_depths'].to(device)    # b v 1 h w=504
                    val_hq_views = val_batch['hq_views'].to(device)     # b v 3 h w=504
                    val_lq_views = val_batch['lq_views'].to(device)     # b v 3 h w=504
                    logger.info(f"Val sample shape: {val_hq_views.shape}")
                    
                    # save_image(val_batch['lq_views'].squeeze(0), 'val_lq.png', normalize=True)
                    # save_image(val_batch['hq_views'].squeeze(0), 'val_hq.png', normalize=True)
                    
    
                    # apply imagenet normalization
                    val_b, val_v, val_c, val_h, val_w = val_lq_views.shape 
                    # val_hq_views = IMAGENET_NORMALIZE(val_hq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)
                    val_hq_views = IMAGENET_NORMALIZE(val_hq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)
                    val_lq_views = IMAGENET_NORMALIZE(val_lq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)


                    # val - LQ view forward pass
                    with torch.no_grad():
                        val_lq_encoder_out, val_lq_mvrm_out = models['encoder'](
                                                            image=val_lq_views, 
                                                            export_feat_layers=export_feat_layers, 
                                                            mvrm_cfg=full_cfg.mvrm.train, 
                                                            mode='train'
                                                            )
                    val_lq_pred_pose_enc = val_lq_encoder_out.pose_enc 
                    val_lq_pred_pose = val_lq_encoder_out['extrinsics'] # b v 3 4
                    val_lq_ref_b_idx = val_lq_encoder_out.ref_b_idx                    
                    val_lq_depth = val_lq_encoder_out.depth.unsqueeze(2)    # b v 1 h w 
                    # val_lq_encoder_out = processors['encoder_output_processor'](val_lq_encoder_out)
                    # val_lq_pred_depth_np = val_lq_encoder_out.depth
                    # val_lq_pred_depth = torch.from_numpy(val_lq_pred_depth_np).to(device) 
                    # val_lq_latent = val_lq_mvrm_out['extract_feat']      # b v 973 3072
                    val_lq_latent = val_lq_mvrm_out[('extract_feat', full_cfg.mvrm.train.extract_feat_layers[0])]      # b v 973 3072


                    # val_lq_encoder_out['aux']['feat_layer_18'].shape (b v n+1 d)

                    # val - HQ view forward pass
                    with torch.no_grad():
                        val_hq_encoder_out, _ = models['encoder'](
                                                            image=val_hq_views, 
                                                            export_feat_layers=export_feat_layers, 
                                                            mvrm_cfg=None,
                                                            mode=None,
                                                            ref_b_idx = val_lq_ref_b_idx,
                                                            # ref_b_idx = None
                                                            )
                    val_hq_pred_pose_enc = val_hq_encoder_out.pose_enc
                    val_hq_pred_pose = val_hq_encoder_out['extrinsics'] # b v
                    val_hq_depth = val_hq_encoder_out.depth.unsqueeze(2)    # b v 1 h w
                    # val_hq_encoder_out = processors['encoder_output_processor'](val_hq_encoder_out)
                    # val_hq_pred_depth_np = val_hq_encoder_out.depth
                    # val_hq_pred_depth = torch.from_numpy(val_hq_pred_depth_np).to(device) 
                    # val_lq_latent = val_lq_mvrm_out['extract_feat']      # b v 973 3072
                    # val_hq_latent = val_hq_mvrm_out[('extract_feat', full_cfg.mvrm.train.extract_feat_layers[0])]
                    

                    val_noise_generator.manual_seed(global_seed)
                    val_pure_noise = torch.randn(val_lq_latent.shape, generator=val_noise_generator, device=device, dtype=torch.float32)
                    
                                        
                    # lq2hq training w/ different noise levels
                    noise_lvl = full_cfg.mvrm.get('noise_lvl', None)     # typically 0.3
                    
                    
                    if noise_lvl is not None:
                        val_xt = val_pure_noise * noise_lvl + val_lq_latent
                        val_lq_latent_for_kwargs = None
                    
                    else:
                        # lq_latent condition method
                        # For 'addition': bake lq_latent into the initial ODE state (shapes stay d-dim throughout)
                        # For 'concat': keep ODE state as d-dim pure noise; lq_latent is injected inside the drift call
                        if full_cfg.mvrm.lq_latent_cond == 'addition':
                            val_xt = val_pure_noise + val_lq_latent
                            val_lq_latent_for_kwargs = None
                        elif full_cfg.mvrm.lq_latent_cond == 'concat':
                            val_xt = val_pure_noise
                            val_lq_latent_for_kwargs = val_lq_latent


                    val_model_kwargs={
                        'model_img_size': (val_h, val_w),
                        'guidance': full_cfg.mvrm.val.guidance,
                        'lq_latent': val_lq_latent_for_kwargs,
                        'lq_latent_cond': full_cfg.mvrm.lq_latent_cond,
                    }
                    
                    
                    
                    with torch.no_grad():
                        restored_samples = eval_sampler(val_xt, ema_model_fn, **val_model_kwargs)[-1]     # b v n d


                    mvrm_result={}
                    # mvrm_result['restored_latent'] = restored_samples
                    mvrm_result[('restored_latent', full_cfg.mvrm.train.extract_feat_layers[0])] = restored_samples


                    with torch.no_grad():
                        val_encoder_out, val_mvrm_out = models['encoder'](
                                                                    image=val_lq_views, 
                                                                    export_feat_layers=export_feat_layers, 
                                                                    mvrm_cfg=full_cfg.mvrm.val, 
                                                                    mvrm_result=mvrm_result,
                                                                    mode='val'
                                                                    )
                    val_res_depth = val_encoder_out.depth.unsqueeze(2)    # b v 1 h w
                    val_res_pred_pose = val_encoder_out['extrinsics'] # b v 3 4
                    # val_encoder_out = processors['encoder_output_processor'](val_encoder_out)
                    # val_pred_depth_np = val_encoder_out.depth     # num_view h w -> 1 num_view 1 h w           
                    # val_pred_depth = torch.from_numpy(val_pred_depth_np).to(device)


                    pose_setting = 'unposed'
                    data = 'hypersim'
                    scene = f"{val_batch['lq_ids'][0][0]}_step{global_train_step:07}"
                    
                    
                    imgs = val_hq_views 
                    lq_imgs = val_lq_views 
                    
                    hq_depth = val_hq_depth
                    lq_depth = val_lq_depth
                    res_depth = val_res_depth
                    
                    hq_pred_pose = val_hq_pred_pose
                    lq_pred_pose = val_lq_pred_pose
                    res_pred_pose = val_res_pred_pose
                    
                    
                    vis_save_root = os.path.join(experiment_dir, 'pho_vis_results', data, pose_setting)
                    vis_all(
                        vis_save_root=vis_save_root,
                        scene=scene,
                        hq_img=imgs[0],
                        lq_img=lq_imgs[0],
                        hq_depth=hq_depth[0],
                        lq_depth=lq_depth[0],
                        res_depth=res_depth[0],
                    )
                    metric_save_root = os.path.join(experiment_dir, 'pho_metric_results', data, pose_setting)
                    result_metric_all = metric_all(
                                        metric_save_root=metric_save_root,
                                        scene=scene,
                                        poses = (hq_pred_pose[0], lq_pred_pose[0],res_pred_pose[0]),
                                        depths = (hq_depth, lq_depth, res_depth),
                                        return_metric=True
                                    )
                    featsim_log = featsim_all(val_hq_encoder_out, val_lq_encoder_out, val_encoder_out)
                    for layer_idx, layer_dict in featsim_log.items():

                        # initialize per layer
                        if layer_idx not in val_featsim_metrics:
                            val_featsim_metrics[layer_idx] = {
                                "all_hq_lq": [],
                                "all_hq_res": [],
                                "cam_hq_lq": [],
                                "cam_hq_res": [],
                                "patch_hq_lq_mean": [],
                                "patch_hq_res_mean": [],
                            }

                        val_featsim_metrics[layer_idx]["all_hq_lq"].append(
                            layer_dict["all_tokens"]["hq_vs_lq_mean"]
                        )
                        val_featsim_metrics[layer_idx]["all_hq_res"].append(
                            layer_dict["all_tokens"]["hq_vs_res_mean"]
                        )

                        val_featsim_metrics[layer_idx]["cam_hq_lq"].append(
                            layer_dict["camera_token"]["hq_vs_lq_mean"]
                        )
                        val_featsim_metrics[layer_idx]["cam_hq_res"].append(
                            layer_dict["camera_token"]["hq_vs_res_mean"]
                        )

                        val_featsim_metrics[layer_idx]["patch_hq_lq_mean"].append(
                            layer_dict["patch_tokens"]["hq_vs_lq"]["mean"]
                        )
                        val_featsim_metrics[layer_idx]["patch_hq_res_mean"].append(
                            layer_dict["patch_tokens"]["hq_vs_res"]["mean"]
                        )
                    featsim_save_root = os.path.join(experiment_dir, 'pho_featsim_results', data, pose_setting)
                    plot_three_similarity_panels(
                        featsim_log,
                        save_path=f"{featsim_save_root}/{scene}_sim_all_combined.png"
                    )
                    cam_save_root = os.path.join(experiment_dir, 'pho_cam_traj_results', data, pose_setting)
                    plot_cam_trajectory(hq_pred_pose[0], lq_pred_pose[0], res_pred_pose[0], visualize_direction=False, save_path=f"{cam_save_root}/{scene}.png")
            

                    val_lq_metrics['pose_auc30'].append(result_metric_all['pose_lq'].auc30)
                    val_lq_metrics['pose_auc15'].append(result_metric_all['pose_lq'].auc15)
                    val_lq_metrics['pose_auc05'].append(result_metric_all['pose_lq'].auc05)
                    val_lq_metrics['pose_auc03'].append(result_metric_all['pose_lq'].auc03)
                    val_lq_metrics['depth_abs_rel'].append(result_metric_all['depth_lq']['abs_rel'])
                    val_lq_metrics['depth_sq_rel'].append(result_metric_all['depth_lq']['sq_rel'])
                    val_lq_metrics['depth_rmse'].append(result_metric_all['depth_lq']['rmse'])
                    val_lq_metrics['depth_d1'].append(result_metric_all['depth_lq']['d1'])
                    val_lq_metrics['depth_d2'].append(result_metric_all['depth_lq']['d2'])
                    val_lq_metrics['depth_d3'].append(result_metric_all['depth_lq']['d3'])

                    val_res_metrics['pose_auc30'].append(result_metric_all['pose_res'].auc30)
                    val_res_metrics['pose_auc15'].append(result_metric_all['pose_res'].auc15)
                    val_res_metrics['pose_auc05'].append(result_metric_all['pose_res'].auc05)
                    val_res_metrics['pose_auc03'].append(result_metric_all['pose_res'].auc03)
                    val_res_metrics['depth_abs_rel'].append(result_metric_all['depth_res']['abs_rel'])
                    val_res_metrics['depth_sq_rel'].append(result_metric_all['depth_res']['sq_rel'])
                    val_res_metrics['depth_rmse'].append(result_metric_all['depth_res']['rmse'])
                    val_res_metrics['depth_d1'].append(result_metric_all['depth_res']['d1'])
                    val_res_metrics['depth_d2'].append(result_metric_all['depth_res']['d2'])
                    val_res_metrics['depth_d3'].append(result_metric_all['depth_res']['d3'])


                if rank == 0 and full_cfg.log.tracker.name == 'wandb':


                    # flatten featsim logs
                    featsim_wandb_log = {}

                    for layer_idx, metrics in val_featsim_metrics.items():
                        for key, values in metrics.items():
                            featsim_wandb_log[f"val_featsim/layer_{layer_idx}/{key}"] = sum(values) / len(values)



                    wandb_utils.log({

                        # LQ pose metrics
                        "val_metric_lq_pose/auc30": sum(val_lq_metrics['pose_auc30']) / len(val_loader),
                        "val_metric_lq_pose/auc15": sum(val_lq_metrics['pose_auc15']) / len(val_loader),
                        "val_metric_lq_pose/auc05": sum(val_lq_metrics['pose_auc05']) / len(val_loader),
                        "val_metric_lq_pose/auc03": sum(val_lq_metrics['pose_auc03']) / len(val_loader),

                        # LQ depth metrics
                        "val_metric_lq_depth/abs_rel": sum(val_lq_metrics['depth_abs_rel']) / len(val_loader),
                        "val_metric_lq_depth/sq_rel": sum(val_lq_metrics['depth_sq_rel']) / len(val_loader),
                        "val_metric_lq_depth/rmse": sum(val_lq_metrics['depth_rmse']) / len(val_loader),
                        "val_metric_lq_depth/d1": sum(val_lq_metrics['depth_d1']) / len(val_loader),
                        "val_metric_lq_depth/d2": sum(val_lq_metrics['depth_d2']) / len(val_loader),
                        "val_metric_lq_depth/d3": sum(val_lq_metrics['depth_d3']) / len(val_loader),

                        # Res pose metrics
                        "val_metric_res_pose/auc30": sum(val_res_metrics['pose_auc30']) / len(val_loader),
                        "val_metric_res_pose/auc15": sum(val_res_metrics['pose_auc15']) / len(val_loader),
                        "val_metric_res_pose/auc05": sum(val_res_metrics['pose_auc05']) / len(val_loader),
                        "val_metric_res_pose/auc03": sum(val_res_metrics['pose_auc03']) / len(val_loader),

                        # Res depth metrics
                        "val_metric_res_depth/abs_rel": sum(val_res_metrics['depth_abs_rel']) / len(val_loader),
                        "val_metric_res_depth/sq_rel": sum(val_res_metrics['depth_sq_rel']) / len(val_loader),
                        "val_metric_res_depth/rmse": sum(val_res_metrics['depth_rmse']) / len(val_loader),
                        "val_metric_res_depth/d1": sum(val_res_metrics['depth_d1']) / len(val_loader),
                        "val_metric_res_depth/d2": sum(val_res_metrics['depth_d2']) / len(val_loader),
                        "val_metric_res_depth/d3": sum(val_res_metrics['depth_d3']) / len(val_loader),
                        
                        **featsim_wandb_log,

                    }, step=global_train_step)

                                    
                    # breakpoint()
                    # # ------------------------------------------
                    # # VISUALIZE + METRICS (VALIDATION)
                    # # ------------------------------------------
                    # vis_lq_rgbs = []
                    # vis_lq_depths = []
                    # vis_res_depths = []
                    # vis_lq_err_maps = []
                    # vis_res_err_maps = []
                    # # metric accumulators (per batch item)
                    # for v in range(val_v):
                    #     # RGB (unnormalized)
                    #     vis_lq_rgbs.append(tensor_to_uint8_image(val_lq_views[0, v]))
                    #     # Depth predictions
                    #     lq_depth_np  = val_lq_pred_depth_np[v]
                    #     res_depth_np = val_pred_depth_np[v]
                    #     gt_depth_np  = val_gt_depth[0, v, 0].cpu().numpy()
                    #     vis_lq_depths.append(depth_to_colormap(lq_depth_np))
                    #     vis_res_depths.append(depth_to_colormap(res_depth_np))
                    #     # torch
                    #     lq_depth_t  = torch.from_numpy(lq_depth_np).to(device)
                    #     res_depth_t = torch.from_numpy(res_depth_np).to(device)
                    #     gt_depth_t  = torch.from_numpy(gt_depth_np).to(device)
                    #     lq_depth_align = align_scale_median(gt_depth_t, lq_depth_t)
                    #     res_depth_align = align_scale_median(gt_depth_t, res_depth_t)
                    #     # Metrics
                    #     lq_metrics = compute_depth_metrics(gt_depth_t, lq_depth_align)  # (7,)
                    #     res_metrics = compute_depth_metrics(gt_depth_t, res_depth_align)  # (7,)
                    #     lq_metrics_np = np.array(lq_metrics, dtype=np.float64)
                    #     res_metrics_np = np.array(res_metrics, dtype=np.float64)
                    #     if val_lq_metric_sum is None:
                    #         val_lq_metric_sum = lq_metrics_np.copy()
                    #     else:
                    #         val_lq_metric_sum += lq_metrics_np
                    #     val_lq_metric_count += 1
                    #     if val_res_metric_sum is None:
                    #         val_res_metric_sum = res_metrics_np.copy()
                    #     else:
                    #         val_res_metric_sum += res_metrics_np
                    #     val_res_metric_count += 1
                    #     # Error map (numpy)
                    #     lq_err_map = depth_error_to_colormap_thresholded(gt=gt_depth_np, pred=lq_depth_align.cpu().numpy(), thr=0.1)
                    #     res_err_map = depth_error_to_colormap_thresholded(gt=gt_depth_np, pred=res_depth_align.cpu().numpy(), thr=0.1)
                    #     vis_lq_err_maps.append(lq_err_map)
                    #     vis_res_err_maps.append(res_err_map)
                    # # Concatenate views horizontally
                    # vis_lq_rgbs    = np.concatenate(vis_lq_rgbs, axis=1)
                    # vis_lq_depths  = np.concatenate(vis_lq_depths, axis=1)
                    # vis_res_depths    = np.concatenate(vis_res_depths, axis=1)
                    # vis_lq_err_maps   = np.concatenate(vis_lq_err_maps, axis=1)
                    # vis_res_err_maps  = np.concatenate(vis_res_err_maps, axis=1)
                    # # Stack rows vertically
                    # vis_val_all = np.concatenate(
                    #     [
                    #         vis_lq_rgbs[:, :, ::-1],  # RGB → BGR for OpenCV
                    #         vis_lq_depths,
                    #         vis_res_depths,
                    #         vis_lq_err_maps,
                    #         vis_res_err_maps,
                    #     ],
                    #     axis=0,
                    # )
                    # # Save visualization
                    # vis_val_depth_save_dir = f"{experiment_dir}/vis_val_depth"
                    # os.makedirs(vis_val_depth_save_dir, exist_ok=True)
                    # cv2.imwrite(f"{vis_val_depth_save_dir}/{val_hq_id[0][0]}_step{global_train_step:07}.jpg",vis_val_all,)
                logger.info("Validation done.")
                # lq_mean_metrics = val_lq_metric_sum / val_lq_metric_count
                # res_mean_metrics = val_res_metric_sum / val_res_metric_count
                # lq_abs_rel, lq_sq_rel, lq_rmse, lq_rmse_log, lq_d1, lq_d2, lq_d3 = lq_mean_metrics
                # res_abs_rel, res_sq_rel, res_rmse, res_rmse_log, res_d1, res_d2, res_d3 = res_mean_metrics
                # if full_cfg.log.tracker.name == 'wandb':
                #     wandb_utils.log(
                #         {
                #             "val_lq/AbsRel": lq_abs_rel,
                #             "val_lq/SqRel": lq_sq_rel,
                #             "val_lq/RMSE": lq_rmse,
                #             "val_lq/RMSElog": lq_rmse_log,
                #             "val_lq/d1": lq_d1,
                #             "val_lq/d2": lq_d2,
                #             "val_lq/d3": lq_d3,

                #             "val_res/AbsRel": res_abs_rel,
                #             "val_res/SqRel": res_sq_rel,
                #             "val_res/RMSE": res_rmse,
                #             "val_res/RMSElog": res_rmse_log,
                #             "val_res/d1": res_d1,
                #             "val_res/d2": res_d2,
                #             "val_res/d3": res_d3,
                #         },
                #         step=global_train_step,
                #     )
                # logger.info(
                #     f"[VAL @ step {global_train_step}] "
                #     f"[Before MVRM (LQ)] "
                #     f"AbsRel {lq_abs_rel:.3f} | SqRel {lq_sq_rel:.3f} | "
                #     f"RMSE {lq_rmse:.3f} | RMSElog {lq_rmse_log:.3f} | "
                #     f"δ1 {lq_d1:.3f} | δ2 {lq_d2:.3f} | δ3 {lq_d3:.3f}"
                # )
                # logger.info(
                #     f"[VAL @ step {global_train_step}] "
                #     f"[After MVRM (Res))] "
                #     f"AbsRel {res_abs_rel:.3f} | SqRel {res_sq_rel:.3f} | "
                #     f"RMSE {res_rmse:.3f} | RMSElog {res_rmse_log:.3f} | "
                #     f"δ1 {res_d1:.3f} | δ2 {res_d2:.3f} | δ3 {res_d3:.3f}"
                # )
                models['ddp_denoiser'].train()
                del val_lq_encoder_out, val_lq_mvrm_out
                del val_hq_encoder_out
                del val_encoder_out, val_mvrm_out
                # del vis_val_all
                # del vis_lq_rgbs, vis_lq_depths, vis_res_depths
                # del vis_lq_err_maps, vis_res_err_maps
                torch.cuda.empty_cache()
            num_batches += 1
            global_train_step += 1
        
        
        # log epoch stats
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches 
            epoch_stats = {
                "epoch/loss": avg_loss,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if full_cfg.log.tracker.name == 'wandb':
                wandb_utils.log(epoch_stats, step=global_train_step)
    
    
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        initialize.save_checkpoint(
            ckpt_path,
            global_train_step,
            num_epochs,
            models['ddp_denoiser'],
            models['ema_denoiser'],
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()



if __name__ == "__main__":
    main()
