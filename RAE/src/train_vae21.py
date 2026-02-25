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


from torchvision.utils import save_image 
from utils.vis_utils import depth_to_colormap, depth_error_to_colormap_thresholded, tensor_to_uint8_image
import torchvision.transforms as T 

from einops import rearrange
from RAE.src import initialize
from motionblur.motionblur import Kernel 


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
    scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    
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
        num_params = sum(p.numel() for p in models['vae'].parameters())
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


    # IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
    

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
            # train_hq_views = IMAGENET_NORMALIZE(train_hq_views.view(train_b*train_v, train_c, train_h, train_w)).view(train_b, train_v, train_c, train_h, train_w)
            # train_lq_views = IMAGENET_NORMALIZE(train_lq_views.view(train_b*train_v, train_c, train_h, train_w)).view(train_b, train_v, train_c, train_h, train_w)


            train_hq_views = rearrange(train_hq_views, 'b v c h w -> (b v) c h w')  
            train_lq_views = rearrange(train_lq_views, 'b v c h w -> (b v) c h w')
                        

            # hq vae encoding
            train_hq_views = train_hq_views * 2.0 - 1.0   # (b v) c h w
            hq_latents = models['vae'].encode(train_hq_views).latent_dist.sample()  # (b v) c h w 
            # hq_latent = (hq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor    # (b v) c h w 
            hq_latent = hq_latents * models['vae'].config.scaling_factor    # (b v) c h w 

            
            # lq vae encoding
            train_lq_views = train_lq_views * 2.0 - 1.0   # (b v) c h w 
            lq_latents = models['vae'].encode(train_lq_views).latent_dist.sample()  # (b v) c h w 
            # lq_latent = (lq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor   # (b v) c h w 
            lq_latent = lq_latents * models['vae'].config.scaling_factor   # (b v) c h w 


            hq_latent = rearrange(hq_latent, '(b v) d h w -> b v d h w', b=train_b, v=train_v)
            lq_latent = rearrange(lq_latent, '(b v) d h w -> b v d h w', b=train_b, v=train_v)
            

            # compute loss (per microbatch)
            transport_output = transport.training_losses_mvrm_VAE(
                model=models['ddp_denoiser'],
                x1=hq_latent,
                xcond=lq_latent,
                cfg=full_cfg
            )
            
            loss = transport_output["loss"].mean()


            if rank == 0 and training_cfg.vis.train_depth_every > 0 and global_train_step % training_cfg.vis.train_depth_every == 0:
                # Pick first batch
                B_vis = 1
                train_hq_vis = train_hq_views[:B_vis * train_v]  # (V, 3, H, W)
                train_lq_vis = train_lq_views[:B_vis * train_v]  # (V, 3, H, W)
                # Convert from [-1,1] → [0,1] if needed
                train_hq_vis = ((train_hq_vis + 1) / 2).clamp(0, 1)
                train_lq_vis = ((train_lq_vis + 1) / 2).clamp(0, 1)
                # -------------------------
                # LQ restored from model
                # -------------------------
                with torch.no_grad():
                    vis_lat = transport_output['pred']  # model output
                    B, V, D, H, W = vis_lat.shape
                    vis_lat = rearrange(vis_lat, 'b v d h w -> (b v) d h w')
                    # undo scaling
                    vis_lat = vis_lat / models['vae'].config.scaling_factor
                    # vis_lat = vis_lat + models['vae'].config.shift_factor
                    # decode
                    decoded = models['vae'].decode(vis_lat).sample
                    vis_lq_restored = ((decoded + 1) / 2).clamp(0, 1)
                    vis_lq_restored = vis_lq_restored[:B_vis * V]  # only first batch
                # -------------------------
                # Stack vertically: HQ, LQ, LQ restored
                # -------------------------
                # vis_hq_cat = rearrange(train_hq_vis, 'v c h w -> c (v h) w')
                # vis_lq_cat = rearrange(train_lq_vis, 'v c h w -> c (v h) w')
                # vis_restored_cat = rearrange(vis_lq_restored, 'v c h w -> c (v h) w')

                # Concatenate all three vertically
                final_vis = torch.cat([train_hq_vis, train_lq_vis, vis_lq_restored], dim=2)  # (v, 3, 3*H, W)
                # save
                vis_train_depth_save_dir = f"{experiment_dir}/vis_train_depth"
                os.makedirs(vis_train_depth_save_dir, exist_ok=True)
                # vis_id = "-".join(train_hq_id[0])
                save_image(final_vis, f"{vis_train_depth_save_dir}/step{global_train_step:07}.jpg")

                            
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
                    "train/lr": optimizer.param_groups[0]["lr"],
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
                
                
                val_lq_metric_sum = None
                val_res_metric_sum = None
                val_lq_metric_count = 0
                val_res_metric_count = 0
                models['ddp_denoiser'].eval()
                # val loop
                for val_step, val_batch in enumerate(tqdm(val_loader)):
                    
                    logger.info(f'Validating Samples: {val_step+1}/{len(val_loader)}')    
                                    
                    # load val batch 
                    # val_frame_id = val_batch['frame_ids']               # b v
                    val_hq_id = val_batch['hq_ids']                     # len(hq_id) = b, len(hq_id[i]) = v
                    # val_gt_depth = val_batch['gt_depths'].to(device)    # b v 1 h w=504
                    val_hq_views = val_batch['hq_views'].to(device)     # b v 3 h w=504 [0,1]
                    val_lq_views = val_batch['lq_views'].to(device)     # b v 3 h w=504 [0,1]
                    logger.info(f"Val sample shape: {val_hq_views.shape}")
                    
                    # apply imagenet normalization
                    val_b, val_v, val_c, val_h, val_w = val_lq_views.shape 
                    # val_hq_views = IMAGENET_NORMALIZE(val_hq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)
                    # val_lq_views = IMAGENET_NORMALIZE(val_lq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)


                    val_hq_views = rearrange(val_hq_views, 'b v c h w -> (b v) c h w')  
                    val_lq_views = rearrange(val_lq_views, 'b v c h w -> (b v) c h w')


                    # lq vae encoding
                    val_lq_views = val_lq_views * 2.0 - 1.0   # (b v) c h w 
                    val_lq_latents = models['vae'].encode(val_lq_views).latent_dist.sample()  # (b v) c h w 
                    # val_lq_latent = (val_lq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor   # (b v) c h w 
                    val_lq_latent = val_lq_latents  * models['vae'].config.scaling_factor   # (b v) c h w 
                    val_lq_latent = rearrange(val_lq_latent, '(b v) d h w -> b v d h w', b=val_b, v=val_v)
      

                    val_noise_generator.manual_seed(global_seed)
                    val_pure_noise = torch.randn(val_lq_latent.shape, generator=val_noise_generator, device=device, dtype=torch.float32)

                    
                    # lq_latent condition method
                    # val_pure_noise = pure_noise
                    if full_cfg.mvrm.lq_latent_cond == 'addition':
                        val_xt = val_pure_noise + val_lq_latent
                    elif full_cfg.mvrm.lq_latent_cond == 'concat':
                        val_xt = torch.concat([val_pure_noise, val_lq_latent], dim=1)
                    
                    
                    val_model_kwargs={
                        # 'model_img_size': (val_h, val_w)
                    }
                    
                    with torch.no_grad():
                        restored_samples = eval_sampler(val_xt, ema_model_fn, **val_model_kwargs)[-1]     # b v n d


                    # vae decode
                    with torch.no_grad():
                        if full_cfg.mvrm.lq_latent_cond == 'concat':
                            # concat was along channel dim
                            D_total = restored_samples.shape[2]
                            D_latent = D_total // 2
                            restored_latent = restored_samples[:, :, :D_latent]
                        else:
                            restored_latent = restored_samples
                        B, V, D, H, W = restored_latent.shape
                        # flatten batch + views
                        restored_latent = rearrange(restored_latent, 'b v d h w -> (b v) d h w')
                        # ---- undo scaling ----
                        restored_latent = restored_latent / models['vae'].config.scaling_factor
                        # restored_latent = restored_latent + models['vae'].config.shift_factor
                        # decode
                        decoded = models['vae'].decode(restored_latent).sample  # (B*V, 3, H_img, W_img)
                        # convert to [0,1]
                        restored_images = ((decoded + 1) / 2).clamp(0, 1)
                        # reshape back
                        restored_images = rearrange(restored_images, '(b v) c h w -> b v c h w', b=B, v=V)
                    # Restore original shape`s (currently flattened)
                    val_hq_views_vis = rearrange(val_hq_views, '(b v) c h w -> b v c h w', b=val_b, v=val_v)
                    val_lq_views_vis = rearrange(val_lq_views, '(b v) c h w -> b v c h w', b=val_b, v=val_v)
                    # Convert from [-1,1] → [0,1]
                    val_hq_views_vis = val_hq_views_vis.clamp(0, 1)
                    val_lq_views_vis = ((val_lq_views_vis + 1) / 2).clamp(0, 1) 
                    # Take first batch
                    hq = val_hq_views_vis[0]          # (1, V, 3, H, W)
                    lq = val_lq_views_vis[0]          # (1, V, 3, H, W)
                    restored = restored_images[0]     # (1, V, 3, H, W)
                    # Stack HQ / LQ / Restored vertically
                    final_vis = torch.cat([hq, lq, restored], dim=2)
                    # Save
                    vis_val_depth_save_dir = f"{experiment_dir}/vis_val_depth"
                    os.makedirs(vis_val_depth_save_dir, exist_ok=True)
                    save_image(final_vis, f"{vis_val_depth_save_dir}/{val_hq_id[0][0]}_step{global_train_step:07}.jpg")
                    








                #     with torch.no_grad():
                #         val_encoder_out, val_mvrm_out = models['encoder'](
                #                                                     image=restored_images, 
                #                                                     export_feat_layers=[], 
                #                                                     mvrm_cfg=full_cfg.mvrm.train, 
                #                                                     mode='train'
                #                                                     )
                #     val_encoder_out = processors['encoder_output_processor'](val_encoder_out)
                #     val_pred_depth_np = val_encoder_out.depth   # num_view h w
                #     val_pred_depth = torch.from_numpy(val_pred_depth_np).to(device)
                    
                                    
                #     # ------------------------------------------
                #     # VISUALIZE + METRICS (VALIDATION)
                #     # ------------------------------------------
                #     vis_lq_rgbs = []
                #     vis_lq_depths = []
                #     vis_res_depths = []
                #     vis_lq_err_maps = []
                #     vis_res_err_maps = []
                #     # metric accumulators (per batch item)
                #     for v in range(val_v):
                #         # RGB (unnormalized)
                #         vis_lq_rgbs.append(tensor_to_uint8_image(val_lq_views[0, v]))
                #         # Depth predictions
                #         lq_depth_np  = val_lq_pred_depth_np[v]
                #         res_depth_np = val_pred_depth_np[v]
                #         gt_depth_np  = val_gt_depth[0, v, 0].cpu().numpy()
                #         vis_lq_depths.append(depth_to_colormap(lq_depth_np))
                #         vis_res_depths.append(depth_to_colormap(res_depth_np))
                #         # torch
                #         lq_depth_t  = torch.from_numpy(lq_depth_np).to(device)
                #         res_depth_t = torch.from_numpy(res_depth_np).to(device)
                #         gt_depth_t  = torch.from_numpy(gt_depth_np).to(device)
                #         lq_depth_align = align_scale_median(gt_depth_t, lq_depth_t)
                #         res_depth_align = align_scale_median(gt_depth_t, res_depth_t)
                #         # Metrics
                #         lq_metrics = compute_depth_metrics(gt_depth_t, lq_depth_align)  # (7,)
                #         res_metrics = compute_depth_metrics(gt_depth_t, res_depth_align)  # (7,)
                #         lq_metrics_np = np.array(lq_metrics, dtype=np.float64)
                #         res_metrics_np = np.array(res_metrics, dtype=np.float64)
                #         if val_lq_metric_sum is None:
                #             val_lq_metric_sum = lq_metrics_np.copy()
                #         else:
                #             val_lq_metric_sum += lq_metrics_np
                #         val_lq_metric_count += 1
                #         if val_res_metric_sum is None:
                #             val_res_metric_sum = res_metrics_np.copy()
                #         else:
                #             val_res_metric_sum += res_metrics_np
                #         val_res_metric_count += 1
                #         # Error map (numpy)
                #         lq_err_map = depth_error_to_colormap_thresholded(gt=gt_depth_np, pred=lq_depth_align.cpu().numpy(), thr=0.1)
                #         res_err_map = depth_error_to_colormap_thresholded(gt=gt_depth_np, pred=res_depth_align.cpu().numpy(), thr=0.1)
                #         vis_lq_err_maps.append(lq_err_map)
                #         vis_res_err_maps.append(res_err_map)
                #     # Concatenate views horizontally
                #     vis_lq_rgbs    = np.concatenate(vis_lq_rgbs, axis=1)
                #     vis_lq_depths  = np.concatenate(vis_lq_depths, axis=1)
                #     vis_res_depths    = np.concatenate(vis_res_depths, axis=1)
                #     vis_lq_err_maps   = np.concatenate(vis_lq_err_maps, axis=1)
                #     vis_res_err_maps  = np.concatenate(vis_res_err_maps, axis=1)
                #     # Stack rows vertically
                #     vis_val_all = np.concatenate(
                #         [
                #             vis_lq_rgbs[:, :, ::-1],  # RGB → BGR for OpenCV
                #             vis_lq_depths,
                #             vis_res_depths,
                #             vis_lq_err_maps,
                #             vis_res_err_maps,
                #         ],
                #         axis=0,
                #     )
                #     # Save visualization
                #     vis_val_depth_save_dir = f"{experiment_dir}/vis_val_depth"
                #     os.makedirs(vis_val_depth_save_dir, exist_ok=True)
                #     cv2.imwrite(f"{vis_val_depth_save_dir}/{val_hq_id[0][0]}_step{global_train_step:07}.jpg",vis_val_all,)
                # logger.info("Validation done.")
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
                # models['ddp_denoiser'].train()
                # breakpoint()
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
