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


from torchvision.utils import save_image 
from utils.vis_utils import depth_to_colormap, depth_error_to_colormap_thresholded
import torchvision.transforms as T 

from einops import rearrange


from RAE.src.initialize import (save_checkpoint, load_checkpoint,
                         load_train_data, load_val_data, 
                         load_model,
                         load_sampler,)


from motionblur.motionblur import Kernel 


# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)



def tensor_stats(x: torch.Tensor):
    x_detached = x.detach()
    finite = torch.isfinite(x_detached)
    return {
        "min": x_detached[finite].min().item() if finite.any() else float("nan"),
        "max": x_detached[finite].max().item() if finite.any() else float("nan"),
        "mean": x_detached[finite].mean().item() if finite.any() else float("nan"),
        "std": x_detached[finite].std().item() if finite.any() else float("nan"),
        "norm": x_detached.norm().item(),
        "nan_frac": (~torch.isfinite(x_detached)).float().mean().item(),
        "dtype": str(x_detached.dtype),
    }


def has_nan_or_inf(x: torch.Tensor):
    return not torch.isfinite(x).all()






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    args = parser.parse_args()
    return args



def main():
    
    # NAN DEBUG
    torch.autograd.set_detect_anomaly(True)
    
    
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
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    # checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ckpt_step_interval = int(training_cfg.get('ckpt_step_interval', 25000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    global_seed = int(training_cfg.get("global_seed", 0))
    

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    
    
    # load encoder and denoiser 
    models, processors = load_model(full_cfg, rank, device)
    

    # load training and validation data 
    train_loader, train_sampler = load_train_data(full_cfg, micro_batch_size, rank, world_size)
    loader_batches = len(train_loader)
    steps_per_epoch = math.ceil(loader_batches / grad_accum_steps)
    
    
    val_loader, val_sampler = load_val_data(full_cfg, 1, rank, world_size)


    # load optimizer
    optimizer, optim_msg = build_optimizer([p for p in models['denoiser'].parameters() if p.requires_grad], training_cfg)


    # load scheduler 
    scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    
    # load Transport 
    transport = create_transport(**full_cfg.transport.params, time_dist_shift=time_dist_shift,)
    transport_sampler = Sampler(transport)


    # load sampler 
    eval_sampler = load_sampler(full_cfg, transport_sampler)


    
    # make noise latent 
    # pure_noise = torch.randn(micro_batch_size, *full_cfg.misc.latent_size, device=device, dtype=torch.float32) # always use float for noise sampling
    
    val_noise_generator = torch.Generator(device=device)
    val_noise_generator.manual_seed(global_seed)  # any fixed seed you like

    ema_model_fn = models['ema_denoiser'].forward

    
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
            start_epoch, global_train_step = load_checkpoint(
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
        logger.info(f"Dataset contains {len(train_loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")



    IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)

    dist.barrier() 
    for epoch in range(start_epoch, num_epochs):
        models['ddp_denoiser'].train()
        train_sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_count = 0


        # train loop
        for train_step, batch in enumerate(train_loader):

                    
            # load batch data
            train_frame_id = batch['frame_ids']               # b v
            train_hq_id = batch['hq_ids']                     # len(hq_id) = b, len(hq_id[i]) = v
            train_gt_depth = batch['gt_depths'].to(device)    # b v 1 378 504
            train_hq_views = batch['hq_views'].to(device)     # b v 3 378 504
            train_lq_views = batch['lq_views'].to(device)     # b v 3 378 504
            
            
            print(train_hq_views.shape)


            # apply imagenet normalization
            b, v, c, h, w = train_hq_views.shape 
            train_hq_views = IMAGENET_NORMALIZE(train_hq_views.view(b*v, c, h, w)).view(b, v, c, h, w)
            train_lq_views = IMAGENET_NORMALIZE(train_lq_views.view(b*v, c, h, w)).view(b, v, c, h, w)


            # hq forward pass
            with torch.no_grad():
                hq_encoder_out, hq_mvrm_out = models['encoder'](
                                                    image=train_hq_views, 
                                                    export_feat_layers=[], 
                                                    mvrm_cfg=full_cfg.mvrm.train, 
                                                    mode='train'
                                                    )
            hq_encoder_out = processors['encoder_output_processor'](hq_encoder_out)
            train_hq_pred_depth_np = hq_encoder_out.depth                  # b v 378 504
            train_hq_pred_depth = torch.from_numpy(train_hq_pred_depth_np).to(device) 
            hq_latent = hq_mvrm_out['extract_feat']
        
        

            # lq view forward pass
            with torch.no_grad():
                lq_encoder_out, lq_mvrm_out = models['encoder'](
                                                    image=train_lq_views, 
                                                    export_feat_layers=[], 
                                                    mvrm_cfg=full_cfg.mvrm.train, 
                                                    mode='train'
                                                    )
            lq_encoder_out = processors['encoder_output_processor'](lq_encoder_out)
            train_lq_pred_depth_np = lq_encoder_out.depth                  # b v 378 504
            train_lq_pred_depth = torch.from_numpy(train_lq_pred_depth_np).to(device) 
            lq_latent = lq_mvrm_out['extract_feat']      # b v 973 3072
            assert lq_latent.shape == hq_latent.shape 
            

            # save_image(batch['hq_views'].view(-1,3,h,w), 'img_hq.jpg')
            # save_image(batch['lq_views'].view(-1,3,h,w), 'img_lq.jpg')
            # save_image(train_hq_pred_depth.view(-1,1,h,w), 'img_depth_hq.jpg')
            # save_image(train_lq_pred_depth.view(-1,1,h,w), 'img_depth_lq.jpg')
            
           
            # compute loss (per microbatch)
            transport_output = transport.training_losses_mvrm(
                model=models['ddp_denoiser'],
                x1=hq_latent,
                xcond=lq_latent,
                model_img_size=(h,w),
                cfg=full_cfg
            )

            loss_raw = transport_output["loss"].mean()
            
            
            # NAN DEBUG LOSS
            if rank == 0 and global_train_step % log_interval == 0:
                wandb_utils.log(
                    {
                        "debug/loss_raw": loss_raw.item(),
                        "debug/loss_isfinite": float(torch.isfinite(loss_raw)),
                    },
                    step=global_train_step,
                )
                            
    
    
            loss_scaled = loss_raw / grad_accum_steps
            # print(f"global step: {global_train_step}, lq_id: {batch['lq_ids']} hq_id: {batch['hq_ids']}")
            
            
            if rank == 0 and global_train_step % log_interval == 0:
                wandb_utils.log(
                    {"debug/loss_scaled": loss_scaled.item()},
                    step=global_train_step,
                )
                

            
            # compute gradients
            if (global_train_step + 1) % grad_accum_steps != 0:
                with models['ddp_denoiser'].no_sync():
                    loss_scaled.backward()
            else:
                loss_scaled.backward()


            accum_loss += loss_raw.item()
            accum_count += 1


            if (global_train_step + 1) % grad_accum_steps == 0:
                loss_accum_avg = accum_loss / accum_count  # == mean over grad_accum_steps
                if rank == 0:
                    wandb_utils.log(
                        {
                            "train/loss_accum_avg": loss_accum_avg,
                        },
                        step=global_train_step,
                    )
                # reset accumulators
                accum_loss = 0.0
                accum_count = 0


            
            # optimizer update
            if (global_train_step + 1) % grad_accum_steps == 0:
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        models['ddp_denoiser'].parameters(), clip_grad
                    )

                if rank == 0:
                    total_norm = torch.norm(
                        torch.stack([
                            p.grad.norm()
                            for p in models['ddp_denoiser'].parameters()
                            if p.grad is not None
                        ])
                    )
                    wandb_utils.log(
                        {"train/grad_norm": total_norm.item()},
                        step=global_train_step,
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
            
                            
            running_loss += loss_raw.item()
            epoch_metrics['loss'] += loss_raw.detach()


            if rank == 0 and log_interval > 0 and global_train_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                stats = {
                    "train/loss_interval_avg": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_train_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                wandb_utils.log(stats, step=global_train_step)
                running_loss = 0.0
                        
            
            

            if rank==0 and full_cfg.training.sample and global_train_step % sample_every == 0:
                
                logger.info(f'Num Validation Samples: {len(val_loader.dataset)}')
                
                # val loop
                for val_step, val_batch in enumerate(val_loader):
                                    
                    
                    # load val batch 
                    val_frame_id = val_batch['frame_ids']               # b v
                    val_hq_id = val_batch['hq_ids']                     # len(hq_id) = b, len(hq_id[i]) = v
                    val_gt_depth = val_batch['gt_depths'].to(device)    # b v 1 h w=504
                    val_hq_views = val_batch['hq_views'].to(device)     # b v 3 h w=504
                    val_lq_views = val_batch['lq_views'].to(device)     # b v 3 h w=504
                    
                    # from torchvision.utils import save_image 
                    # save_image(val_hq_views.squeeze(), 'tmp_hq.jpg')
                    # save_image(val_lq_views.squeeze(), 'tmp_lq.jpg')
                    # save_image(val_gt_depth.squeeze(0,1), 'tmp_depth.jpg', normalize=True)

                    
                    
                    # apply imagenet normalization
                    val_b, val_v, val_c, val_h, val_w = val_lq_views.shape 
                    # val_hq_views = IMAGENET_NORMALIZE(val_hq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)
                    val_lq_views = IMAGENET_NORMALIZE(val_lq_views.view(val_b*val_v, val_c, val_h, val_w)).view(val_b, val_v, val_c, val_h, val_w)


                    # val - lq view forward pass
                    with torch.no_grad():
                        val_lq_encoder_out, val_lq_mvrm_out = models['encoder'](
                                                            image=val_lq_views, 
                                                            export_feat_layers=[], 
                                                            mvrm_cfg=full_cfg.mvrm.train, 
                                                            mode='train'
                                                            )
                    val_lq_encoder_out = processors['encoder_output_processor'](val_lq_encoder_out)
                    val_lq_pred_depth_np = val_lq_encoder_out.depth                  # b v 378 504
                    val_lq_pred_depth = torch.from_numpy(val_lq_pred_depth_np).to(device) 
                    val_lq_latent = val_lq_mvrm_out['extract_feat']      # b v 973 3072


                    val_noise_generator.manual_seed(global_seed)
                    val_pure_noise = torch.randn(val_lq_latent.shape, generator=val_noise_generator, device=device, dtype=torch.float32)

                            
                    # lq_latent condition method
                    # val_pure_noise = pure_noise
                    if full_cfg.mvrm.lq_latent_cond == 'addition':
                        val_xt = val_pure_noise + val_lq_latent
                    elif full_cfg.mvrm.lq_latent_cond == 'concat':
                        val_xt = torch.concat([val_pure_noise, val_lq_latent], dim=1)
                    
                    
                    val_model_kwargs={
                        'model_img_size': (val_h, val_w)
                    }
                    
                    
                    models['ddp_denoiser'].eval()
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        restored_samples = eval_sampler(val_xt, ema_model_fn, **val_model_kwargs)[-1]     # b v n d
                    restored_samples.float()


                    mvrm_result={}
                    mvrm_result['restored_latent'] = restored_samples


                    with torch.no_grad():
                        val_encoder_out, val_mvrm_out = models['encoder'](
                                                                    image=val_lq_views, 
                                                                    export_feat_layers=[], 
                                                                    mvrm_cfg=full_cfg.mvrm.val, 
                                                                    mvrm_result=mvrm_result,
                                                                    mode='val'
                                                                    )
                    val_encoder_out = processors['encoder_output_processor'](val_encoder_out)
                    val_pred_depth_np = val_encoder_out.depth                            # b v 378 504
                    val_pred_depth = torch.from_numpy(val_pred_depth_np).to(device)
                    

                    # VIS DEPTH 
                    # gt_depth_np: b v 378 504
                    # model_input: b v 3 378 504 
                    # train_lq_pred_depth_np: b*v 378 504
                    # val_pred_depth_np: b*v 378 504
                    
                    
                    np_model_input = val_lq_views[0,0]   # # 378 504 3  
                    np_model_input = (np_model_input - np_model_input.min()) / (np_model_input.max() - np_model_input.min()) * 255.0
                    np_model_input = np_model_input.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]     
                    np_gt_depth = val_gt_depth[0,0,0].detach().cpu().numpy()    # 378 504
                    np_lq_depth = val_lq_pred_depth_np[0]                 # 378 504
                    np_restored_depth = val_pred_depth_np[0]                     # 378 504

                    vis_gt_depth = depth_to_colormap(np_gt_depth)               # 378 504 3
                    vis_lq_depth = depth_to_colormap(np_lq_depth)         # 378 504 3
                    vis_restored_depth = depth_to_colormap(np_restored_depth)             # 378 504 3
                    
                    vis_err_lq_depth = depth_error_to_colormap_thresholded(np_gt_depth, np_lq_depth, thr=0.1)
                    vis_err_restored_depth = depth_error_to_colormap_thresholded(np_gt_depth, np_restored_depth, thr=0.1)
                    
                    vis_depth_cat = np.concatenate(
                        [np_model_input, vis_lq_depth, vis_restored_depth, vis_gt_depth, vis_err_lq_depth, vis_err_restored_depth, ],
                        axis=1
                    )

                    # Output path
                    vis_depth_save_dir = f'{experiment_dir}/vis_depth'
                    os.makedirs(vis_depth_save_dir, exist_ok=True)
                    cv2.imwrite(f'{vis_depth_save_dir}/{val_hq_id[0][0]}_step{global_train_step:07}.jpg', vis_depth_cat)               
                    # dist.barrier()
                logger.info("Validation done.")
                models['ddp_denoiser'].train()


            # ckpt saving
            # if rank==0 and optimizer_step > 0 and optimizer_step % ckpt_step_interval == 0:
            if rank==0 and global_train_step > 0 and global_train_step % ckpt_step_interval == 0:
                logger.info(f"Saving checkpoint at global_train_step {global_train_step}...")
                ckpt_path = f"{checkpoint_dir}/ep-{global_train_step:07d}.pt" 
                save_checkpoint(
                    ckpt_path,
                    global_train_step,
                    epoch,
                    models['ddp_denoiser'],
                    models['ema_denoiser'],
                    optimizer,
                    scheduler,
                )
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
        save_checkpoint(
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
