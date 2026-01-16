# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
from collections import defaultdict, OrderedDict
import cv2 
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from pathlib import Path
import math
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf


##### model imports
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *

##### Eval utils
from eval import evaluate_generation_distributed

from torchvision.utils import save_image 
from utils.vis_utils import depth_to_colormap, depth_error_to_colormap_thresholded
import torchvision.transforms as T 

from einops import rearrange


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for stage1_enc.encode and model.forward).")
    args = parser.parse_args()
    return args


def main():
    """Trains a new SiT model using config-driven hyperparameters."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    (
        stage1_cfg,
        stage2_cfg,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        eval_config
    ) = parse_configs(full_cfg)
    

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
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
    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    # checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ckpt_step_interval = int(training_cfg.get('ckpt_step_interval', 25000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    global_seed = int(training_cfg.get("global_seed", 0))
    
    if eval_config:
        """
        FID online evaluation setup
        """
        do_eval = True
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = eval_config.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_data = eval_config.get("data_path", None)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
    else:
        do_eval = False
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_fp16 = full_cfg.training.precision == "fp16"
    use_bf16 = full_cfg.training.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    autocast_enabled = use_fp16 or use_bf16
    autocast_kwargs = dict(dtype=autocast_dtype, enabled=autocast_enabled)
    scaler = GradScaler(enabled=use_fp16)
    
    
    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))
    
    experiment_dir, checkpoint_dir, vis_recon_dir, logger = configure_experiment_dirs(full_cfg, rank)


    # load stage1 encoder 
    if full_cfg.stage_1.model == 'rae':
        from stage1 import RAE
        stage1_enc: RAE = instantiate_from_config(stage1_cfg).to(device)
        stage1_enc.eval()
        if args.compile:
            try:
                stage1_enc.encode = torch.compile(stage1_enc.encode)
            except:
                print('RAE ENCODE compile meets error, falling back to no compile')
            try:
                model.forward = torch.compile(model.forward)
            except:
                print('MODEL FORWARD compile meets error, falling back to no compile')
        # else:
        #     raise NotImplementedError('ARGS>COMPILE')
        else:
            print("Running without torch.compile")
            
    elif full_cfg.stage_1.model == 'da3':
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.io.output_processor import OutputProcessor
        stage1_input_processor = InputProcessor()
        stage1_output_processor = OutputProcessor()
        stage1_enc = DepthAnything3.from_pretrained(full_cfg.stage_1.da3.ckpt).to(device)
        stage1_enc.eval()
        
    elif full_cfg.stage_1.model == 'vggt':
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        stage1_enc = VGGT.from_pretrained(full_cfg.stage_1.vggt.ckpt).to(device)
        stage1_enc.eval()
    
    
    
    # rae stage2 model 
    model: Stage2ModelProtocol = instantiate_from_config(stage2_cfg).to(device)         
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True) # train stage2 model
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    # ddp_model = torch.compile(ddp_model) # fix shape compile, see if it works
    model = ddp_model.module
    ddp_model.train()
    # no need to put RAE into DDP since it's frozen
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(full_cfg)
    
    train_loader, train_sampler = prepare_dataloader(full_cfg, micro_batch_size, num_workers, rank, world_size)
    
    # if do_eval:
    #     eval_dataset = ImageFolder(
    #         str(eval_data),
    #         transform=transforms.Compose([
    #             transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #             transforms.ToTensor(),
    #         ])
    #     )
    #     logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
    
    
    loader_batches = len(train_loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    #### Transport init
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")
    
    
    # breakpoint()
    ### Guidance Init
    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward
            
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    use_guidance = guidance_scale > 1.0
    # make noise latent 
    pure_noise = torch.randn(micro_batch_size, *latent_size, device=device, dtype=torch.float32) # always use float for noise sampling
    # if use_guidance:
    #     pure_noise = torch.cat([pure_noise, pure_noise], dim=0)
    #     y_null = torch.full((n,), null_label, device=device)
    #     ys = torch.cat([ys, y_null], dim=0)
    #     sample_model_kwargs = dict(
    #         cfg_scale=guidance_scale,
    #         cfg_interval=(t_min, t_max),
    #     )
    #     if guidance_method == "autoguidance":
    #         if guid_model_forward is None:
    #             raise RuntimeError("Guidance model forward is not initialized.")
    #         sample_model_kwargs["additional_model_forward"] = guid_model_forward
    #         ema_model_fn = ema_model.forward_with_autoguidance
    #         model_fn = model.forward_with_autoguidance
    #     else:
    #         ema_model_fn = ema_model.forward_with_cfg
    #         model_fn = model.forward_with_cfg
    # else:
    #     sample_model_kwargs = dict()
    #     ema_model_fn = ema_model.forward
    #     model_fn = model.forward

    sample_model_kwargs = dict()
    ema_model_fn = ema_model.forward
    model_fn = model.forward
    

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in stage1_enc.parameters())
        logger.info(f"Stage-1 RAE parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model parameters: {num_params/1e6:.2f}M")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(train_loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],)

    dist.barrier() 
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        # if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
        # if global_step > 0 and global_step % ckpt_step_interval == 0 and rank == 0:
        #     logger.info(f"Saving checkpoint at epoch {epoch}...")
        #     ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
        #     save_checkpoint(
        #         ckpt_path,
        #         global_step,
        #         epoch,
        #         ddp_model,
        #         ema_model,
        #         optimizer,
        #         scheduler,
        #     )
            
        for train_step, batch in enumerate(train_loader):

            if 'imagenet' in full_cfg.data.train.list:
                (images, labels) = batch            
                images = images.to(device)  # b 3 512 512 [0,1]
                labels = labels.to(device)
                model_kwargs = dict(y=labels)
                # latent encoding
                with torch.no_grad():
                    with autocast(**autocast_kwargs):
                        z = stage1_enc.encode(images)          # b 768 32 32 
                optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    transport_output = transport.training_losses(ddp_model, z, model_kwargs)
                    loss = transport_output["loss"].mean()
            else:
                frame_id = batch['frame_ids']                           # b v
                hq_id = batch['hq_ids']                                 # len(hq_id) = b, len(hq_id[i]) = v
                gt_depth = batch['gt_depths'].to(device)                 # b v 1 378 504


                # model input 
                if full_cfg.mvrm.input_img == 'hq':
                    model_input = batch['hq_views'].to(device)                  # b v 3 378 504
                elif full_cfg.mvrm.input_img == 'lq':
                    model_input = batch['lq_views'].to(device)                  # b v 3 378 504
                
                
                # NORMALIZATION
                b, v, c, h, w = model_input.shape 
                model_input = IMAGENET_NORMALIZE(model_input.view(b*v, c, h, w)).view(b, v, c, h, w)


                with torch.no_grad():
                    stage1_out_raw, mvrm_out = stage1_enc(
                                                        image=model_input, 
                                                        export_feat_layers=[], 
                                                        mvrm_cfg=full_cfg.mvrm.train, 
                                                        mode='train'
                                                        )
                    stage1_out = stage1_output_processor(stage1_out_raw)
                    train_pred_depth_np = stage1_out.depth                  # b v 378 504
                    train_pred_depth = torch.from_numpy(train_pred_depth_np).to(device) 
                    
                    
                    # save_image(model_input.squeeze(1), './img.jpg')
                    # save_image(pred_depth, './img_pred.jpg', normalize=True)
                    # save_image(gt_depth, './img_gt.jpg', normalize=True)
                    # save_image(batch['lq_views'].squeeze(1), './img_lq.jpg', normalize=True)
                    # save_image(batch['hq_views'].squeeze(1), './img_hq.jpg', normalize=True)
                    # breakpoint()


                    # FOR SAVING HQ LATENT
                    if full_cfg.mvrm.save_hq_latent:
                        lq_latent = mvrm_out['extract_feat']      # b v 973 3072
                        B = lq_latent.shape[0]
                        for i in range(B):
                            batch_id = hq_id[i]
                            full_id = batch_id[0]
                            volume = full_id.split('_')[-4]
                            scene = full_id.split('_')[-3]
                            camera = full_id.split('_')[-2]
                            frame_id = full_id.split('_')[-1]
                            save_root_path = f'/mnt/dataset1/MV_Restoration/hypersim/da3_clean_latent/input_singleview/ai_{volume}_{scene}/scene_cam_{camera}'
                            os.makedirs(save_root_path, exist_ok=True)
                            save_feat = lq_latent[i].reshape(973, 3072).detach().cpu()
                            torch.save(save_feat, f'{save_root_path}/{frame_id}.pt')
                            print(f'saving hq latent: {volume}_{scene}_{camera}_{frame_id}')
                        continue
                    
                            
                    # hypersim 
                    orig_H, orig_W = 768, 1024 
                    model_H, model_W = 378, 504
                    pH = pW = 14 
                    num_pH = model_H//pH    # 27
                    num_pW = model_W//pW    # 36
                    
                                        
                    lq_latent = mvrm_out['extract_feat']      # b v 973 3072
                    lq_cam_tkn = lq_latent[:,:,0]       # lq latent camera token (b v 3072)
                    lq_patch_tkn = lq_latent[:,:,1:]    # lq latent patch tokens (b v 972 3072)
                    lq_patch_tkn = rearrange(lq_patch_tkn, 'b v (num_pH num_pW) d -> b v d num_pH num_pW', v=1, num_pH=num_pH, num_pW=num_pW)   # b v 3072 27 36
                    
                    
                    hq_latent = batch['hq_latent_views'].to(device)
                    hq_cam_tkn = hq_latent[:,:,0]       # hq latent camera token (b v 3072)
                    hq_patch_tkn = hq_latent[:,:,1:]    # hq latent patch tokens (b v 972 3072)
                    hq_patch_tkn = rearrange(hq_patch_tkn, 'b v (num_pH num_pW) d -> b v d num_pH num_pW', v=1, num_pH=num_pH, num_pW=num_pW)   # b v 3072 27 36
                
                
                optimizer.zero_grad(set_to_none=True)
                with autocast(**autocast_kwargs):
                    transport_output = transport.training_losses_mvrm(model=ddp_model, x1=hq_patch_tkn, xcond=lq_patch_tkn, cfg=full_cfg)
                    loss = transport_output["loss"].mean()
                
            loss = loss.float()
            if scaler:  # f
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            if clip_grad:   # t
                if scaler:
                    scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            
            if global_step % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, ddp_model.module, decay=ema_decay)
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval # flow loss often has large variance so we record avg loss
                steps = torch.tensor(log_interval, device=device)
                stats = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if full_cfg.log.tracker.name == 'wandb':
                    wandb_utils.log(
                        stats,
                        step=global_step,
                    )
                running_loss = 0.0
                
            if global_step % sample_every == 0:
                model.eval()
                logger.info("Generating EMA samples...")
                with torch.no_grad():


                    # safety check
                    val_b, val_v, val_c, val_h, val_w = lq_patch_tkn.shape
                    assert pure_noise.shape == lq_patch_tkn.shape


                    # reshape
                    lq_patch_tkn = lq_patch_tkn.view(-1, val_c, val_h, val_w)
                    val_pure_noise = pure_noise.view(-1, val_c, val_h, val_w)


                    # lq_latent condition method
                    if full_cfg.mvrm.lq_latent_cond == 'addition':
                        val_xt = val_pure_noise + lq_patch_tkn
                    elif full_cfg.mvrm.lq_latent_cond == 'concat':
                        val_xt = torch.concat([val_pure_noise, lq_patch_tkn], dim=1)
                    
                    
                    # forward pass
                    with autocast(**autocast_kwargs):
                        restored_samples = eval_sampler(val_xt, ema_model_fn, **sample_model_kwargs)[-1]     # b*v c h w    
                    restored_samples.float()
                    restored_samples=restored_samples.view(val_b, val_v, val_c, val_h, val_w)   # b v c h w

                    # hypersim 
                    orig_H, orig_W = 768, 1024 
                    model_H, model_W = 378, 504
                    pH = pW = 14 
                    num_pH = model_H//pH    # 27
                    num_pW = model_W//pW    # 36
                    
                    # b v 3072 27 36 -> b v 972 3072
                    restored_samples = rearrange(restored_samples, 'b v d num_pH num_pW -> b v (num_pH num_pW) d', v=val_v, num_pH=num_pH, num_pW=num_pW) # b v n d 

                    mvrm_result={}
                    mvrm_result['restored_latent'] = restored_samples

                    val_stage1_out_raw, val_mvrm_out = stage1_enc(
                                                                image=model_input, 
                                                                export_feat_layers=[], 
                                                                mvrm_cfg=full_cfg.mvrm.val, 
                                                                mvrm_result=mvrm_result,
                                                                mode='val'
                                                                )
                    val_stage1_out = stage1_output_processor(val_stage1_out_raw)
                    val_pred_depth_np = val_stage1_out.depth                            # b v 378 504
                    val_pred_depth = torch.from_numpy(val_pred_depth_np).to(device)
                    

                    # VIS DEPTH 
                    # gt_depth_np: b v 378 504
                    # model_input: b v 3 378 504 
                    # train_pred_depth_np: b*v 378 504
                    # val_pred_depth_np: b*v 378 504
                    
                    np_model_input = model_input[0,0]   # # 378 504 3  
                    np_model_input = (np_model_input - np_model_input.min()) / (np_model_input.max() - np_model_input.min()) * 255.0
                    np_model_input = np_model_input.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]     
                    np_gt_depth = gt_depth[0,0,0].detach().cpu().numpy()    # 378 504
                    np_train_depth = train_pred_depth_np[0]                 # 378 504
                    np_val_depth = val_pred_depth_np[0]                     # 378 504

                    vis_gt_depth = depth_to_colormap(np_gt_depth)               # 378 504 3
                    vis_train_depth = depth_to_colormap(np_train_depth)         # 378 504 3
                    vis_val_depth = depth_to_colormap(np_val_depth)             # 378 504 3
                    
                    vis_err_train_depth = depth_error_to_colormap_thresholded(np_gt_depth, np_train_depth, thr=0.1)
                    vis_err_val_depth = depth_error_to_colormap_thresholded(np_gt_depth, np_val_depth, thr=0.1)
                    
                    vis_depth_cat = np.concatenate(
                        [np_model_input, vis_train_depth, vis_val_depth, vis_gt_depth, vis_err_train_depth, vis_err_val_depth, ],
                        axis=1
                    )

                    # Output path
                    vis_depth_save_dir = f'{experiment_dir}/vis_depth'
                    os.makedirs(vis_depth_save_dir, exist_ok=True)
                    cv2.imwrite(f'{vis_depth_save_dir}/step{global_step:07}_{hq_id[0][0]}.jpg', vis_depth_cat)               

                    breakpoint()
                
                    dist.barrier()
                logger.info("Generating EMA samples done.")
                model.train()

            # ckpt saving
            if global_step > 0 and global_step % ckpt_step_interval == 0 and rank == 0:
                logger.info(f"Saving checkpoint at globalstep {global_step}...")
                ckpt_path = f"{checkpoint_dir}/ep-{global_step:07d}.pt" 
                save_checkpoint(
                    ckpt_path,
                    global_step,
                    epoch,
                    ddp_model,
                    ema_model,
                    optimizer,
                    scheduler,
                )
                
            # if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
            #     logger.info("Starting evaluation...")
            #     model.eval()
            #     eval_models = [(ema_model_fn, "ema")]
            #     if eval_model:
            #         eval_models.append((model_fn, "model"))
            #     for fn, mod_name in eval_models:
            #         eval_stats = evaluate_generation_distributed(
            #             fn,
            #             eval_sampler,
            #             latent_size,
            #             sample_model_kwargs,
            #             use_guidance,
            #             stage1_enc, 
            #             eval_dataset,
            #             len(eval_dataset),
            #             rank = rank,
            #             world_size = world_size,
            #             device = device,
            #             batch_size = micro_batch_size,
            #             experiment_dir = experiment_dir,
            #             global_step = global_step,
            #             autocast_kwargs = autocast_kwargs,
            #             reference_npz_path = reference_npz_path
            #         )
            #         # log with prefix
            #         eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
            #         if full_cfg.log.tracker.name == 'wandb':
            #             wandb_utils.log(eval_stats, step=global_step)
            #         model.train()
            #     logger.info("Evaluation done.")
            global_step += 1
            num_batches += 1
        
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
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()



if __name__ == "__main__":
    main()
