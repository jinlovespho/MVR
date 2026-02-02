import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import cv2 
import numpy as np
import torch 
import torch.nn.functional as F 



import val_initialize
import val_utils 




# from test_utils import load_depth, align_scale_median, compute_depth_metrics, depth_to_colormap, depth_error_to_colormap, fmt, fmt_int, write_scene_header


def main(cfg):
    
    
    # set cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)




    
    # -------------------------------------
    #           load val data
    # -------------------------------------
    val_datasets = val_initialize.load_data(cfg)



    # -------------------------------------
    #           load models 
    # -------------------------------------
    val_models = val_initialize.load_model(cfg, device, dtype)
    
    
    
    # -------------------------------------
    #           print val info
    # -------------------------------------
    print('-'*50)
    print('Val models loaded: ', val_models.keys())
    print('Val datasets loaded: ', val_datasets.keys())
    print('-'*50)
    
    


    # -----------------------
    #       Metrics
    # -----------------------
    metrics={}



    # ------------------------------------------
    #        validation loop (per dataset)
    # ------------------------------------------
    for val_data_name, val_data in val_datasets.items():


        # -------------------------------------------------------------------------
        # Initialize metric for full-image and cropped-image evaluation
        # -------------------------------------------------------------------------

        # ===== Full image metrics =====
        metrics[f'{val_data_name}_full_psnr'] = []
        metrics[f'{val_data_name}_full_ssim'] = []
        metrics[f'{val_data_name}_full_lpips'] = []
        metrics[f'{val_data_name}_full_dists'] = []
        metrics[f'{val_data_name}_full_niqe'] = []
        metrics[f'{val_data_name}_full_musiq'] = []
        metrics[f'{val_data_name}_full_maniqa'] = []
        metrics[f'{val_data_name}_full_clipiqa'] = []

        # Min–max normalized (full)
        metrics[f'{val_data_name}_full_norm_psnr'] = []
        metrics[f'{val_data_name}_full_norm_ssim'] = []
        metrics[f'{val_data_name}_full_norm_lpips'] = []
        metrics[f'{val_data_name}_full_norm_dists'] = []
        metrics[f'{val_data_name}_full_norm_niqe'] = []
        metrics[f'{val_data_name}_full_norm_musiq'] = []
        metrics[f'{val_data_name}_full_norm_maniqa'] = []
        metrics[f'{val_data_name}_full_norm_clipiqa'] = []



        # ------------------------------------------
        #        validation loop (per sample)
        # ------------------------------------------
        for val_sample_idx, val_sample in enumerate(val_data):
            
            print(f'{val_data_name} - {val_sample_idx+1}/{len(val_data)}') 
            
            if val_data_name == 'hypersim':
                pass 
            elif val_data_name == 'tartanair':
                pass
            elif val_data_name == 'eth3d':
                lq = val_sample['lq_path']
                hq = val_sample['hq_path']
                depth = val_sample['depth_path']
            
            
            
            if cfg.val.model == 'da3':
                pred_da3 = val_models['da3'].inference(imgs_path)
                pred_depth = torch.from_numpy(pred_da3.depth).to(device)                 # (F, 336 504)

            
            # set seed 
            generator = None
            if accelerator.is_main_process and cfg.init.seed is not None:
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(cfg.init.seed)


            # load val anns
            val_lq_path = val_sample['lq_path']
            val_hq_path = val_sample['hq_path']
            val_gt_text = val_sample['text']
            val_bbox = val_sample['bbox']       # xyxy
            val_polys = val_sample['poly']
            val_img_id = val_sample['img_id']
            val_vlm_cap = val_sample['vlm_cap']
            
            
            # process hq image 
            val_hq_pil = Image.open(val_hq_path).convert("RGB") 
            
            
            # process lq image 
            val_lq_pil = Image.open(val_lq_path).convert("RGB") # 128 128 
            ori_width, ori_height = val_lq_pil.size
            rscale = 4  # upscale x4
            # for shortest side smaller than 128, resize
            if ori_width < 512//rscale or ori_height < 512//rscale:
                scale = (512//rscale)/min(ori_width, ori_height)
                tmp_image = val_lq_pil.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)
                val_lq_pil = tmp_image
            val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]*rscale, val_lq_pil.size[1]*rscale), Image.BICUBIC)
            val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]//8*8, val_lq_pil.size[1]//8*8), Image.BICUBIC)
            






    
    # loop through eval data paths
    for data_path in cfg.data.val.eth3d.data_paths:
        print(f'Evaluating ETH3D - {data_path.split("/")[-1]}')
        
        
        # set save path
        if cfg.log.save_path is not None:
            print('Saving path to: ', cfg.log.save_path)
            save_path = cfg.log.save_path
        else:
            print('Saving path set to data path: ', data_path)
            save_path = data_path


        depth_metric_path = f'{save_path}/{cfg.model.val.mv_3dff.model}'
        os.makedirs(depth_metric_path, exist_ok=True)
        
        depth_metric_txt = open(f'{save_path}/eth3d_metric_depth.txt', 'w', encoding='utf-8')
        all_metrics = []
        
        # loop through eth3d scenes
        for scene in tqdm(cfg.data.val.eth3d.eval_scenes):
            print('processing scene: ', scene)
            
            
            # load imgs 
            imgs_path = sorted(glob.glob(f'{data_path}/image/{scene}/*'))
            tot_num_input_view = len(imgs_path)
            assert len(imgs_path) != 0
            
            # set number of input views
            if cfg.data.val.eth3d.num_input_view is not None:
                num_input_view = cfg.data.val.eth3d.num_input_view
                if num_input_view > tot_num_input_view:
                    num_input_view = tot_num_input_view
                imgs_path = imgs_path[:num_input_view]
            else:
                num_input_view=tot_num_input_view
            print(f'Num_input_view: {num_input_view}/{tot_num_input_view}')
            
            
            # if gt depth is provided we calculate the depth metrics
            if cfg.data.val.eth3d.gt_depth_path is not None:
                gt_depths_path = sorted(glob.glob(f'{cfg.data.val.eth3d.gt_depth_path}/{scene}_dslr_depth/{scene}/ground_truth_depth/dslr_images/*'))

            
            # set pred depth save dir
            save_pred_depth_dir = f'{save_path}/{cfg.model.val.mv_3dff.model}__{cfg.log.msg}/pred_depth/{scene}'
            os.makedirs(save_pred_depth_dir, exist_ok=True)
            
            # set pred depth vis save dir
            save_pred_depth_vis_dir = f'{save_path}/{cfg.model.val.mv_3dff.model}__{cfg.log.msg}/pred_depth_vis/{scene}'
            os.makedirs(save_pred_depth_vis_dir, exist_ok=True)


            # inference
            with torch.no_grad():
                
                # vggt inference 
                if cfg.model.val.mv_3dff.model == 'vggt':
                    with torch.cuda.amp.autocast(dtype=dtype):
                        imgs = load_and_preprocess_images(imgs_path).to(device)
                        pred_vggt = model(imgs)
                        pred_depth = pred_vggt['depth']  # [F, 350, 518, 1]
                        pred_depth = pred_depth.squeeze(dim=(0,-1))
                        # pred_depth = pred_depth.detach().cpu().numpy().astype(np.float32)
                        
                # da3 inference 
                elif cfg.model.val.mv_3dff.model == 'da3':
                    pred_da3 = model.inference(imgs_path)
                    pred_depth = torch.from_numpy(pred_da3.depth).to(device)                 # (F, 336 504)

                    breakpoint()
                
            # Safety check
            assert len(pred_depth) == len(imgs_path)
                    
                    
            scene_metrics = []
            wrote_header = False
            # Save per-image depth
            for f_idx, img_path in enumerate(imgs_path):
                                
                img_id = img_path.split('/')[-1].split('.')[0]

                
                # pred depth 
                depth = pred_depth[f_idx]  # [H, W]
                
                
                # Save pred depth as npy 
                if cfg.log.depth.save_npy:                    
                    np.save(os.path.join(save_pred_depth_dir, f'{img_id}.npy'), depth.detach().cpu().numpy().astype(np.float32))


                # gt depth
                if cfg.data.val.eth3d.gt_depth_path is not None:
                    H, W = 4032, 6048
                    gt_depth = load_depth(gt_depths_path[f_idx], H, W)
                    gt_depth = torch.from_numpy(gt_depth).to(device)

                    # match pred_depth size to gt_depth
                    if depth.shape[-2:] != gt_depth.shape[-2:]:
                        depth = F.interpolate(depth[None, None, :], size=(H,W), mode="bilinear", align_corners=False)[0,0]       
                            
                    # median scaling - pred_depth
                    pred_depth_aligned = align_scale_median(gt_depth, depth)
                    depth_metrics = compute_depth_metrics(gt_depth, pred_depth_aligned)
                    
                    scene_metrics.append(depth_metrics)
                    all_metrics.append(depth_metrics)
                    
                    if not wrote_header:
                        write_scene_header(depth_metric_txt, scene)
                        wrote_header = True
                    
                    abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3 = depth_metrics
                    valid_pixels = np.isfinite(gt_depth.detach().cpu().numpy().astype(np.float32)).sum()

                    depth_metric_txt.write(
                        f"{img_id:<12}"
                        f"{fmt(abs_rel)}{fmt(sq_rel)}{fmt(rmse,9)}{fmt(rmse_log,10)}"
                        f"{fmt(d1)}{fmt(d2)}{fmt(d3)}{fmt_int(valid_pixels)}\n"
                    )
                                        
                    rgb = cv2.imread(img_path)
                    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
                    gt_vis = depth_to_colormap(gt_depth.detach().cpu().numpy().astype(np.float32), resize=(512,512))
                    pred_vis = depth_to_colormap(depth.detach().cpu().numpy().astype(np.float32), resize=(512,512))
                    err_vis = depth_error_to_colormap(
                                    gt_depth.detach().cpu().numpy().astype(np.float32), 
                                    depth.detach().cpu().numpy().astype(np.float32),
                                    resize=(512,512)
                                )
                    concat = np.concatenate([rgb, pred_vis, gt_vis, err_vis], axis=1)
                    cv2.imwrite(f'{save_pred_depth_vis_dir}/{img_id}.png', concat)
                
                else:
                    # --------------------------------------------------
                    # No GT available: visualize RGB + predicted depth
                    # --------------------------------------------------
                    rgb = cv2.imread(img_path)
                    if rgb is None:
                        continue

                    # Resize RGB for visualization
                    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

                    # Pred depth visualization
                    pred_vis = depth_to_colormap(
                        depth.detach().cpu().numpy().astype(np.float32),
                        resize=(512, 512)
                    )

                    # Concatenate RGB | Pred Depth
                    concat = np.concatenate([rgb, pred_vis], axis=1)

                    # Save
                    cv2.imwrite(f'{save_pred_depth_vis_dir}/{img_id}.png',concat)

            if len(scene_metrics) > 0:
                scene_mean = np.mean(scene_metrics, axis=0)
                depth_metric_txt.write("-" * 90 + "\n")
                depth_metric_txt.write(
                    f"{'MEAN':<12}"
                    f"{fmt(scene_mean[0])}{fmt(scene_mean[1])}"
                    f"{fmt(scene_mean[2],9)}{fmt(scene_mean[3],10)}"
                    f"{fmt(scene_mean[4])}{fmt(scene_mean[5])}{fmt(scene_mean[6])}"
                    f"{fmt_int(len(scene_metrics))}\n"
                )

        if len(all_metrics) > 0:
            global_mean = np.mean(all_metrics, axis=0)
            depth_metric_txt.write("\n" + "=" * 90 + "\n")
            depth_metric_txt.write("GLOBAL MEAN (ALL SCENES)\n")
            depth_metric_txt.write("-" * 90 + "\n")
            depth_metric_txt.write(
                f"{'ALL':<12}"
                f"{fmt(global_mean[0])}{fmt(global_mean[1])}"
                f"{fmt(global_mean[2],9)}{fmt(global_mean[3],10)}"
                f"{fmt(global_mean[4])}{fmt(global_mean[5])}{fmt(global_mean[6])}"
                f"{fmt_int(len(all_metrics))}\n"
            )
        depth_metric_txt.close()


    print('Finish !')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mvr_val")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)