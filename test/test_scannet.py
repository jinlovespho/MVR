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

from test_utils import load_depth, align_scale_median, compute_depth_metrics, depth_to_colormap, depth_error_to_colormap, fmt, fmt_int, write_scene_header


def main(cfg):
    
    
    # set cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    
    
    # load 3D Feed Forward (3DFF) model 
    if cfg.model.val.mv_3dff.model == 'vggt':
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
        
    elif cfg.model.val.mv_3dff.model == 'da3':
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device)
        model.eval()

    
    # loop through eval data paths
    for data_path in cfg.data.val.scannet.data_paths:
        print(f'Evaluating ScanNet++ - {data_path.split("/")[-1]}')
        
        
        # set save path
        if cfg.log.save_path is not None:
            print('Saving path to: ', cfg.log.save_path)
            save_path = cfg.log.save_path
        else:
            print('Saving path set to data path: ', data_path)
            save_path = data_path


        depth_metric_path = f'{save_path}/{cfg.model.val.mv_3dff.model}'
        os.makedirs(depth_metric_path, exist_ok=True)
        
        depth_metric_txt = open(f'{save_path}/scannet_metric_depth.txt', 'w', encoding='utf-8')
        all_metrics = []
        
        # loop through scannet scenes
        for scene in tqdm(cfg.data.val.scannet.eval_scenes):
            print('processing scene: ', scene)
            
            
            # load imgs 
            imgs_path = sorted(glob.glob(f'{data_path}/data/{scene}/dslr/resized_undistorted_images/*'))
            tot_num_input_view = len(imgs_path)
            assert len(imgs_path) != 0


            # set number of input views
            if cfg.data.val.scannet.num_input_view is not None:
                num_input_view = cfg.data.val.scannet.num_input_view
                if num_input_view > tot_num_input_view:
                    num_input_view = tot_num_input_view
                imgs_path = imgs_path[:num_input_view]
            print(f'Num_input_view: {num_input_view}/{tot_num_input_view}')
            
            
            # if gt depth is provided we calculate the depth metrics
            if cfg.data.val.scannet.gt_depth_path is not None:
                gt_depths_path = sorted(glob.glob(f'{cfg.data.val.scannet.gt_depth_path}/{scene}_dslr_depth/{scene}/ground_truth_depth/dslr_images/*'))

            
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
                if cfg.data.val.scannet.gt_depth_path is not None:
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
    parser = argparse.ArgumentParser(description="mvr_val_scannet")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
