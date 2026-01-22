import numpy as np 

unposed_pred = np.load('result_val/DA3/eth3d_tmp/model_results/eth3d/courtyard/unposed/exports/mini_npz/results.npz')
unposed_gt = np.load('result_val/DA3/eth3d_tmp/model_results/eth3d/courtyard/unposed/exports/gt_meta.npz')

posed_pred = np.load('result_val/DA3/eth3d_tmp/model_results/eth3d/courtyard/posed/exports/mini_npz/results.npz')
posed_gt = np.load('result_val/DA3/eth3d_tmp/model_results/eth3d/courtyard/posed/exports/gt_meta.npz')

breakpoint()