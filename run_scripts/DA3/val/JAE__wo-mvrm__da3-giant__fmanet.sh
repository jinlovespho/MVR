
#!/bin/bash

NUM_FUSION_WORKERS=4      
GPU_ID=1
CONFIG_PATH="run_configs/DA3/val/wo-mvrm__da3-giant__fmanet.yaml"

echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "GPU: $GPU_ID"
echo "Fusion Workers: $NUM_FUSION_WORKERS"
echo "=========================================="
echo ""

export DA3_LQ_ROOT_PATH='/mnt/dataset1/jaeeun/nips26/FMA-Net/results/FMA-Net/filtered_cam_blur_100_resize_640_selected10'
export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_hi_diff/filtered_cam_blur_100'

echo "LQ: $DA3_LQ_ROOT_PATH"
echo "RES: $DA3_RES_ROOT_PATH"
echo ""

time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.evaluator --config ${CONFIG_PATH}
