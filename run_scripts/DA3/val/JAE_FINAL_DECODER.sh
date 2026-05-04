#!/bin/bash

NUM_FUSION_WORKERS=4      
GPU_ID=1
CONFIG_PATH="run_configs/DA3/val/ALL_IN_ONE_MVRM_JAE_decoder.yaml"

echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "GPU: $GPU_ID"
echo "Fusion Workers: $NUM_FUSION_WORKERS"
echo "=========================================="
echo ""

export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/restormer_benchmark/GoPro/input'
# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/restormer_benchmark/HIDE/input'
# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/restormer_benchmark/RealBlur_J/input'
# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/restormer_benchmark/RealBlur_R/input'

export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_600'

echo "LQ: $DA3_LQ_ROOT_PATH"
echo "RES: $DA3_RES_ROOT_PATH"
echo ""

time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
    --config ${CONFIG_PATH}

# =============================================================================
# filtered_cam_blur_100 
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH}


# =============================================================================
# filtered_cam_blur_150 
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_150'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_150'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH}
#     eval.datasets=[7scenes]



# =============================================================================
# filtered_cam_blur_200 
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_600'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_600'

# echo "LQ: $DA3_LQ_ROOT_PATH"
# echo "RES: $DA3_RES_ROOT_PATH"
# echo ""

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH}


# =============================================================================
# HI-Diff
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_hi_diff/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH}


# =============================================================================
# InstructIR
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_instructir/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH}


# =============================================================================
# Only one dataset
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH} \
#     eval.datasets=[dtu]


# =============================================================================
# Only pose
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH} \
#     eval.modes=[pose]


# =============================================================================
# Only recon_unposed
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH} \
#     eval.modes=[recon_unposed]


# =============================================================================
# Only depth
# =============================================================================

# export DA3_LQ_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_100'
# export DA3_RES_ROOT_PATH='/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/restored_restormer/filtered_cam_blur_100'

# time CUDA_VISIBLE_DEVICES=${GPU_ID} python -m depth_anything_3.bench.JAE_evaluator \
#     --config ${CONFIG_PATH} \
#     eval.modes=[recon_unposed]

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
