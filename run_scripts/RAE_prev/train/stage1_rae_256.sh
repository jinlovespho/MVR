export EXPERIMENT_NOTE="gan-ep4"
# export EXPERIMENT_NOTE="gan-None"
export WANDB_KEY="e32eed0c2509bf898b850b0065ab62345005fb73"
export ENTITY="jinlovespho"
export PROJECT="eccv26_mv_restoration"

CUDA=4
export CUDA=${CUDA}

CUDA_VISIBLE_DEVICES=${CUDA} torchrun --standalone --nproc_per_node=1 RAE/src/train_stage1.py \
  --config RAE/configs/stage1/training/DINOv2-B_decXL_256.yaml \
  --data-path /media/data1/ImageNet2012/train \
  --results-dir result_ckpts/RAE \
  --image-size 256 --precision bf16 --wandb