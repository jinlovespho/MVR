export EXPERIMENT_NAME="decoder_for_vggt"
export WANDB_KEY="e32eed0c2509bf898b850b0065ab62345005fb73"
export ENTITY="jinlovespho"
export PROJECT="eccv26_mv_restoration"

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 RAE/src/train_stage1.py \
  --config RAE/configs/stage1/training/DINOv2-B_decXL.yaml \
  --data-path /media/data1/ImageNet2012/train \
  --results-dir result_ckpts/RAE/pho_tmp \
  --image-size 256 --precision bf16 --wandb