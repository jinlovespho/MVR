CUDA=4

export CUDA=${CUDA}
export EXPERIMENT_NAME="s20-g${CUDA}__fp32__train__RAE-256-stage2__imagenet__ep1400-bs32-lr2e-4__gan__ckpt-scratch"
export WANDB_KEY="e32eed0c2509bf898b850b0065ab62345005fb73"
export ENTITY="jinlovespho"
export PROJECT="eccv26_mv_restoration"


# RAE-256 stage2 training
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 RAE/src/train.py \
  --config RAE/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path /media/data1/ImageNet2012/train \
  --results-dir result_ckpts/RAE/imagenet/stage2 \
  --image-size 256 --precision fp32 --wandb --compile