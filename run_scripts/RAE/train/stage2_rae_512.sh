CUDA=5

export CUDA=${CUDA}
export EXPERIMENT_NAME="s20-g${CUDA}__bf16__train__RAE-512-stage2__imagenet__ep1400-bs8-lr2e-4__gan__ckpt-scratch"
# export EXPERIMENT_NAME="s20-g${CUDA}__fp32__train__RAE-512-stage2__imagenet__ep1400-bs8-lr2e-4__gan__ckpt-scratch"
export WANDB_KEY="e32eed0c2509bf898b850b0065ab62345005fb73"
export ENTITY="jinlovespho"
export PROJECT="eccv26_mv_restoration"


# RAE-512 stage2 training
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 RAE/src/train.py \
  --config RAE/configs/stage2/training/ImageNet512/DiTDH-XL_DINOv2-B.yaml \
  --data-path /media/data1/ImageNet2012/train \
  --results-dir result_ckpts/RAE/imagenet/stage2 \
  --image-size 512 --precision bf16 --wandb --compile 