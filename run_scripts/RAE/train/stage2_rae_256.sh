
SERVER=20
CUDA=4


IMAGE_SIZE=256
PRECISION=bf16
RESULT_DIR=result_ckpts/RAE/imagenet256/stage2


export SERVER=${SERVER}
export CUDA=${CUDA}
export EXPERIMENT_NAME="${PRECISION}__train__RAE-256-stage2__imagenet__ep1400-bs32-lr2e-4__gan__ckpt-scratch"

export ENTITY="jinlovespho"
export PROJECT="eccv26_mv_restoration"
export WANDB_KEY="e32eed0c2509bf898b850b0065ab62345005fb73"

# RAE-256 stage2 training
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=1 RAE/src/train.py \
  --config RAE/configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml \
  --data-path /media/data1/ImageNet2012/train \
  --results-dir ${RESULT_DIR} \
  --image-size ${IMAGE_SIZE} --precision ${PRECISION} --wandb --compile