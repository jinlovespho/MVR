CUDA=5

export CUDA=${CUDA}
export EXPERIMENT_NAME="s20-g${CUDA}__fp32__train__RAE-256__ep8-bs32-lr2e-4__gan__ckpt-pretrained"
export WANDB_KEY="e32eed0c2509bf898b850b0065ab62345005fb73"
export ENTITY="jinlovespho"
export PROJECT="eccv26_mv_restoration"


CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=1 RAE/src/train_stage1.py \
  --config RAE/configs/stage1/training/DINOv2-B_decXL_256.yaml \
  --data-path /media/data1/ImageNet2012/train \
  --results-dir result_ckpts/RAE \
  --image-size 256 --precision fp32 --wandb