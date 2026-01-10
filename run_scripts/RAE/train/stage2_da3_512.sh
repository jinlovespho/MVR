
SERVER=20
CUDA=3

export SERVER=${SERVER}
export CUDA=${CUDA}

# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=1 RAE/src/train.py \
  --config RAE/configs/stage2/training/hypersim/da3.yaml \
  # --compile