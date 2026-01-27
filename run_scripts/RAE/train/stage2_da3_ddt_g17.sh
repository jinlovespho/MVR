
SERVER=20
NUM_GPUS=2
CUDA=6,7

export SERVER=${SERVER}
export CUDA=${CUDA}

# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train.py \
  --config RAE/configs/stage2/training/hypersim/da3_ddt-g17.yaml \
