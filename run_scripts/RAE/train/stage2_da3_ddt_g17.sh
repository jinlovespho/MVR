
SERVER=20
CUDA=4

export SERVER=${SERVER}
export CUDA=${CUDA}

# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=1 RAE/src/train.py \
  --config RAE/configs/stage2/training/hypersim/da3_ddt-g17.yaml \




# SERVER=20
# CUDA=6,7

# export SERVER=${SERVER}
# export CUDA=${CUDA}

# # stage2 training (stage1 - da3)
# CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=2 RAE/src/train.py \
#   --config RAE/configs/stage2/training/hypersim/da3_ddt-g17.yaml \
