
SERVER=Nvidia
NUM_GPUS=1
CUDA=4

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"
# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train.py \
  --config run_configs/train/NVIDIA2_train_multiview_da3-GIANT_ddt-g17_lqkernel50_ema95.yaml \