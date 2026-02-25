
SERVER=20
NUM_GPUS=1
CUDA=0

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"
# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train_vae21.py\
  --config run_configs/train/train_multiview_VAE21_ddt.yaml \