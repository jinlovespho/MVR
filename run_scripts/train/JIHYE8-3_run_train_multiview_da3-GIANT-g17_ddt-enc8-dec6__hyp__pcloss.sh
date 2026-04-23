
SERVER=JIHYE8-3
NUM_GPUS=4
CUDA=0,1,2,3

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"

# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train.py \
  --config run_configs/train/JIHYE8-3_run_train_multiview_da3-GIANT-g17_ddt-enc8-dec6__hyp__pcloss.yaml