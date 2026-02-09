

# Set server log message
SERVER=JIHYE
# set total number of gpus for training
NUM_GPUS=4
# set gpu_id for training
CUDA=0,1,2,3

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train.py \
  --config run_configs/train/JIHYE_train_multiview_da3_ddt-g17_kernel50.yaml \