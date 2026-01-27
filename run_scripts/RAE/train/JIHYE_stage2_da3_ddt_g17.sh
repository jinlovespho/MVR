
SERVER=SAMSUNG
# -------- JIHYE --------
NUM_GPUS=8
CUDA=0,1,2,3,4,5,6,7
# -----------------------


export SERVER=${SERVER}
export CUDA=${CUDA}
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/jihye_train.py \
  --config RAE/configs/stage2/training/hypersim/JIHYE_da3_ddt-g17.yaml \
