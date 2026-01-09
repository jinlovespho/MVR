CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run --standalone --nproc_per_node=1 vggt/training/launch.py
