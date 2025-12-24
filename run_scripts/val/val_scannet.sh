#!/bin/bash

CUDA="3"
CUDA_VISIBLE_DEVICES=${CUDA} python test/test_scannet.py --config run_configs/val/val_scannet.yaml


