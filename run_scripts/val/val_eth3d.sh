#!/bin/bash

CUDA="4"
CUDA_VISIBLE_DEVICES=${CUDA} python test/test_eth3d.py --config run_configs/val/val_eth3d.yaml


