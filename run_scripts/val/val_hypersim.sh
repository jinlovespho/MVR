#!/bin/bash

CUDA=7
CUDA_VISIBLE_DEVICES=${CUDA} python test/test_hypersim.py --config run_configs/val/val_hypersim.yaml


