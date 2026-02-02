#!/bin/bash

CUDA="4"
CUDA_VISIBLE_DEVICES=${CUDA} python val/val.py --config run_configs/val/val.yaml


