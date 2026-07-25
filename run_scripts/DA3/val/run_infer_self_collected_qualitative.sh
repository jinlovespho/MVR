#!/bin/bash

CUDA="0"

# `python -m depth_anything_3.bench.evaluator` (used elsewhere) puts the cwd on
# sys.path automatically; a plain `python script.py` invocation does not, so
# the top-level `mvr` package (imported by depth_anything_3/api.py) isn't
# found unless we add the repo root to PYTHONPATH explicitly.
PYTHONPATH="$(pwd):${PYTHONPATH}" CUDA_VISIBLE_DEVICES=${CUDA} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_scripts/DA3/val/infer_self_collected_qualitative.py
