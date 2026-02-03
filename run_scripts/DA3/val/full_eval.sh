cd Depth-Anything-3
MODEL=depth-anything/DA3-GIANT-1.1


CUDA_VISIBLE_DEVICES=4 python -m depth_anything_3.bench.evaluator \
    model.path=$MODEL \
    eval.datasets=[eth3d,7scenes] \
    eval.modes=[pose,recon_unposed,recon_posed] \
    inference.debug=true


# # Full evaluation (inference + evaluation + print results)
# CUDA_VISIBLE_DEVICES=4 python -m depth_anything_3.bench.evaluator \
#     model.path=$MODEL 


# # Skip inference, only evaluate existing predictions
# CUDA_VISIBLE_DEVICES=5 python -m depth_anything_3.bench.evaluator eval.eval_only=true

# # Only print saved metrics
# CUDA_VISIBLE_DEVICES=4 python -m depth_anything_3.bench.evaluator eval.print_only=true


# # Evaluate specific datasets
# CUDA_VISIBLE_DEVICES=4 python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.datasets=[hiroom]

# # Evaluate specific modes
# CUDA_VISIBLE_DEVICES=4 python -m depth_anything_3.bench.evaluator model.path=$MODEL eval.modes=[pose,recon_unposed]

# # Combine dataset and mode selection
# CUDA_VISIBLE_DEVICES=4 python -m depth_anything_3.bench.evaluator model.path=$MODEL \
#     eval.datasets=[hiroom] \
#     eval.modes=[pose]



# # Debug specific scenes
# CUDA_VISIBLE_DEVICES=5 python -m depth_anything_3.bench.evaluator \
#     model.path=$MODEL \
#     eval.datasets=[eth3d] \
#     eval.modes=[pose,recon_unposed,recon_posed] \
#     eval.scenes=[courtyard] \
#     inference.debug=true
