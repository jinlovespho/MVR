---
license: apache-2.0
pipeline_tag: image-to-3d
tags:
- novel-view-synthesis
- multi-view-diffusion
- depth-estimation
- 3d-reconstruction
---

# GLD: Geometric Latent Diffusion

**Repurposing Geometric Foundation Models for Multi-view Diffusion**

[[Paper]](https://huggingface.co/papers/2603.22275) | [[arXiv]](https://arxiv.org/abs/2603.22275) | [[Project Page]](https://cvlab-kaist.github.io/GLD/) | [[Code]](https://github.com/cvlab-kaist/GLD)

Geometric Latent Diffusion (GLD) is a framework that repurposes the geometrically consistent feature space of geometric foundation models (such as [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-3) and [VGGT](https://github.com/facebookresearch/vggt)) as the latent space for multi-view diffusion. By operating in this space rather than a view-independent VAE latent space, GLD achieves consistent novel view synthesis (NVS) and 3D reconstruction with significantly faster training convergence.

## Quick Start

```bash
git clone https://github.com/cvlab-kaist/GLD.git
cd GLD
conda env create -f environment.yml
conda activate gld

# Download all checkpoints
python -c "from huggingface_hub import snapshot_download; snapshot_download('SeonghuJeon/GLD', local_dir='.')"

# Run demo
./run_demo.sh da3
```

## Files

| File | Description | Size |
|------|-------------|------|
| `checkpoints/da3_level1.pt` | DA3 Level-1 diffusion | 3.0G |
| `checkpoints/da3_cascade.pt` | DA3 Cascade (L1→L0) | 1.8G |
| `checkpoints/vggt_level1.pt` | VGGT Level-1 diffusion | 3.1G |
| `checkpoints/vggt_cascade.pt` | VGGT Cascade (L1→L0) | 3.1G |
| `pretrained_models/da3/model.safetensors` | DA3-Base encoder | 0.5G |
| `pretrained_models/da3/dpt_decoder.pt` | DPT decoder (depth + geometry) | 0.4G |
| `pretrained_models/mae_decoder.pt` | DA3 MAE decoder (RGB) | 1.6G |
| `pretrained_models/vggt/mae_decoder.pt` | VGGT MAE decoder (RGB) | 1.6G |

## Citation

```bibtex
@article{jang2026gld,
  title={Repurposing Geometric Foundation Models for Multi-view Diffusion},
  author={Jang, Wooseok and Jeon, Seonghu and Han, Jisang and Choi, Jinhyeok and Kwon, Minkyung and Kim, Seungryong and Xie, Saining and Liu, Sainan},
  journal={arXiv preprint arXiv:2603.22275},
  year={2026}
}
```

## Acknowledgements

Built upon [RAE](https://github.com/nicknign/RAE_release), [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-3), [VGGT](https://github.com/facebookresearch/vggt), [CUT3R](https://github.com/naver/CUT3R), and [SiT](https://github.com/willisma/SiT).
