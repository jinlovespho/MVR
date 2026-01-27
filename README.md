<div align="center">
<h1>
UniT: Unified Diffusion Transformer for High-Fidelity Text-Aware Image Restoration</h1>


[**Jin Hyeon Kim**](https://github.com/jinlovespho)<sup>1</sup>,&nbsp;&nbsp;
**Paul Hyunbin Cho**<sup>1</sup>,&nbsp;&nbsp;
**Claire Kim**<sup>1</sup>,&nbsp;&nbsp;

[**Jaewon Min**](https://github.com/Min-Jaewon/)<sup>1</sup>,&nbsp;&nbsp; 
[**Jaeeun Lee**](https://github.com/babywhale03)<sup>2</sup>,&nbsp;&nbsp;
**Jihye Park**<sup>2</sup>,&nbsp;&nbsp; **Yeji Choi**<sup>1</sup>,&nbsp;&nbsp; [**Seungryong Kim**](https://scholar.google.com/citations?hl=zh-CN&user=cIK1hS8AAAAJ)<sup>1</sup>

<sup>1</sup> KAIST&nbsp;AI &nbsp;&nbsp;&nbsp;
<sup>2</sup> Samsung&nbsp;Electronics

<a href="https://arxiv.org/abs/2512.08922">
  <img src="https://img.shields.io/badge/arXiv-2512.08922-B31B1B">
</a>
        <a href="https://cvlab-kaist.github.io/UniT/"><img src="https://img.shields.io/badge/Project%20Page-online-1E90FF"></a>
</div>


# 🔈 News 
- [ ] 🔥 Training code coming soon.
- [ ] 📄 **2026.xx.xx** — Arxiv paper released!
- [ ] ⚔️ **2026.xx.xx** — Demo inference code released!
- [x] 🚀 **2026.01.27** — Shared code to jihye



# 📁 Dataset Preparation
**1. Hypersim**
```bash 
# run the bash file to download hypersim at ./data/hypersim
bash download_bash/data/hypersim/download_hypersim.sh
```

**2. Tartanair**
```bash 
# run the bash file to download tartanair at ./data/tartanair
pip install boto3 colorama minio huggingface_hub
bash download_bash/data/tartanair_tools/download_tartanair.sh
```




#  👟 Installation Walkthrough


**1. Conda Environment**

```bash
conda create -n mvr python=3.10 -y
conda activate mvr
```

**2. Library Installation**
- Download the following libraries in the order listed below.
```bash
# download torch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# download other libraries
pip install -r requirements.txt

# download da3
cd Depth-Anything-3/
pip install -e . 
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70


# # download vggt (이건 나중에 지금 말고)
# cd vggt/
# pip install -e .
```





# 🔥 Training 



### 1. Configure bash file
First, modify the bash file to set gpu ddp training -> [here](run_scripts/RAE/train/JIHYE_stage2_da3_ddt_g17.sh)



### 2. Configure yaml file
Second, modify the yaml file to set up training_data_path, batch_size, saving_dir, etc -> [here](RAE/configs/stage2/training/hypersim/JIHYE_da3_ddt-g17.yaml)



### 3. Run the following bash file
Lastly, run the bash file for training
```bash
bash run_scripts/RAE/train/JIHYE_stage2_da3_ddt_g17.sh
```




<!-- # 🚀 Inference Demo
### Demo Script 
Download the released UniT weight from [google drive](https://drive.google.com/drive/folders/1JJOdEgM-rdu9GYGifG4kFxKF66JMeagR?usp=drive_link), and set the corresponding path in the demo [configuration file](run_configs/val/val_unit_demo.yaml). Then, run the script below to perform text-aware image restoration on low-quality image samples. The results will be saved in result_val/ by default.
```bash
bash run_scripts/val/run_val_unit_demo.sh
```


### Demo Result
Running the demo inference script will generate the following text restoration results. The visualized images are shown in the order: **Low-Quality image / Restored High-Quality image / Ground Truth image.** Checkout our [Project Page](https://cvlab-kaist.github.io/UniT/) for full text restoration qualitative results!
<p align="center">
  <img src="assets/demo_result/demo1.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_result/demo2.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_result/demo3.png" width="800">
</p>
<p align="center">
  <img src="assets/demo_result/demo4.png" width="800">
</p> -->



<!-- 
# 🔥 Training Recipe  

### 1. Data Preparation
- First, download the SA-Text training data from this [repo](https://github.com/cvlab-kaist/TAIR).

### 2. Model Weight Preparation
- Second, prepare the DiT and TSM model weights as described in section **👟Installation Walkthrough - 3.Model Pretrained Weights**.
- The model weights will be automatically downloaded into the **preset/models/** directory.



### 3. Training Script
Modify both the training bash script and the YAML configuration file before starting training.

- Stage 1 training bash file: [run_train_unit_stage1.sh](run_scripts/train/run_train_unit_stage1.sh)
- Stage 1 yaml config file: [train_unit_stage1.yaml](run_configs/train/train_unit_stage1.yaml)
```bash
# stage1 training 
bash run_scripts/train/JIHYE_train_stage1_dit4sr.sh
```


- Stage 2 training bash file: [run_train_unit_stage2.sh](run_scripts/train/run_train_unit_stage2.sh)
- Stage 2 yaml config file: [train_unit_stage2.yaml](run_configs/train/train_unit_stage2.yaml)
```bash
# stage2 training 
bash run_scripts/train/JIHYE_train_stage2_testr.sh
``` -->
<!-- 

## Citation

```
@article{kim2025unit,
  title={Unified Diffusion Transformer for High-fidelity Text-Aware Image Restoration},
  author={Kim, Jin Hyeon and Cho, Paul Hyunbin and Kim, Claire and Min, Jaewon and Lee, Jaeeun and Park, Jihye and Choi, Yeji and Kim, Seungryong},
  journal={arXiv preprint arXiv:2512.08922},
  year={2025},
}
```

## Acknowledgement
We thank the authors of [DiT4SR](https://github.com/Adam-duan/DiT4SR), [TAIR](https://github.com/cvlab-kaist/TAIR), and [TESTR](https://github.com/mlpc-ucsd/TESTR) for their excellent work and code, which served as the foundation for this project. -->