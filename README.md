# SuperLightNet-2D — Breast Ultrasound Segmentation

> **22AIE437 Medical Image Processing — Final Project**  
> *Adapting SuperLightNet for 2-D Breast Ultrasound Tumour Segmentation on BUSI*

---

## Overview

This repository contains the full implementation of **SuperLightNet-2D**, a lightweight (4.1 M parameters) segmentation network adapted from the original 3-D SuperLightNet architecture for 2-D breast ultrasound image segmentation on the **BUSI dataset**.

A **U-Net baseline** (31 M params) is included for direct comparison under identical training settings.

| Model | Params | Best Val DSC | Test DSC | Test HD95 |
|---|---|---|---|---|
| U-Net Baseline | ~31 M | 0.7768 | 0.7129 | 39.25 |
| **SuperLightNet-2D (Ours)** | **4.1 M** | **0.7206** | **0.6613** | **31.49** |

SuperLightNet-2D achieves **7.5× fewer parameters** and **lower HD95** (better boundary precision), at the cost of slightly lower DSC due to the small dataset size.

---

## Key Features

- **GHPA**: Grouped multi-axis Hadamard Product Attention — efficient O(HW) long-range attention
- **THPAEncFR2**: Encoder block with channel splitting + horizontal-flip symmetry
- **UpRepr2d**: Decoder with learnable skip-connection and residual scaling
- Pure PyTorch — no MONAI dependency
- Mixed-precision (FP16) training with AMP
- Full evaluation pipeline: DSC, IoU, HD95, Sensitivity, Specificity, Precision

---

## Repository Structure

```
BUSI_SuperLightNet/
├── networks/
│   ├── superlightnet_2d.py   # SuperLightNet-2D (GHPA + Encoder/Decoder)
│   └── egeunet.py            # EGE-UNet (reference implementation)
├── dataset.py                # BUSI data loader + albumentations augmentation
├── losses.py                 # BCEWithDiceLoss, FocalDiceLoss
├── metrics.py                # DSC, IoU, HD95, Sensitivity, Specificity, Precision
├── train.py                  # Full training script (AMP, cosine LR, best ckpt)
├── evaluate.py               # Test evaluation + 3-panel visualisation
├── split_dataset.py          # Stratified 70/15/15 train/val/test split
├── config.py                 # TrainConfig dataclass (all hyperparameters)
├── unet_baseline.ipynb       # U-Net baseline training + evaluation notebook
├── presentation.tex          # LaTeX Beamer presentation (15 slides)
├── presentation.bib          # BibTeX references
├── ieee_paper.tex            # IEEE-format LaTeX paper
├── generate_superlightnet_plots.py  # Generates all result plots
├── eval_output/
│   ├── busi_v1/results.txt           # SuperLightNet-2D per-image metrics
│   └── unet_baseline/results.txt     # U-Net per-image metrics
├── logs/
│   ├── busi_v1/train.log             # SuperLightNet-2D training log (200 epochs)
│   └── unet_baseline/train.log       # U-Net training log (200 epochs)
└── requirements.txt
```

---

## Setup

### 1. Create conda environment

```bash
conda create -n superlightnet2d python=3.10
conda activate superlightnet2d
pip install -r requirements.txt
```

### 2. Download the BUSI Dataset

Download from [Kaggle — Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and place it at:

```
BUSI_SuperLightNet/BUSI_Dataset/Dataset_BUSI_with_GT/
```

### 3. Split the dataset

```bash
python split_dataset.py \
    --data_dir BUSI_Dataset/Dataset_BUSI_with_GT \
    --output_dir BUSI_Dataset/Dataset_BUSI_with_GT \
    --train_ratio 0.70 --val_ratio 0.15
```

---

## Training

### SuperLightNet-2D

```bash
python train.py \
    --data_dir BUSI_Dataset/Dataset_BUSI_with_GT \
    --experiment busi_v1 \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-3 \
    --gpus 0
```

### U-Net Baseline

Open and run all cells in `unet_baseline.ipynb`.

---

## Evaluation

```bash
python evaluate.py \
    --data_dir BUSI_Dataset/Dataset_BUSI_with_GT \
    --checkpoint checkpoints/busi_v1/best_model.pth \
    --output_dir eval_output/busi_v1
```

---

## Results & Plots

Generate all result plots from training logs:

```bash
python generate_superlightnet_plots.py
```

This produces:
- `training_curves_superlightnet.png` — SuperLightNet-2D training curves
- `training_curves_comparison.png` — Side-by-side comparison with U-Net
- `sample_predictions_superlightnet.png` — 6-sample prediction grid
- `qualitative_comparison_unet_vs_superlightnet.png` — 4-column qualitative comparison

---

## Hardware

- **GPU**: NVIDIA GeForce RTX 3050 (6 GB VRAM)
- **Framework**: PyTorch 2.x + CUDA 13.0
- **Training time**: ~3.0 hours (200 epochs, SuperLightNet-2D)

---

## References

1. Yu et al., "SuperLightNet," arXiv 2025
2. Ronneberger et al., "U-Net," MICCAI 2015
3. Al-Dhabyani et al., "BUSI Dataset," Data in Brief 2020
4. Wu et al., "EGE-UNet," arXiv 2023

---

## Citation

If you use this code, please cite the original SuperLightNet paper and the BUSI dataset paper.
