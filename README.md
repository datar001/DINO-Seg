# DINO-Seg

Code for **DINO-Seg**, an improved DINOv3-based semantic segmentation method for UAV imagery of traditional villages.

This repository contains the core training, evaluation, and inference scripts used in the project. The method is designed for fine-grained village element segmentation from aerial images.

## Overview

DINO-Seg focuses on semantic segmentation of traditional village UAV orthophotos. In the current codebase, the segmentation labels are defined in `dataset.py` as:

- `unlabelled`
- `TR`
- `NBU`
- `OBU`
- `WA`

The manuscript evaluates the method on a traditional village benchmark and reports strong performance on village element segmentation.

## Repository Structure

The current public GitHub repository contains the following main files:

```text
.
├── color_mask.py
├── dataset.py
├── linear_regression.py
├── model.py
├── predict_without_gt.py
├── seg_12_village.py
├── test.py
├── train_baseline.py
└── trainer.py
```

File summary:

- `dataset.py`: dataset definition and data augmentation
- `model.py`: PyTorch Lightning segmentation model wrapper
- `train_baseline.py`: baseline model training script
- `trainer.py`: DINO-Seg training script
- `test.py`: test-set evaluation and predicted mask export
- `color_mask.py`: generate color masks and overlay visualizations
- `predict_without_gt.py`: batch prediction for unlabeled images
- `seg_12_village.py`: segmentation script for village-scale image folders
- `linear_regression.py`: class-wise regression analysis between predictions and labels

## Dataset

The dataset will be released on Hugging Face.

Dataset link:

- [Hugging Face Dataset Placeholder](https://huggingface.co/datasets/datar001/DINO-Seg)

If the page is not available yet, it means the dataset upload is still in progress.

### Expected Directory Structure

The training and evaluation scripts expect a dataset structure like this:

```text
dataset/
├── train/
│   ├── images/
│   └── mask/
├── val/
│   ├── images/
│   └── mask/
└── test/
    ├── images/
    └── mask/
```

## Requirements

Recommended environment:

- Python 3.10
- CUDA-enabled GPU

Main dependencies:

- `torch`
- `torchvision`
- `pytorch_lightning`
- `segmentation_models_pytorch`
- `albumentations`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `Pillow`
- `tqdm`
- `pydensecrf`

## Training

### Baseline Training

After editing the dataset path in `train_baseline.py`, run:

```bash
python train_baseline.py
```

The default baseline setting in the current script is:

- architecture: `fpn`
- encoder: `resnet34`

### DINO-Seg Training

After editing the dataset path in `trainer.py`, run:

```bash
python trainer.py
```

The default DINO-Seg setting in the current script is:

- architecture: `dpt_simple`
- encoder: `dinov3_encoder`

Training outputs are saved under `lightning_logs/`.

## Evaluation

After editing the checkpoint path and dataset path in `test.py`, run:

```bash
python test.py
```

This script:

- loads a trained checkpoint
- evaluates the test set
- exports predicted masks
- saves metrics to `metric.json`

## Visualization

After generating predicted masks, you can create color masks and overlay results with:

```bash
python color_mask.py
```

Please edit the image and mask directory paths in the script before running.

## Inference on Unlabeled Data

For batch inference on unlabeled village images, see:

- `predict_without_gt.py`
- `seg_12_village.py`

These scripts are intended for project-specific inference workflows and also require manual path adjustment.
