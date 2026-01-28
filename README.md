# Attention_AI

Fine-tuning Vision Transformer (ViT) on ImageWoof Dataset with Optional LoRA

## Overview

This project implements fine-tuning of a pre-trained Vision Transformer (ViT) model on the ImageWoof dataset with support for Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique.

## Objective

The script trains a `vit_tiny_patch16_224` model (from the timm library) to classify images in the ImageWoof dataset. It supports two training modes:

1. **Full Fine-Tuning**: All model parameters are updated during training
2. **LoRA Fine-Tuning**: Only a small subset of parameters are trained using Low-Rank Adaptation, significantly reducing the number of trainable parameters while maintaining performance

## Features

- **Pre-trained Vision Transformer**: Uses `vit_tiny_patch16_224` from the timm library
- **LoRA Support**: Implements Low-Rank Adaptation for efficient fine-tuning
- **ImageWoof Dataset**: Trains on a subset of ImageNet containing 10 dog breeds
- **Best Model Checkpointing**: Automatically saves the best model during training
- **GPU Support**: Automatically detects and uses CUDA if available


## Usage

### Basic Training (Full Fine-Tuning)

```bash
python src/main.py --datapath /path/to/data --batch_size 128 --num_epochs 5
```

### LoRA Fine-Tuning

```bash
python src/main.py --datapath /path/to/data --batch_size 128 --num_epochs 5 --lora
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datapath` | str | `/projects/ec232/data/` | Path to data directory |
| `--batch_size` | int | 128 | Batch size for training |
| `--img_size` | int | [224, 224] | Input image dimensions |
| `--num_classes` | int | 10 | Number of output classes |
| `--num_epochs` | int | 5 | Number of training epochs |
| `--learning_rate` | float | 1e-4 | Learning rate for optimizer |
| `--lora` | flag | False | Enable LoRA fine-tuning |

## LoRA Implementation

When LoRA is enabled:
- Only the layer normalization and classification head are fully trainable
- Query, Key, Value (QKV) projection and attention projection layers are replaced with LoRAWrapper
- LoRA rank is set to 12
- Significantly reduces trainable parameters while maintaining model performance

## Model Architecture

- **Base Model**: Vision Transformer (ViT) Tiny - Patch size 16, Image size 224x224
- **LoRA Modifications** (when enabled):
  - Applied to `qkv` and `proj` layers in attention blocks
  - Rank: 12

## Output

The script will:
1. Train the model and display progress with tqdm
2. Save the best model checkpoint as `model_imagewoof`
3. Evaluate on the validation set
4. Save the final model as either `lora_model.pth` or `full_model.pth`



