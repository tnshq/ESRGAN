# ESRGAN
ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)

This project implements an ESRGAN model for enhancing image resolution using adversarial training. It includes the following features:

Dataset Handling: Downloads and processes the DIV2K dataset for high-resolution and low-resolution image training.
Model Components:
Generator: Incorporates Residual-in-Residual Dense Blocks (RRDB) and upscaling layers for generating high-resolution images.
Discriminator: Differentiates between real and generated high-resolution images.
Loss Functions: Combines VGG-based perceptual loss, L1 loss, and adversarial loss for effective training.
Training Pipeline:
Implements gradient penalty for stable discriminator training.
Uses mixed precision training with PyTorch AMP for improved performance.
Data Augmentation: Uses Albumentations for cropping, flipping, and random rotations.
Visualization: Includes scripts for visualizing low-resolution and high-resolution outputs.
The project aims to generate visually appealing high-resolution images from low-resolution inputs, making it suitable for tasks like image restoration and enhancement.
