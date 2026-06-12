# MRI-to-CT Image Translation using GANs

**Integration of MRI Information into CT Imaging through Advanced Image Translation Techniques**

A deep-learning project that synthesizes Computed Tomography (CT) images from Magnetic Resonance Imaging (MRI) scans using Generative Adversarial Networks (GANs). The goal is to combine the soft-tissue contrast of MRI with the high spatial / bone detail of CT, producing unified images with greater diagnostic value, without exposing patients to additional ionizing radiation.

The project implements and compares **three progressively enhanced GAN architectures**, each in its own notebook.

---

## Table of Contents

- [Overview](#overview)
- [Architectures](#architectures)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)


---

## Overview

Each model is a **pix2pix-style conditional GAN** that learns a mapping from a single-channel MRI slice to the corresponding CT slice:

- **Generator** — a U-Net (encoder–decoder with skip connections) that takes a `1 × 256 × 256` MRI slice and outputs a `1 × 256 × 256` CT slice.
- **Discriminator** — a convolutional classifier with global average pooling that scores how realistic an image is.
- **Training** — generator and discriminator are trained adversarially; the generator also minimizes a reconstruction / perceptual loss against the ground-truth CT.

The three versions differ only in the generator's attention and in the loss function used, allowing a clean ablation of each addition.

---

## Architectures

| Version | Notebook | Generator | Loss | Idea |
|--------|----------|-----------|------|------|
| **v1** | `v1_Unet.ipynb` | Standard U-Net | Adversarial (BCE) + L1 pixel loss (`λ = 100`) | Baseline GAN |
| **v2** | `v2_AttentionUnet.ipynb` | Attention U-Net | Adversarial (BCE) + L1 pixel loss | Attention gates on skip connections to focus on relevant regions |
| **v3** | `v3_AttentionUnet_VggLoss.ipynb` | Attention U-Net | Adversarial (BCE) + VGG-19 perceptual (content) loss (`λ = 1`) | Perceptual loss for structural / semantic fidelity |

**Key implementation details**

- **U-Net encoder:** 6 down-sampling blocks (`Conv2d → InstanceNorm → LeakyReLU`), channels `64 → 128 → 256 → 512 → 512 → 512`, dropout on the deeper blocks.
- **U-Net decoder:** transposed-conv up-sampling blocks with skip connections, ending in `Upsample → Conv2d → Tanh`.
- **Attention block (v2, v3):** an additive attention gate (`W_g`, `W_x`, `psi` with a sigmoid mask) applied to each skip connection before concatenation, following the Attention U-Net formulation.
- **VGG loss (v3):** a frozen, pretrained `vgg19` truncated at `features[:36]` (classification head removed). Generated and real CT slices are repeated to 3 channels and compared via L1 distance on the extracted feature maps.
- **Discriminator:** 4 strided conv blocks (`Conv2d → BatchNorm → LeakyReLU`, `64 → 128 → 256 → 512`) → adaptive average pool → `1 × 1` conv → sigmoid.
- **Optimizer:** Adam, `lr = 0.0002`, `betas = (0.5, 0.999)` for both networks.
- **Default training config:** `num_epochs = 50`, `batch_size = 4`, image size `256 × 256`, single channel.

---

## Repository Structure

```
.
├── v1_Unet.ipynb                  # Architecture 1: baseline U-Net GAN
├── v2_AttentionUnet.ipynb         # Architecture 2: Attention U-Net GAN
├── v3_AttentionUnet_VggLoss.ipynb # Architecture 3: Attention U-Net + VGG perceptual loss
├── dataprocess.py                 # Helper module (see note below)
├── archive/
│   └── data(processed)/
│       ├── train_input.npy        # MRI training slices  (570, 256, 256)
│       ├── train_output.npy       # CT  training slices  (570, 256, 256)
│       ├── test_input.npy         # MRI test slices      (150, 256, 256)
│       └── test_output.npy        # CT  test slices       (150, 256, 256)
└── README.md
```

> **Note on `dataprocess.py`:** The notebooks import helper functions from a local module:
> - `from dataprocess import process_gen, vprocess_gen`  (v1)
> - `from dataprocess import process, vprocess`           (v3)
>
> These functions post-process the generator output before metric computation / visualization (e.g. rescaling generated values to match the ground-truth CT range). Make sure the project's own `dataprocess.py` containing these functions is present in the repo root. The version of `dataprocess.py` currently in this folder is **not** the project module and must be replaced with the correct one before the notebooks will run.

---

## Dataset

- **Format:** Paired MRI/CT slices stored as NumPy `.npy` arrays.
- **Split:** 570 training pairs and 150 test pairs.
- **Shape:** Each slice is `256 × 256`, single channel (grayscale).
- **Preprocessing:** Intensities normalized (zero mean / unit variance), all slices resized to `256 × 256`.

Place the four `.npy` files under `archive/data(processed)/` as shown above, or update the load paths at the top of each notebook.

---

## Requirements

- Python 3.8+
- PyTorch 1.8+ and TorchVision
- NumPy
- Matplotlib
- scikit-image (for SSIM)
- torchviz *(optional — architecture diagrams)*
- tensorboard *(optional — graph visualization)*

Install with:

```bash
pip install torch torchvision numpy matplotlib scikit-image torchviz tensorboard
```

A CUDA-capable NVIDIA GPU is strongly recommended for training. 16 GB RAM minimum (32 GB+ preferred); SSD storage recommended for fast data loading.

---

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision numpy matplotlib scikit-image torchviz tensorboard

# 4. Add the dataset
#    Place the .npy files under archive/data(processed)/

# 5. Make sure the project dataprocess.py (with process_gen / vprocess_gen / process / vprocess) is in the repo root
```

---

## Usage

Open any of the notebooks in Jupyter and run the cells top to bottom:

```bash
jupyter notebook v1_Unet.ipynb
```

Each notebook follows the same flow:

1. **Load data** — reads the `.npy` arrays and visualizes sample MRI/CT pairs.
2. **Define models** — builds the generator and discriminator.
3. **Configure loss & optimizers.**
4. **Train** — runs the adversarial training loop for `num_epochs`, then saves the weights with `torch.save(...)` (e.g. `v2generator.pth`, `v2discriminator.pth`).
5. **Load & evaluate** — reloads the generator and computes all metrics on the test set.
6. **Visualize predictions** — displays MRI input, real CT, and generated CT side by side.

Adjust `num_epochs`, `batch_size`, and the loss weight (`lambda_pixel` in v1/v2, `lambda_content` in v3) at the top of the training cell to experiment.

---

## Evaluation Metrics

Generated CT images are compared against the ground truth using:

| Metric | Meaning | Better |
|--------|---------|--------|
| **ME**  | Mean Error (intensity bias) | closer to 0 |
| **MAE** | Mean Absolute Error | lower |
| **MSE** | Mean Square Error | lower |
| **RMSE**| Root Mean Square Error | lower |
| **PSNR**| Peak Signal-to-Noise Ratio | higher |
| **SSIM**| Structural Similarity Index | higher |
| **NCC** | Normalized Cross-Correlation | closer to 1 |
| **SI**  | Similarity Index | closer to 1 |

---

## Results

Test-set performance across the three architectures:

| Metric | Architecture 1 (U-Net) | Architecture 2 (Attention U-Net) | Architecture 3 (Attention + VGG) |
|--------|-----------------------:|---------------------------------:|---------------------------------:|
| ME    | 0.000101 | -0.077070 | -0.000024 |
| MAE   | 0.159544 |  0.092102 |  0.079790 |
| MSE   | 0.039980 |  0.025528 |  0.009999 |
| RMSE  | 0.199950 |  0.139206 |  0.099997 |
| PSNR  | 13.98 | 18.58 | **20.00** |
| SSIM  | 0.1276 | **0.7050** | 0.2407 |
| NCC   | 0.8168 | 0.9167 | **0.9427** |
| SI    | 0.7776 | 0.8255 | **0.9392** |

**Takeaways**

- **Architecture 1** is a reasonable baseline but preserves structure poorly (low SSIM/PSNR).
- **Architecture 2** improves nearly every metric — attention notably boosts SSIM and overall image quality.
- **Architecture 3** achieves the best PSNR, NCC, and SI (highest correlation/similarity to target), at the cost of a lower SSIM than v2. Overall, adding attention and a VGG perceptual loss gives the most faithful translations.

---
