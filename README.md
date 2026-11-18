# Deep Learning Approach for Distorted Bangla Handwritten Word Recognition

**B.Sc. CSE Thesis — BRAC University**

A three-stage pipeline for restoring distorted Bangla handwritten word images using a Pix2Pix conditional GAN, enabling downstream OCR on degraded historical and aged documents.

---

## Overview

This repository contains the **Pix2Pix restoration stage** — the third and final stage of a pipeline:

```
Stage 1: YOLO word detector  →  Stage 2: TrOCR recogniser  →  Stage 3: Pix2Pix restoration (this repo)
```

The key innovation is **online damage augmentation**: instead of training on a static set of pre-degraded images, fresh random damage is applied to each clean word crop at every epoch. This prevents the model from memorising specific degradation patterns and substantially improves generalisation.

### Best checkpoint results (test set, 2,640 samples)

| Metric | Value |
|--------|-------|
| PSNR (mean) | 26.62 dB |
| SSIM (median) | 0.9016 |
| Val L1 (best epoch) | 0.0389 (epoch 57) |
| Trained for | 75 epochs |

---

## Six Synthetic Damage Types

The `damage_utils.py` module implements six parametric degradation functions, applied online during training:

| # | Damage type | Parameters |
|---|-------------|------------|
| 1 | Gaussian noise | σ = 20 |
| 2 | Heavy Gaussian blur | kernel 7–9 px |
| 3 | Ink fading | factor 0.52–0.68 |
| 4 | Morphological erosion | 2 px elliptic kernel |
| 5 | Water stains | 2–3 blobs, radius 8–16 px |
| 6 | Paper yellowing | strength 0.15–0.25 |

Application order: erosion → blur → fade → noise → stains → yellowing.

---

## Repository Structure

```
├── pipeline.py           # End-to-end three-stage pipeline (detection → TrOCR → Pix2Pix)
├── train.py              # Main training loop (adversarial + L1 + VGG-19 perceptual)
├── models.py             # U-Net generator (7-block, 59.5M params) + PatchGAN discriminator
├── dataset.py            # Dataset loader with online damage augmentation
├── damage_utils.py       # Six-type parametric damage pipeline
├── evaluate.py           # PSNR / SSIM evaluation on held-out test set
├── inference.py          # Single-image inference with a saved checkpoint
├── gpu_utils.py          # Mixed-precision and GPU memory utilities
├── crop_words.py         # Crop word regions from full document images
├── verify_setup.py       # Environment and data sanity check
├── config.yaml           # Enhanced model config (λ_L1=80, λ_perc=10)
├── config_quick_test.yaml# 1-epoch smoke-test config
├── requirements.txt      # Python dependencies
└── scripts/              # Thesis figure generation scripts
    ├── gen_charts.py       # PSNR bar, loss family, λ sweep, dataset pie
    ├── gen_damage_fig.py   # Damage-type grid figure
    ├── gen_pivot.py        # Three-stage pipeline diagram
    └── gen_progression.py  # Development journey figure
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+, PyTorch 2.0+, CUDA recommended.

### Data

The model trains on clean Bangla handwritten word crops (26,391 images).  
Dataset split (seed 42, 80/10/10): **21,112 train / 2,639 val / 2,640 test**.

Place your word crop images in `data_raw/word_crops/`.

### Model Weights

The best checkpoint (`best_model.pth`, epoch 57, ~689 MB) is hosted separately:

> **[Download best_model.pth — Google Drive / Hugging Face]** *(link to be added)*

Place the downloaded file at `checkpoints/best_model.pth`.

---

## End-to-End Pipeline

`pipeline.py` chains all three stages into a single script:

```
Stage 1 — Detection    : OpenCV morphological word segmentation
                         (swap in YOLO weights from Phase 2 for production use)
Stage 2 — Recognition  : microsoft/trocr-base-handwritten (locally cached, ~1.3 GB)
                         NOTE: English base model — Bangla fine-tuning = future work
Stage 3 — Restoration  : Pix2Pix best_model.pth (epoch 57, PSNR 26.62 dB)
```

```bash
# Demo: 6 BanglaWriting words, apply damage, run pipeline, save results
python pipeline.py --demo --output results/demo/

# Single damaged word image
python pipeline.py --word path/to/word.png --output results/

# Full handwritten page (detection → recognition → restoration)
python pipeline.py --page path/to/page.jpg --output results/

# Skip TrOCR to save load time
python pipeline.py --page path/to/page.jpg --output results/ --no-recogniser
```

The pipeline outputs: individual restored crops, a `*_grid.png` side-by-side summary, and an annotated page with detected word boxes.

---

## Training

```bash
# Full training run (enhanced model config)
python train.py --config config.yaml

# Quick smoke test (1 epoch)
python train.py --config config_quick_test.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/best_model.pth
```

Key hyperparameters (`config.yaml`):

```yaml
training:
  lambda_l1: 80
  lambda_gan: 2.0
  lambda_perceptual: 10   # VGG-19 perceptual loss
  num_epochs: 75
  learning_rate_gen: 0.0001
dataset:
  online_damage: true     # fresh random damage every epoch
  damage_preset: "distorted"
```

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --config config.yaml
```

Reports per-sample PSNR and SSIM on the held-out test split, and saves a summary JSON.

---

## Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --input path/to/damaged_word.png \
                    --output restored.png
```

---

## Ablation Study

Fifteen configurations were tested on the static Kaggle dataset before the enhanced model. The best ablation run (λ_L1=150, val PSNR=20.78 dB) revealed that static data caused overfitting that made auxiliary losses harmful — motivating the switch to online augmentation. Full ablation results are reported in the thesis (Chapter 5, Table 4.2).

---

## Citation

If you use this code or the dataset in your work, please cite:

```bibtex
@thesis{nassah2026bangla,
  title={Deep Learning Approach for Distorted Bangla Handwritten Word Recognition},
  author={Hassan, Mehedi},
  school={BRAC University},
  year={2026},
  type={B.Sc. Thesis}
}
```

---

## License

MIT License. The model weights and word-crop dataset are provided for academic use only.
