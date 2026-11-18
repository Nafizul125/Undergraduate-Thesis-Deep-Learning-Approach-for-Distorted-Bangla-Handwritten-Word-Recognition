"""
generate_thesis_images.py
=========================
Generates professional-quality figures from the best pix2pix model
for use in a thesis paper.

Outputs (saved to thesis_figures/):
  1. individual/  — clean 3-panel strips  (Damaged | Generated | Ground Truth)
  2. grid_figure.png — multi-row grid of the best examples, suitable for a paper figure
  3. diversity_grid.png — variety showcase: easy / medium / hard samples

Usage:
    python generate_thesis_images.py
    python generate_thesis_images.py --checkpoint checkpoints/best_model.pth --n 50
"""

import os
import random
import argparse
import numpy as np
import torch
import yaml
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from pathlib import Path

from models import Pix2PixModel
from gpu_utils import GPUManager
from dataset import OnlineRestorationDataset
from damage_utils import make_damaged, prepare_crop, PRESETS

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH     = "config.yaml"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
OUT_DIR         = "thesis_figures"
NUM_SAMPLES     = 60        # evaluate this many; pick best subset for grids
GRID_ROWS       = 8         # rows in the main grid figure
SEED            = 42

PANEL_W  = 256
PANEL_H  = 256
GAP      = 6    # gap between panels
LABEL_H  = 28   # height of label strip above panels
METRIC_H = 22   # height of metric strip below panels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def try_load_font(size):
    candidates = [
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_LABEL  = try_load_font(14)
FONT_SMALL  = try_load_font(12)
FONT_TITLE  = try_load_font(15)


def denormalize(tensor):
    img = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def compute_ssim(gt, pred):
    if not SKIMAGE_OK:
        return float('nan')
    return ssim_fn(gt, pred, channel_axis=2, data_range=255)


def draw_centered_text(draw, text, x_left, y, width, font, color):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = len(text) * 7
    draw.text((x_left + (width - tw) // 2, y), text, font=font, fill=color)


# ---------------------------------------------------------------------------
# Single strip: [Damaged | Generated | Ground Truth]
# ---------------------------------------------------------------------------

def make_strip(damaged_np, generated_np, gt_np, ssim_val, show_labels=True):
    """
    Returns a PIL Image of one comparison strip with labels + metric.
    """
    total_w = PANEL_W * 3 + GAP * 2
    top_pad = LABEL_H if show_labels else 4
    total_h = top_pad + PANEL_H + METRIC_H

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    # Paste panels
    offsets = [0, PANEL_W + GAP, (PANEL_W + GAP) * 2]
    panels  = [damaged_np, generated_np, gt_np]
    labels  = ["(a) Damaged Input", "(b) Generated Output", "(c) Ground Truth"]
    label_colors = [(180, 60, 60), (40, 120, 200), (40, 160, 80)]

    for i, (panel, label, color) in enumerate(zip(panels, labels, label_colors)):
        x = offsets[i]
        canvas.paste(Image.fromarray(panel), (x, top_pad))

        if show_labels:
            draw_centered_text(draw, label, x, 5, PANEL_W, FONT_LABEL, color)

        # thin border around each panel
        draw.rectangle(
            [x, top_pad, x + PANEL_W - 1, top_pad + PANEL_H - 1],
            outline=(200, 200, 200), width=1
        )

    # Divider line
    draw.line([(0, top_pad + PANEL_H), (total_w, top_pad + PANEL_H)],
              fill=(210, 210, 210), width=1)

    # Metric row
    if SKIMAGE_OK and not np.isnan(ssim_val):
        metric_txt = f"SSIM = {ssim_val:.4f}"
        quality    = "Excellent" if ssim_val >= 0.90 else ("Good" if ssim_val >= 0.80 else "Fair")
        full_txt   = f"{metric_txt}   [{quality}]"
    else:
        full_txt = ""
    draw_centered_text(draw, full_txt, 0, top_pad + PANEL_H + 4, total_w, FONT_SMALL, (100, 100, 100))

    return canvas


# ---------------------------------------------------------------------------
# Multi-row grid
# ---------------------------------------------------------------------------

def make_grid(samples, title="Pix2Pix Bangla Handwriting Restoration Results",
              cols=3, show_labels_on_first_row=True):
    """
    samples: list of dicts with keys damaged, generated, gt, ssim
    Arranges them in a grid of (N // cols) rows.
    """
    strip_w = PANEL_W * 3 + GAP * 2
    strip_h = LABEL_H + PANEL_H + METRIC_H

    rows      = (len(samples) + cols - 1) // cols
    title_h   = 44
    row_gap   = 12
    total_w   = strip_w * cols + GAP * (cols - 1)
    total_h   = title_h + rows * strip_h + (rows - 1) * row_gap + 12

    canvas = Image.new("RGB", (total_w, total_h), color=(248, 248, 248))
    draw   = ImageDraw.Draw(canvas)

    # Title
    draw_centered_text(draw, title, 0, 12, total_w, FONT_TITLE, (30, 30, 30))
    draw.line([(20, title_h - 6), (total_w - 20, title_h - 6)], fill=(180, 180, 180), width=1)

    for idx, s in enumerate(samples):
        row = idx // cols
        col = idx  % cols
        show_lbl = (row == 0) if show_labels_on_first_row else False
        strip    = make_strip(s['damaged'], s['generated'], s['gt'], s['ssim'],
                               show_labels=show_lbl)
        x = col * (strip_w + GAP)
        y = title_h + row * (strip_h + row_gap)
        canvas.paste(strip, (x, y))

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=CONFIG_PATH)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--n",          type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--out",        default=OUT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Setup ──────────────────────────────────────────────────────────────
    gpu    = GPUManager(verbose=False)
    device = gpu.device
    print(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = Pix2PixModel(config).to(device)
    ck    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.generator.load_state_dict(ck['generator_state_dict'])
    model.generator.eval()
    print(f"Loaded checkpoint (epoch {ck.get('epoch','?')}, val_L1={ck.get('best_val_loss',0):.4f})")

    ds_cfg        = config['dataset']
    image_size    = ds_cfg.get('image_size', 256)
    damage_preset = ds_cfg.get('damage_preset', 'distorted')

    # Load test split (no augmentation)
    dataset = OnlineRestorationDataset(
        root_dir=ds_cfg['crops_dir'],
        split='test',
        image_size=image_size,
        damage_preset=damage_preset,
        augment=False,
    )
    print(f"Test samples: {len(dataset)}")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # ── Collect samples ────────────────────────────────────────────────────
    n      = min(args.n, len(dataset))
    idxs   = random.sample(range(len(dataset)), n)

    records = []
    print(f"Generating {n} samples ...")
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            batch   = dataset[idx]
            a_t     = batch['A'].unsqueeze(0).to(device)
            b_t     = batch['B'].unsqueeze(0).to(device)
            fake_t  = model.generator(a_t)

            dmg_np  = denormalize(a_t)
            gt_np   = denormalize(b_t)
            gen_np  = denormalize(fake_t)
            sv      = compute_ssim(gt_np, gen_np)

            records.append({
                'idx':       idx,
                'filename':  batch['filename'],
                'damaged':   dmg_np,
                'generated': gen_np,
                'gt':        gt_np,
                'ssim':      sv,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n} done …")

    valid = [r for r in records if not np.isnan(r['ssim'])]
    valid.sort(key=lambda x: x['ssim'], reverse=True)
    mean_ssim = np.mean([r['ssim'] for r in valid])
    print(f"\nMean SSIM over {len(valid)} samples: {mean_ssim:.4f}")

    # ── Output dirs ────────────────────────────────────────────────────────
    out     = Path(args.out)
    ind_dir = out / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Save all individual strips ──────────────────────────────────────
    print("\nSaving individual strips …")
    for r in valid:
        strip = make_strip(r['damaged'], r['generated'], r['gt'], r['ssim'])
        fname = f"ssim{r['ssim']:.4f}_{Path(r['filename']).stem}.png"
        strip.save(ind_dir / fname)

    print(f"  Saved {len(valid)} strips to {ind_dir}")

    # ── 2. Main thesis grid: top GRID_ROWS×3 samples ───────────────────────
    top_n   = GRID_ROWS * 3
    top_set = valid[:top_n]
    grid    = make_grid(
        top_set,
        title="Pix2Pix Bangla Handwriting Restoration - Best Results",
        cols=3,
    )
    grid_path = out / "grid_best.png"
    grid.save(grid_path)
    print(f"\nBest-results grid ({top_n} samples) saved to {grid_path}")

    # ── 3. Diversity grid: easy / medium / hard ─────────────────────────────
    n_each   = 6
    best_set = valid[:n_each]
    n_mid    = len(valid) // 2
    mid_set  = valid[n_mid - n_each//2 : n_mid + n_each//2]
    hard_set = valid[-n_each:]
    diverse  = best_set + mid_set + hard_set

    div_grid = make_grid(
        diverse,
        title="Pix2Pix Restoration - Easy / Medium / Hard Examples",
        cols=3,
    )
    div_path = out / "grid_diversity.png"
    div_grid.save(div_path)
    print(f"Diversity grid  ({len(diverse)} samples) saved to {div_path}")

    # ── 4. Single-column showcase (6 best) for a tight paper figure ─────────
    showcase_samples = valid[:6]
    sc_strip_w = PANEL_W * 3 + GAP * 2
    sc_strip_h = LABEL_H + PANEL_H + METRIC_H
    showcase_h = sc_strip_h * 6 + 10 * 5 + 50
    showcase   = Image.new("RGB", (sc_strip_w, showcase_h), (248, 248, 248))
    sc_draw    = ImageDraw.Draw(showcase)
    draw_centered_text(sc_draw,
                       "Bangla Handwriting Restoration - Top 6 Results",
                       0, 10, sc_strip_w, FONT_TITLE, (30, 30, 30))
    sc_draw.line([(10, 38), (sc_strip_w - 10, 38)], fill=(180, 180, 180), width=1)

    for i, r in enumerate(showcase_samples):
        strip = make_strip(r['damaged'], r['generated'], r['gt'], r['ssim'],
                           show_labels=(i == 0))
        showcase.paste(strip, (0, 44 + i * (sc_strip_h + 10)))

    sc_path = out / "showcase_top6.png"
    showcase.save(sc_path)
    print(f"Top-6 showcase saved to {sc_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"""
===========================================================
  THESIS FIGURES GENERATED
  Output folder : {out.resolve()}

  individual/         {len(valid)} comparison strips (sorted by SSIM)
  grid_best.png       {top_n}-sample grid (best results)
  grid_diversity.png  easy/medium/hard showcase
  showcase_top6.png   single-column top-6 figure

  Mean SSIM : {mean_ssim:.4f}
  Best SSIM : {valid[0]['ssim']:.4f}   ({valid[0]['filename']})
  Worst SSIM: {valid[-1]['ssim']:.4f}  ({valid[-1]['filename']})
===========================================================
""")


if __name__ == "__main__":
    main()
