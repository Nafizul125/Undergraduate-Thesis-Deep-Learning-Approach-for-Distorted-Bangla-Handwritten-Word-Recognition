"""
Pix2Pix Evaluation — Bangla Handwritten Word Restoration
=========================================================
Computes SSIM and PSNR on the test set and saves visual comparisons.

Usage:
    python evaluate.py --config config.yaml --checkpoint checkpoints/checkpoint_epoch_010.pth
    python evaluate.py --config config.yaml --checkpoint checkpoints/checkpoint_epoch_010.pth --output results/
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# scikit-image for standard image quality metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARN] scikit-image not found. Install with: pip install scikit-image")
    print("       Falling back to numpy-based PSNR only.")

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import Pix2PixModel
from gpu_utils import GPUManager
from dataset import OnlineRestorationDataset, Pix2PixDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = Pix2PixModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.generator.eval()
    return model, config


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor from [-1, 1] to uint8 numpy image."""
    img = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def compute_psnr_numpy(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Fallback PSNR when scikit-image is unavailable."""
    mse = np.mean((img_true.astype(float) - img_pred.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_metrics(img_true: np.ndarray, img_pred: np.ndarray) -> dict:
    """Compute SSIM and PSNR between two uint8 RGB images."""
    if SKIMAGE_AVAILABLE:
        ssim_val = ssim(img_true, img_pred,
                        channel_axis=2,          # RGB
                        data_range=255)
        psnr_val = psnr(img_true, img_pred, data_range=255)
    else:
        ssim_val = float('nan')
        psnr_val = compute_psnr_numpy(img_true, img_pred)
    return {'ssim': ssim_val, 'psnr': psnr_val}


def save_comparison(img_input: np.ndarray,
                    img_generated: np.ndarray,
                    img_target: np.ndarray,
                    path: str,
                    metrics: dict):
    """Save a 3-panel side-by-side comparison with labels: Damaged | Generated | Ground Truth."""
    from PIL import ImageDraw
    h, w = img_input.shape[:2]

    # Extra vertical space: 22px for panel labels + 22px for metrics
    footer_h = 44
    canvas = Image.new('RGB', (w * 3, h + footer_h), color=(240, 240, 240))

    canvas.paste(Image.fromarray(img_input),     (0,     0))
    canvas.paste(Image.fromarray(img_generated), (w,     0))
    canvas.paste(Image.fromarray(img_target),    (w * 2, 0))

    draw = ImageDraw.Draw(canvas)

    # Draw thin separator line between image and footer
    draw.line([(0, h), (w * 3, h)], fill=(180, 180, 180), width=1)

    # Panel labels — centred under each panel
    panel_labels = ["Damaged", "Generated", "Ground Truth"]
    for i, label in enumerate(panel_labels):
        try:
            bbox = draw.textbbox((0, 0), label)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = len(label) * 6   # fallback for older Pillow
        x = i * w + (w - text_w) // 2
        draw.text((x, h + 6), label, fill=(40, 40, 40))

    # Metrics row — centred across full width
    ssim_str = f"SSIM={metrics['ssim']:.4f}" if not np.isnan(metrics['ssim']) else "SSIM=N/A"
    psnr_str = f"PSNR={metrics['psnr']:.2f}dB"
    metric_text = f"{ssim_str}   {psnr_str}"
    try:
        bbox = draw.textbbox((0, 0), metric_text)
        text_w = bbox[2] - bbox[0]
    except AttributeError:
        text_w = len(metric_text) * 6
    x = (w * 3 - text_w) // 2
    draw.text((x, h + 26), metric_text, fill=(80, 80, 80))

    canvas.save(path)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(config_path: str,
             checkpoint_path: str,
             output_dir: str,
             num_comparisons: int = 20,
             device_str: str = None):

    # ---- Setup -------------------------------------------------------
    gpu = GPUManager(verbose=True)
    device = torch.device(device_str) if device_str else gpu.device

    model, config = load_model(config_path, checkpoint_path, device)

    ds_cfg     = config['dataset']
    online     = ds_cfg.get('online_damage', False)
    image_size = ds_cfg.get('image_size', 256)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    comp_dir = os.path.join(output_dir, 'comparisons')
    Path(comp_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build test dataset -----------------------------------------
    if online:
        test_dataset = OnlineRestorationDataset(
            root_dir=ds_cfg['crops_dir'],
            split='test',
            image_size=image_size,
            damage_preset=ds_cfg.get('damage_preset', 'distorted'),
        )
        data_source = 'online (fresh damage per sample)'
    else:
        test_dir = ds_cfg['test_dir']
        test_dataset = Pix2PixDataset(
            root_dir=test_dir,
            image_size=image_size,
            augment=False,
        )
        data_source = test_dir

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    n_test = len(test_dataset)

    print(f"\n{'='*65}")
    print(f"  Evaluating on {n_test} test samples")
    print(f"  Mode       : {data_source}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Output     : {output_dir}")
    print(f"{'='*65}\n")

    if n_test == 0:
        print("[ERROR] No test samples found.")
        return

    # ---- Inference + metrics -----------------------------------------
    all_ssim, all_psnr = [], []
    save_count = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            a_tensor = batch['A'].to(device)   # damaged
            b_tensor = batch['B'].to(device)   # clean ground truth
            filename = batch['filename'][0] if 'filename' in batch else f"sample_{i:05d}.png"

            fake_b_tensor = model.generator(a_tensor)

            # Convert to uint8 numpy
            a_uint8    = denormalize(a_tensor)
            b_uint8    = denormalize(b_tensor)
            fake_uint8 = denormalize(fake_b_tensor)

            # Metrics: generated vs ground truth
            m = compute_metrics(b_uint8, fake_uint8)
            all_ssim.append(m['ssim'])
            all_psnr.append(m['psnr'])

            # Save comparison images for the first N samples
            if save_count < num_comparisons:
                comp_path = os.path.join(comp_dir, f'cmp_{os.path.basename(filename)}')
                save_comparison(a_uint8, fake_uint8, b_uint8, comp_path, m)
                save_count += 1

    # ---- Aggregate report --------------------------------------------
    valid_ssim = [v for v in all_ssim if not np.isnan(v)]
    valid_psnr = [v for v in all_psnr if not np.isfinite(v) is False]  # exclude inf

    report = {
        'checkpoint': checkpoint_path,
        'data_source': data_source,
        'num_samples': n_test,
        'ssim': {
            'mean':   float(np.mean(valid_ssim)) if valid_ssim else None,
            'median': float(np.median(valid_ssim)) if valid_ssim else None,
            'std':    float(np.std(valid_ssim)) if valid_ssim else None,
            'min':    float(np.min(valid_ssim)) if valid_ssim else None,
            'max':    float(np.max(valid_ssim)) if valid_ssim else None,
        },
        'psnr_db': {
            'mean':   float(np.mean(valid_psnr)) if valid_psnr else None,
            'median': float(np.median(valid_psnr)) if valid_psnr else None,
            'std':    float(np.std(valid_psnr)) if valid_psnr else None,
            'min':    float(np.min(valid_psnr)) if valid_psnr else None,
            'max':    float(np.max(valid_psnr)) if valid_psnr else None,
        }
    }

    report_path = os.path.join(output_dir, 'metrics_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ---- Pretty-print summary ----------------------------------------
    print(f"\n{'='*65}")
    print("  EVALUATION RESULTS")
    print(f"{'='*65}")
    print(f"  Samples evaluated : {n_test}")
    print(f"  Comparisons saved : {save_count}  →  {comp_dir}")
    print()
    if valid_ssim:
        print(f"  SSIM  (higher is better, max=1.0)")
        print(f"    Mean   : {report['ssim']['mean']:.4f}")
        print(f"    Median : {report['ssim']['median']:.4f}")
        print(f"    Std    : {report['ssim']['std']:.4f}")
        print(f"    Range  : [{report['ssim']['min']:.4f}, {report['ssim']['max']:.4f}]")
    else:
        print("  SSIM: N/A (install scikit-image)")
    print()
    print(f"  PSNR  (higher is better, unit: dB)")
    print(f"    Mean   : {report['psnr_db']['mean']:.2f} dB")
    print(f"    Median : {report['psnr_db']['median']:.2f} dB")
    print(f"    Std    : {report['psnr_db']['std']:.2f} dB")
    print(f"    Range  : [{report['psnr_db']['min']:.2f}, {report['psnr_db']['max']:.2f}] dB")
    print()
    print(f"  Full report saved : {report_path}")
    print(f"{'='*65}\n")

    # ---- Quick quality interpretation --------------------------------
    if valid_ssim:
        m = report['ssim']['mean']
        if m >= 0.90:
            grade = "Excellent reconstruction"
        elif m >= 0.80:
            grade = "Good reconstruction"
        elif m >= 0.65:
            grade = "Acceptable — more epochs may help"
        else:
            grade = "Poor — check loss weights / training length"
        print(f"  Quality assessment: {grade}  (SSIM={m:.3f})\n")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Pix2Pix on Bangla handwriting test set')
    parser.add_argument('--config',     type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--output',     type=str, default='./eval_results',
                        help='Directory to save evaluation output')
    parser.add_argument('--num-comparisons', type=int, default=20,
                        help='Number of comparison images to save')
    parser.add_argument('--device',     type=str, default=None,
                        help='cuda / cpu (auto-detected if omitted)')
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_comparisons=args.num_comparisons,
        device_str=args.device,
    )


if __name__ == '__main__':
    main()
