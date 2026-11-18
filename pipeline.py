"""
pipeline.py  --  Three-Stage End-to-End Pipeline
=================================================
Implements the full pipeline described in Chapter 1 of the thesis:

  Stage 1  Detection    OpenCV morphological word segmentation.
                        (Production: swap in the YOLO detector weights from
                        Phase 2 by replacing segment_words() with YOLO inference.)

  Stage 2  Recognition  microsoft/trocr-base-handwritten (locally cached).
                        NOTE: this is an English-trained base model; Bangla
                        transcription accuracy is limited until the model is
                        fine-tuned on Bangla data.  The pipeline API is designed
                        so that Stage 2 can be swapped for any recogniser that
                        returns a string for a 256x256 RGB word crop.

  Stage 3  Restoration  Pix2Pix conditional GAN.
                        Checkpoint: checkpoints/best_model.pth (epoch 57).
                        Test-set performance: PSNR 26.62 dB, SSIM 0.9016.

Usage
-----
# Process a single word image:
    python pipeline.py --word path/to/damaged_word.png --output results/

# Process a full handwritten page (detection -> recognition -> restoration):
    python pipeline.py --page path/to/page.jpg --output results/

# Built-in demo using a BanglaWriting sample page:
    python pipeline.py --demo --output results/demo/
"""

import os
import sys
import io
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ── UTF-8 console (Windows cp1252 fix) ────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR       = Path(__file__).parent
CHECKPOINT     = REPO_DIR / 'checkpoints' / 'best_model.pth'
CONFIG_FILE    = REPO_DIR / 'config.yaml'
BANGLA_IMAGES  = Path('D:/bangla dataset/images')
BANGLA_ANN     = Path('D:/bangla dataset/annotations_json')


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordResult:
    """Result for one detected word."""
    box: Tuple[int, int, int, int]          # (x, y, w, h) in original page coords
    crop_damaged: Image.Image               # input crop fed to pipeline
    crop_restored: Image.Image              # Pix2Pix output
    transcription: str                      # TrOCR output string
    psnr: Optional[float] = None            # filled if clean GT available
    ssim: Optional[float] = None            # filled if clean GT available


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Word Detection
# ─────────────────────────────────────────────────────────────────────────────

class WordDetector:
    """
    OpenCV morphological segmenter.

    Replicate the YOLO-based detector from Phase 2 by swapping this class for
    a YOLOv8 inference wrapper that returns the same (crop_bgr, box) pairs.
    """

    DPI_PRESETS = {
        72:  dict(h_merge=8,  v_merge=4,  min_w=15,  min_h=8,   max_w=400,  max_h=150),
        150: dict(h_merge=15, v_merge=7,  min_w=25,  min_h=15,  max_w=700,  max_h=280),
        300: dict(h_merge=28, v_merge=12, min_w=45,  min_h=25,  max_w=1400, max_h=500),
    }

    def __init__(self, dpi: int = 150, pad: int = 6):
        nearest = min(self.DPI_PRESETS, key=lambda k: abs(k - dpi))
        self.p = {**self.DPI_PRESETS[nearest], 'pad': pad}

    def detect(self, img_bgr: np.ndarray) -> List[dict]:
        """
        Returns a list of {'crop': np.ndarray, 'box': (x, y, w, h)},
        sorted in approximate reading order.
        """
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bin_  = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        k_h    = cv2.getStructuringElement(cv2.MORPH_RECT, (self.p['h_merge'], 1))
        k_v    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.p['v_merge']))
        merged = cv2.dilate(cv2.dilate(bin_, k_h), k_v)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged)
        h_img, w_img = img_bgr.shape[:2]
        pad = self.p['pad']
        words = []

        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            if not (self.p['min_w'] <= w <= self.p['max_w'] and
                    self.p['min_h'] <= h <= self.p['max_h']):
                continue
            if w / max(h, 1) > 20:
                continue
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad);  y2 = min(h_img, y + h + pad)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            words.append({'crop': crop, 'box': (x1, y1, x2 - x1, y2 - y1)})

        words.sort(key=lambda d: (d['box'][1] // 30, d['box'][0]))
        return words


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Recognition
# ─────────────────────────────────────────────────────────────────────────────

class TrOCRRecogniser:
    """
    Wrapper around microsoft/trocr-base-handwritten.

    The model is loaded once and reused for all subsequent calls.
    The HuggingFace cache at
      C:/Users/nassa/.cache/huggingface/hub/models--microsoft--trocr-base-handwritten
    is used automatically (no re-download).

    NOTE: This is an English-language model.  Output on Bangla crops will be a
    best-effort ASCII transliteration rather than correct Bangla Unicode.
    Replacing it with a Bangla-fine-tuned TrOCR variant is identified as the
    immediate next step for end-to-end recognition evaluation.
    """

    def __init__(self, device=None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[Stage 2] Loading TrOCR on {self.device} ...')
        t0 = time.time()
        self.processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-handwritten').to(self.device)
        self.model.eval()
        print(f'[Stage 2] TrOCR loaded in {time.time()-t0:.1f}s')

    def recognise(self, pil_rgb: Image.Image) -> str:
        """Run recognition on a PIL RGB image; return decoded string."""
        import torch
        pixel_values = self.processor(
            images=pil_rgb, return_tensors='pt').pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Restoration
# ─────────────────────────────────────────────────────────────────────────────

class Pix2PixRestorer:
    """
    Pix2Pix U-Net generator (epoch 57, PSNR 26.62 dB, SSIM 0.9016).
    Loaded from CHECKPOINT = checkpoints/best_model.pth.
    """

    def __init__(self, config_path=CONFIG_FILE, ckpt_path=CHECKPOINT, device=None):
        import torch
        import yaml
        import torchvision.transforms as T

        sys.path.insert(0, str(REPO_DIR))
        from models import Pix2PixModel

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[Stage 3] Loading Pix2Pix restorer on {self.device} ...')
        t0 = time.time()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.net = Pix2PixModel(config).to(self.device)
        ckpt = torch.load(str(ckpt_path), map_location=self.device,
                          weights_only=False)
        self.net.generator.load_state_dict(ckpt['generator_state_dict'])
        self.net.generator.eval()

        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        print(f'[Stage 3] Pix2Pix loaded in {time.time()-t0:.1f}s')

    def restore(self, pil_rgb: Image.Image) -> Image.Image:
        """Restore a 256x256 RGB damaged word crop; return restored PIL image."""
        import torch
        import torchvision.transforms as T

        inp = pil_rgb.resize((256, 256), Image.BICUBIC)
        t   = self.tf(inp).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net.generator(t).squeeze(0).cpu()
        out = (out * 0.5 + 0.5).clamp(0, 1)
        return T.ToPILImage()(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class BanglaRestorationPipeline:
    """
    Three-stage pipeline that jointly produces a transcription (Stage 2) and
    a clean visual reproduction (Stage 3) for each handwritten word in the input.

    Parameters
    ----------
    load_recogniser : bool
        Set False to skip Stage 2 (TrOCR) and save load time.
    device : str | None
        'cuda' / 'cpu' / None (auto-detect).
    """

    def __init__(self, load_recogniser: bool = True, device: str = None):
        self.detector   = WordDetector()
        self.restorer   = Pix2PixRestorer(device=device)
        self.recogniser = TrOCRRecogniser(device=device) if load_recogniser else None

    # ── public API ─────────────────────────────────────────────────────────

    def process_word(self, pil_rgb: Image.Image) -> WordResult:
        """
        Run Stages 2-3 on a pre-cropped, possibly-damaged word image.

        Parameters
        ----------
        pil_rgb : PIL.Image  (any size; resized to 256x256 internally)

        Returns
        -------
        WordResult with .transcription and .crop_restored filled.
        """
        crop_resized = pil_rgb.resize((256, 256), Image.BICUBIC)
        transcription = (self.recogniser.recognise(crop_resized)
                         if self.recogniser else '')
        restored = self.restorer.restore(crop_resized)
        return WordResult(
            box=(0, 0, 256, 256),
            crop_damaged=crop_resized,
            crop_restored=restored,
            transcription=transcription,
        )

    def process_page(self, img_path: str,
                     output_dir: str,
                     save_annotated: bool = True) -> List[WordResult]:
        """
        Run the full three-stage pipeline on a handwritten page image.

        Stage 1  Detect word regions (OpenCV segmenter).
        Stage 2  Recognise each crop with TrOCR.
        Stage 3  Restore each crop with Pix2Pix.

        Saves individual restored crops + a grid summary to output_dir.
        Returns list of WordResult (one per detected word).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')

        print(f'\n[Stage 1] Detecting words in {Path(img_path).name} ...')
        words = self.detector.detect(img_bgr)
        print(f'[Stage 1] {len(words)} word regions detected')

        results = []
        for idx, w in enumerate(words):
            crop_bgr = w['crop']
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_in   = Image.fromarray(crop_rgb).resize((256, 256), Image.BICUBIC)

            # Stage 2
            text = self.recogniser.recognise(pil_in) if self.recogniser else ''

            # Stage 3
            restored = self.restorer.restore(pil_in)

            # Save
            stem = Path(img_path).stem
            pil_in.save(str(out / f'{stem}_w{idx:04d}_damaged.png'))
            restored.save(str(out / f'{stem}_w{idx:04d}_restored.png'))

            results.append(WordResult(
                box=w['box'],
                crop_damaged=pil_in,
                crop_restored=restored,
                transcription=text,
            ))
            if (idx + 1) % 10 == 0:
                print(f'  processed {idx+1}/{len(words)} words ...')

        # Annotated page
        if save_annotated:
            ann_img = img_bgr.copy()
            for r in results:
                x, y, ww, hh = r.box
                cv2.rectangle(ann_img, (x, y), (x+ww, y+hh), (0, 180, 50), 2)
            cv2.imwrite(str(out / f'{Path(img_path).stem}_detected.png'), ann_img)

        # Grid summary (first 12 words)
        self._save_grid(results[:12], str(out / f'{Path(img_path).stem}_grid.png'))

        print(f'\n[Pipeline] Done. Results saved to {out}')
        print(f'  Words processed : {len(results)}')
        return results

    # ── helpers ────────────────────────────────────────────────────────────

    def _save_grid(self, results: List[WordResult], out_path: str,
                   cols: int = 4) -> None:
        """Save a side-by-side damaged / restored grid for the first N results."""
        if not results:
            return
        pw = 256;  ph = 300  # panel width / height (extra 44px for text strip)
        rows = (len(results) + cols - 1) // cols
        W = cols * pw * 2    # 2 panels per result (damaged | restored)
        H = rows * ph
        canvas = Image.new('RGB', (W, H), (245, 245, 245))

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('C:/Windows/Fonts/calibri.ttf', 11)
        except Exception:
            font = ImageFont.load_default()

        for i, r in enumerate(results):
            row = i // cols;  col = i % cols
            x0 = col * pw * 2;  y0 = row * ph
            canvas.paste(r.crop_damaged,  (x0,      y0))
            canvas.paste(r.crop_restored, (x0 + pw, y0))
            draw.text((x0,      y0 + 258), 'Damaged',  font=font, fill='#C62828')
            draw.text((x0 + pw, y0 + 258), 'Restored', font=font, fill='#2E7D32')
            # TrOCR text (truncated to fit)
            t = r.transcription[:24] if r.transcription else '—'
            draw.text((x0, y0 + 272), f'TrOCR: {t}', font=font, fill='#555555')

        canvas.save(out_path)
        print(f'[Pipeline] Grid saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(output_dir: str) -> None:
    """
    Demo: pick 6 word crops from BanglaWriting annotations, apply 6-type
    online damage, then run Stage 2 + Stage 3 and save results.
    """
    import random
    sys.path.insert(0, str(REPO_DIR))
    from damage_utils import PRESETS, make_damaged

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── load pipeline (no detection needed for pre-annotated words) ─────────
    pipe = BanglaRestorationPipeline(load_recogniser=True)

    # ── pick sample words from BanglaWriting ─────────────────────────────────
    ann_files = sorted(BANGLA_ANN.glob('*.json'))
    if not ann_files:
        print('[Demo] No annotation files found in', BANGLA_ANN)
        return

    random.seed(2025); np.random.seed(2025)
    samples = []
    for ann_f in ann_files:
        with open(ann_f, encoding='utf-8') as f:
            ann = json.load(f)
        img_path = BANGLA_IMAGES / ann['imagePath']
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        shapes  = ann.get('shapes', [])
        random.shuffle(shapes)
        for sh in shapes[:2]:
            pts = sh['points']
            xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x1 = max(0, int(min(xs)) - 6);  y1 = max(0, int(min(ys)) - 6)
            x2 = min(img_bgr.shape[1], int(max(xs)) + 6)
            y2 = min(img_bgr.shape[0], int(max(ys)) + 6)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size < 32 * 32 * 3:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            clean    = Image.fromarray(crop_rgb).resize((256, 256), Image.BICUBIC)
            samples.append((sh['label'], clean))
            if len(samples) >= 6:
                break
        if len(samples) >= 6:
            break

    print(f'\n[Demo] Running pipeline on {len(samples)} sample words ...')
    demo_results = []
    for label, clean_pil in samples:
        # apply 6-type online damage
        damaged_np  = make_damaged(np.array(clean_pil), PRESETS['distorted'])
        damaged_pil = Image.fromarray(damaged_np)

        result = pipe.process_word(damaged_pil)
        result.psnr = _psnr(np.array(result.crop_restored), np.array(clean_pil))
        result.ssim = _ssim(np.array(result.crop_restored), np.array(clean_pil))
        demo_results.append((label, clean_pil, result))
        print(f'  {label[:12]:<14}  TrOCR="{result.transcription[:20]}"'
              f'  PSNR={result.psnr:.2f}dB  SSIM={result.ssim:.4f}')

    # ── save individual outputs ───────────────────────────────────────────────
    for i, (lbl, clean, r) in enumerate(demo_results):
        r.crop_damaged.save(str(out / f'demo_w{i:02d}_damaged.png'))
        r.crop_restored.save(str(out / f'demo_w{i:02d}_restored.png'))
        clean.save(str(out / f'demo_w{i:02d}_clean_gt.png'))

    # ── save compact figure ───────────────────────────────────────────────────
    _save_demo_figure(demo_results, str(out / 'pipeline_demo.png'))
    print(f'\n[Demo] All outputs saved to {out}')


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255 / np.sqrt(mse))


def _ssim(a: np.ndarray, b: np.ndarray, win: int = 11) -> float:
    """Single-channel mean SSIM approximation (no external dependency)."""
    from scipy.ndimage import uniform_filter
    a = a.mean(axis=2).astype(float)
    b = b.mean(axis=2).astype(float)
    c1, c2 = 6.5025, 58.5225
    mu1  = uniform_filter(a, win);   mu2  = uniform_filter(b, win)
    s1   = uniform_filter(a*a, win) - mu1**2
    s2   = uniform_filter(b*b, win) - mu2**2
    s12  = uniform_filter(a*b, win) - mu1*mu2
    num  = (2*mu1*mu2 + c1) * (2*s12 + c2)
    den  = (mu1**2 + mu2**2 + c1) * (s1 + s2 + c2)
    return float(np.mean(num / den))


def _save_demo_figure(demo_results, out_path: str) -> None:
    """Create compact thesis-style demo figure: damaged | restored | GT (per row)."""
    from PIL import ImageDraw, ImageFont
    N = len(demo_results)
    PW = 256; GAP = 6; LW = 130; HDRH = 32; BORDER = 10
    total_w = BORDER*2 + LW + 3*PW + 2*GAP
    total_h = BORDER*2 + HDRH + N*(PW + GAP) + GAP

    canvas = Image.new('RGB', (total_w, total_h), (250, 250, 250))
    draw   = ImageDraw.Draw(canvas)

    try:
        fhdr = ImageFont.truetype('C:/Windows/Fonts/calibrib.ttf', 13)
        flbl = ImageFont.truetype('C:/Windows/Fonts/calibri.ttf',  12)
        fsml = ImageFont.truetype('C:/Windows/Fonts/calibri.ttf',  10)
    except Exception:
        fhdr = flbl = fsml = ImageFont.load_default()

    # column headers
    hdrs = ['Damaged input', 'Pix2Pix restored', 'GT (clean)']
    col_xs = [BORDER + LW, BORDER + LW + PW + GAP, BORDER + LW + 2*(PW + GAP)]
    for cx, h in zip(col_xs, hdrs):
        draw.text((cx + PW//2, BORDER + HDRH//2), h,
                  font=fhdr, fill='#222222', anchor='mm')
    draw.line([(BORDER, BORDER+HDRH), (total_w-BORDER, BORDER+HDRH)],
              fill='#AAAAAA', width=1)

    row_colors = ['#78909C','#1565C0','#BF360C','#2E7D32','#6A1B9A','#E65100']
    y = BORDER + HDRH + GAP

    for i, (lbl, clean, r) in enumerate(demo_results):
        # label strip
        draw.rectangle([BORDER, y, BORDER+LW-4, y+PW],
                       fill=row_colors[i % len(row_colors)])
        draw.multiline_text((BORDER + LW//2 - 2, y + PW//2),
                            f'Word {i+1}',
                            font=flbl, fill='white', anchor='mm', align='center')

        # images
        canvas.paste(r.crop_damaged,  (BORDER + LW,            y))
        canvas.paste(r.crop_restored, (BORDER + LW + PW + GAP, y))
        canvas.paste(clean,           (BORDER + LW + 2*(PW+GAP), y))

        # dividers
        for col_x in col_xs[1:]:
            draw.line([(col_x, y), (col_x, y+PW)], fill='#DDDDDD', width=1)

        # metrics
        metrics = (f'PSNR {r.psnr:.1f}dB  SSIM {r.ssim:.3f}'
                   if r.psnr is not None else '')
        trocr   = f'TrOCR: "{r.transcription[:18]}"' if r.transcription else ''
        draw.text((BORDER + LW, y + PW + 3), trocr,   font=fsml, fill='#555555')
        draw.text((BORDER + LW + PW + GAP, y + PW + 3), metrics,
                  font=fsml, fill='#1A6600')

        y += PW + GAP + 18

    draw.rectangle([0, 0, total_w-1, total_h-1], outline='#CCCCCC', width=1)
    canvas.save(out_path)
    print(f'[Demo] Figure saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Three-stage Bangla word restoration pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--word',  metavar='IMG',
                       help='Path to a single (possibly damaged) word image')
    group.add_argument('--page',  metavar='IMG',
                       help='Path to a full handwritten page image')
    group.add_argument('--demo',  action='store_true',
                       help='Run built-in demo using BanglaWriting samples')
    parser.add_argument('--output', default='pipeline_results/',
                        help='Output directory (default: pipeline_results/)')
    parser.add_argument('--no-recogniser', action='store_true',
                        help='Skip TrOCR (Stage 2) to save load time')
    parser.add_argument('--device', default=None,
                        help="'cuda' / 'cpu' / None for auto-detect")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.output)

    elif args.word:
        pipe = BanglaRestorationPipeline(
            load_recogniser=not args.no_recogniser,
            device=args.device)
        pil = Image.open(args.word).convert('RGB')
        r   = pipe.process_word(pil)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        r.crop_restored.save(str(out / 'restored.png'))
        print(f'\n  TrOCR: "{r.transcription}"')
        print(f'  Restored saved: {out}/restored.png')

    elif args.page:
        pipe = BanglaRestorationPipeline(
            load_recogniser=not args.no_recogniser,
            device=args.device)
        pipe.process_page(args.page, args.output)


if __name__ == '__main__':
    main()
