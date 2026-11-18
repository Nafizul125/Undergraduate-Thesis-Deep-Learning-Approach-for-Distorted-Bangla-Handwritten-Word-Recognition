"""
crop_words.py  —  Word Segmentation for Bangla Handwritten Page Images
=======================================================================
Takes full-page handwritten images and crops individual word bounding boxes.

How it works (OpenCV-based, no ML required):
  1. Convert to grayscale + Otsu binarization (text → white, bg → black)
  2. Dilate horizontally to merge characters within a word
     (also dilates slightly vertically to merge mātrā headline with body)
  3. Find connected components in the dilated mask
  4. Map each component back to the original image and crop it
  5. Filter by size (remove punctuation marks, noise blobs, full-line blobs)

Usage:
    python crop_words.py --input data_pages/ --output data_raw/word_crops/
    python crop_words.py --input data_pages/ --output data_raw/word_crops/ --dpi 300 --debug

Arguments:
    --input   : directory containing full-page handwritten PNG/JPG images
    --output  : directory to save individual word crop PNG files
    --dpi     : estimated scan resolution (72/150/300) — controls merge distances
    --min-w   : minimum word crop width  in pixels (default: auto from dpi)
    --min-h   : minimum word crop height in pixels (default: auto from dpi)
    --debug   : save annotated page images showing detected word boxes
"""

import os
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DPI-aware parameter presets
# ---------------------------------------------------------------------------
DPI_PRESETS = {
    72:  dict(h_merge=8,  v_merge=4,  min_w=15, min_h=8,  max_w=400, max_h=150, pad=4),
    150: dict(h_merge=15, v_merge=7,  min_w=25, min_h=15, max_w=700, max_h=280, pad=6),
    300: dict(h_merge=28, v_merge=12, min_w=45, min_h=25, max_w=1400, max_h=500, pad=10),
}


def get_params(dpi: int, overrides: dict) -> dict:
    """Return DPI-tuned parameters, with any manual overrides applied."""
    # Find the nearest DPI preset
    nearest = min(DPI_PRESETS.keys(), key=lambda k: abs(k - dpi))
    params = dict(DPI_PRESETS[nearest])
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Robust binarization:
    - Try Otsu's on the full image
    - If the image is very light or very dark, apply CLAHE first
    Returns a binary image where text pixels are WHITE (255).
    """
    mean_val = gray.mean()

    if mean_val < 30 or mean_val > 240:
        # Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def segment_words(img_bgr: np.ndarray, params: dict) -> list:
    """
    Segment word bounding boxes from a full-page image.

    Returns:
        List of dicts: {'crop': np.ndarray, 'box': (x, y, w, h)}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)

    # --- Dilation to merge strokes into word blobs ---
    # Horizontal: merges characters within a word
    k_h = cv2.getStructuringElement(
        cv2.MORPH_RECT, (params['h_merge'], 1))
    merged = cv2.dilate(binary, k_h, iterations=1)

    # Vertical: merges mātrā headline with sub-character bodies
    k_v = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, params['v_merge']))
    merged = cv2.dilate(merged, k_v, iterations=1)

    # --- Connected components ---
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged)

    words = []
    h_img, w_img = img_bgr.shape[:2]
    pad = params['pad']

    for i in range(1, num_labels):           # skip background label 0
        x, y, w, h, area = stats[i]

        # Size filter
        if not (params['min_w'] <= w <= params['max_w'] and
                params['min_h'] <= h <= params['max_h']):
            continue

        # Aspect ratio: reject very thin horizontal lines (could be ruled lines)
        if h < params['min_h'] or w / max(h, 1) > 20:
            continue

        # Padded crop (clipped to image bounds)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        words.append({'crop': crop, 'box': (x1, y1, x2 - x1, y2 - y1)})

    # Sort left-to-right, top-to-bottom (reading order)
    words.sort(key=lambda d: (d['box'][1] // 30, d['box'][0]))

    return words


def draw_debug(img_bgr: np.ndarray, words: list) -> np.ndarray:
    """Draw detected word boxes on a copy of the image."""
    debug = img_bgr.copy()
    for i, w in enumerate(words):
        x, y, ww, hh = w['box']
        cv2.rectangle(debug, (x, y), (x + ww, y + hh), (0, 200, 50), 2)
        cv2.putText(debug, str(i), (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    return debug


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_page(img_path: Path,
                 output_dir: Path,
                 params: dict,
                 debug_dir: Path = None) -> int:
    """
    Process one page image → save individual word crops.
    Returns number of words saved.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read: {img_path.name}")
        return 0

    words = segment_words(img, params)

    stem = img_path.stem
    saved = 0
    for idx, w in enumerate(words):
        out_path = output_dir / f"{stem}_word{idx:04d}.png"
        # Save as clean white-background RGB PNG
        cv2.imwrite(str(out_path), w['crop'])
        saved += 1

    if debug_dir is not None and words:
        debug_img = draw_debug(img, words)
        cv2.imwrite(str(debug_dir / f"debug_{img_path.name}"), debug_img)

    return saved


def run(input_dir: str,
        output_dir: str,
        dpi: int = 150,
        debug: bool = False,
        min_w: int = None,
        min_h: int = None):

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug_path = None
    if debug:
        debug_path = Path(output_dir).parent / 'debug_crops'
        debug_path.mkdir(parents=True, exist_ok=True)

    # Resolve parameters
    params = get_params(dpi, {'min_w': min_w, 'min_h': min_h})

    # Collect page images
    exts = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    pages = sorted([p for p in input_path.iterdir()
                    if p.suffix.lower() in exts])

    if not pages:
        print(f"[ERROR] No image files found in {input_dir}")
        return

    print(f"\n{'='*65}")
    print(f"  Word Segmentation — Bangla Handwriting")
    print(f"{'='*65}")
    print(f"  Input pages   : {len(pages)}")
    print(f"  Output dir    : {output_path}")
    print(f"  DPI preset    : {dpi}")
    print(f"  H-merge kernel: {params['h_merge']} px")
    print(f"  V-merge kernel: {params['v_merge']} px")
    print(f"  Word size     : w=[{params['min_w']},{params['max_w']}]  "
          f"h=[{params['min_h']},{params['max_h']}]")
    print(f"{'='*65}\n")

    total_words = 0
    page_summary = []

    for page_path in tqdm(pages, desc="Segmenting pages"):
        count = process_page(page_path, output_path, params, debug_path)
        total_words += count
        page_summary.append({'page': page_path.name, 'words': count})

    # Save manifest
    manifest = {
        'input_dir': str(input_path),
        'output_dir': str(output_path),
        'dpi': dpi,
        'params': params,
        'pages_processed': len(pages),
        'total_words_cropped': total_words,
        'per_page': page_summary,
    }
    manifest_path = output_path.parent / 'crop_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Pages processed   : {len(pages)}")
    print(f"  Total word crops  : {total_words}")
    print(f"  Avg per page      : {total_words / max(len(pages), 1):.1f}")
    print(f"  Crops saved to    : {output_path}")
    print(f"  Manifest saved to : {manifest_path}")
    if debug_path:
        print(f"  Debug images      : {debug_path}")
    print(f"{'='*65}")
    print(f"\n  Next step:")
    print(f"  python generate_dataset.py --input {output_path} "
          f"--output data_minimal/\n")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Crop individual Bangla handwritten words from page images')
    parser.add_argument('--input',  required=True,
                        help='Directory of full-page handwritten images')
    parser.add_argument('--output', required=True,
                        help='Directory to save word crop PNGs')
    parser.add_argument('--dpi',    type=int, default=150,
                        help='Estimated scan DPI: 72 / 150 / 300  (default: 150)')
    parser.add_argument('--min-w',  type=int, default=None,
                        help='Override minimum word width in pixels')
    parser.add_argument('--min-h',  type=int, default=None,
                        help='Override minimum word height in pixels')
    parser.add_argument('--debug',  action='store_true',
                        help='Save annotated debug images showing detected boxes')
    args = parser.parse_args()

    run(args.input, args.output,
        dpi=args.dpi,
        debug=args.debug,
        min_w=args.min_w,
        min_h=args.min_h)


if __name__ == '__main__':
    main()
