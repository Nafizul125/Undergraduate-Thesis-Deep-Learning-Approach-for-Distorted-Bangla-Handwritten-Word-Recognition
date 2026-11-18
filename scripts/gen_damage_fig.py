"""
gen_damage_fig.py
=================
Generates fig_damage_types.png at high resolution (1800+ px wide)
showing the 6 individual damage types + combined preset.

Output: D:\collects\final year thesis template\images\fig_damage_types.png
"""

import sys, os, random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r'D:\pix2pix')
import damage_utils as du

random.seed(7)
np.random.seed(7)

# ── load 3 clean word crops (show 3 rows for variety) ─────────────────────
CROP_DIR = r'D:\pix2pix\data_raw\word_crops'
files = sorted([f for f in os.listdir(CROP_DIR)
                if f.endswith('.png') or f.endswith('.jpg')])

# pick three different words
sample_paths = [files[4], files[9], files[15]]

def load_clean(fname):
    img_bgr = cv2.imread(os.path.join(CROP_DIR, fname))
    return du.prepare_crop(img_bgr)   # 256x256 RGB

# ── damage definitions ────────────────────────────────────────────────────
def dmg_noise(img):    return du.apply_gaussian_noise(img, sigma=20.0)
def dmg_blur(img):     return du.apply_heavy_blur(img, kernel_range=(7, 9))
def dmg_fade(img):     return du.apply_ink_fading(img, factor_range=(0.52, 0.68))
def dmg_erosion(img):  return du.apply_erosion(img, kernel_range=(2, 2))
def dmg_stains(img):   return du.apply_water_stains(img, count_range=(2, 3), radius_range=(8, 16))
def dmg_yellow(img):   return du.apply_yellowing(img, strength_range=(0.15, 0.25))
def dmg_combined(img): return du.make_damaged(img, du.PRESETS['distorted'])

COLS = [
    ('Clean\n(original)',              lambda x: x),
    ('Gaussian noise\n(σ=20)',    dmg_noise),
    ('Heavy blur\n(7–9 px)',      dmg_blur),
    ('Ink fading\n(factor 0.52–0.68)', dmg_fade),
    ('Morphological\nerosion',         dmg_erosion),
    ('Water\nstains',                  dmg_stains),
    ('Paper\nyellowing',               dmg_yellow),
    ('Combined\n(distorted preset)',   dmg_combined),
]

# ── layout ────────────────────────────────────────────────────────────────
THUMB    = 160     # thumbnail size per cell
GAP_H    = 10      # horizontal gap between cells
GAP_V    = 8       # vertical gap between rows
LABEL_H  = 44      # height of column header (above first row only)
SIDE_W   = 0       # no row labels needed
BORDER   = 18

N_COLS   = len(COLS)
N_ROWS   = len(sample_paths)

W = BORDER*2 + N_COLS * THUMB + (N_COLS-1) * GAP_H
H = BORDER*2 + LABEL_H + N_ROWS * THUMB + (N_ROWS-1) * GAP_V

canvas = Image.new('RGB', (W, H), (248, 248, 248))
draw   = ImageDraw.Draw(canvas)

def try_font(size):
    for p in ["C:/Windows/Fonts/calibri.ttf",
              "C:/Windows/Fonts/arial.ttf",
              "C:/Windows/Fonts/segoeui.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

font_hdr   = try_font(14)
font_small = try_font(12)

for ci, (col_label, fn) in enumerate(COLS):
    x_cell = BORDER + ci * (THUMB + GAP_H)

    # column header (only printed once above first row)
    draw.multiline_text(
        (x_cell + THUMB//2, BORDER + LABEL_H//2),
        col_label,
        font=font_hdr, fill='#222222',
        anchor='mm', align='center'
    )

    # thin separator line under header
    draw.line([(x_cell, BORDER + LABEL_H - 2),
               (x_cell + THUMB - 1, BORDER + LABEL_H - 2)],
              fill='#CCCCCC', width=1)

    for ri, sample_fname in enumerate(sample_paths):
        random.seed(7 + ri * 13 + ci * 7)
        np.random.seed(7 + ri * 13 + ci * 7)
        clean = load_clean(sample_fname)
        damaged = fn(clean.copy())

        y_cell = BORDER + LABEL_H + ri * (THUMB + GAP_V)

        thumb = Image.fromarray(damaged).resize((THUMB, THUMB), Image.LANCZOS)
        canvas.paste(thumb, (x_cell, y_cell))
        # thin border
        draw.rectangle([x_cell, y_cell, x_cell + THUMB - 1, y_cell + THUMB - 1],
                       outline='#BBBBBB', width=1)

# thin outer border
draw.rectangle([0, 0, W-1, H-1], outline='#BBBBBB', width=1)

out_path = r'D:\collects\final year thesis template\images\fig_damage_types.png'
canvas.save(out_path)
print(f'Saved: {out_path}')

im = Image.open(out_path)
print(f'Size: {im.size[0]}x{im.size[1]} px')
