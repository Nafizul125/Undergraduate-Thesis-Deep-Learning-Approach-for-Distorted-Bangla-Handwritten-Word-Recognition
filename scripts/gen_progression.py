"""
gen_progression.py
==================
Generates fig_progression.png: a 5-row comparison figure showing the
development journey from early/failed configurations to the final model.

Rows:
  1. Baseline / early training    (sample_abpair.png)
  2. Mode collapse                 (sample_collapse.png)
  3. OCR-semantic divergence       (sample_ocr.png)
  4. Ablation best (λ=150, static) (sample_best.png)
  5. Enhanced final model          (first strip from fig_showcase_top6.png)

Each existing strip is: [Damaged input | Target | Reconstruction]
Left row label added for context.
"""

from PIL import Image, ImageDraw, ImageFont
import os

IMG = r'D:\collects\final year thesis template\images'

def try_font(size):
    for p in ["C:/Windows/Fonts/calibrib.ttf",
              "C:/Windows/Fonts/calibri.ttf",
              "C:/Windows/Fonts/arial.ttf",
              "C:/Windows/Fonts/segoeui.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

# ── load source strips ────────────────────────────────────────────────────
def load(name):
    return Image.open(os.path.join(IMG, name)).convert('RGB')

strips_raw = [
    ('Baseline\n(early run)',         load('sample_abpair.png')),
    ('Mode collapse\n(VGG+Edge+SSIM)',load('sample_collapse.png')),
    ('Diverged\n(OCR semantic loss)', load('sample_ocr.png')),
    ('Ablation best\n(λ₁=150, 20.8 dB)', load('sample_best.png')),
]

# For enhanced: crop the top strip from showcase_top6
top6 = load('fig_showcase_top6.png')
strip_h = top6.size[1] // 6
enhanced_strip = top6.crop((0, 0, top6.size[0], strip_h))
strips_raw.append(('Enhanced model\n(26.62 dB, SSIM 0.90)', enhanced_strip))

# ── layout ────────────────────────────────────────────────────────────────
LABEL_W  = 170    # width of left label column
STRIP_W  = 780    # target width for each strip
GAP      = 8      # vertical gap between rows
HDRH     = 36     # column header height
BORDER   = 12

font_label  = try_font(15)
font_hdr    = try_font(14)
font_small  = try_font(12)

# Resize strips to uniform width
def resize_strip(im, target_w):
    w, h = im.size
    new_h = int(h * target_w / w)
    return im.resize((target_w, new_h), Image.LANCZOS)

strips = [(lbl, resize_strip(im, STRIP_W)) for lbl, im in strips_raw]
strip_h_vals = [im.size[1] for _, im in strips]

total_h = (BORDER*2 + HDRH + sum(strip_h_vals)
           + GAP * (len(strips) - 1) + GAP)   # extra gap under header
total_w = BORDER*2 + LABEL_W + STRIP_W

canvas = Image.new('RGB', (total_w, total_h), (250, 250, 250))
draw   = ImageDraw.Draw(canvas)

# ── column headers ─────────────────────────────────────────────────────────
col_labels = ['Damaged input', 'Ground truth', 'Model output']
col_w = STRIP_W // 3

y_hdr = BORDER
for i, lbl in enumerate(col_labels):
    cx = BORDER + LABEL_W + i * col_w + col_w // 2
    draw.text((cx, y_hdr + HDRH//2), lbl,
              font=font_hdr, fill='#333333', anchor='mm')

# separator under header
draw.line([(BORDER, y_hdr + HDRH),
           (total_w - BORDER, y_hdr + HDRH)],
          fill='#CCCCCC', width=1)

# ── colour scheme for row labels ──────────────────────────────────────────
row_colors = [
    '#78909C',   # baseline — grey-blue
    '#C62828',   # collapse  — red
    '#BF360C',   # diverged  — deep orange
    '#1565C0',   # ablation best — blue
    '#2E7D32',   # enhanced  — green
]

# ── draw rows ─────────────────────────────────────────────────────────────
y = y_hdr + HDRH + GAP
for ri, (lbl, strip) in enumerate(strips):
    sh = strip.size[1]

    # label box
    lbl_color = row_colors[ri]
    draw.rectangle([BORDER, y, BORDER + LABEL_W - 4, y + sh],
                   fill=lbl_color)
    draw.multiline_text(
        (BORDER + LABEL_W//2 - 2, y + sh//2),
        lbl, font=font_label, fill='white',
        anchor='mm', align='center', spacing=4)

    # paste strip
    canvas.paste(strip, (BORDER + LABEL_W, y))

    # thin border around strip
    draw.rectangle(
        [BORDER + LABEL_W, y,
         BORDER + LABEL_W + STRIP_W - 1, y + sh - 1],
        outline='#BBBBBB', width=1)

    # thin vertical dividers between panels
    for di in [1, 2]:
        x_div = BORDER + LABEL_W + di * col_w
        draw.line([(x_div, y), (x_div, y + sh)], fill='#DDDDDD', width=1)

    y += sh + GAP

# outer border
draw.rectangle([0, 0, total_w-1, total_h-1], outline='#CCCCCC', width=1)

out = os.path.join(IMG, 'fig_progression.png')
canvas.save(out)
print(f'Saved: {out}')
im = Image.open(out)
print(f'Size: {im.size[0]}x{im.size[1]} px')
