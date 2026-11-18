"""
Regenerate fig_pivot.png
- No embedded title (LaTeX caption handles it)
- No bottom caption text (LaTeX caption handles it)
- Designed at exact print width so fonts render at correct pt size
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 0.95 * 150 mm text width = 5.61 inches → image saved at exactly this width
FIG_W = 5.61
FIG_H = 2.85
DPI   = 250

C_GREY   = '#757575'
C_BLUE   = '#1565C0'
C_ORANGE = '#E65100'
C_GREEN  = '#2E7D32'

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ── helpers ────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, color, fs=8.5, tc='white'):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.04",
                       facecolor=color, edgecolor='white',
                       linewidth=1.0, zorder=3)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', zorder=4,
            multialignment='center', linespacing=1.3)

def harrow(ax, x1, x2, y, color='#333333', ls='-'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                                linestyle=ls), zorder=2)

def varrow(ax, x, y_top, y_bot, label, color):
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2), zorder=2)
    ax.text(x + 0.07, (y_top + y_bot) / 2, label,
            ha='left', va='center', fontsize=8, color=color, style='italic')

# ── Layout (in inches) ────────────────────────────────────────────────────
BH  = 0.48    # box height
GAP = 0.05    # box-to-arrowhead gap

# 5 column x-starts (sum of widths + inter-column gaps fills ~5.5 in)
CX = [0.08, 1.24, 2.48, 3.72, 4.76]
CW = [0.88, 1.00, 1.00, 0.88, 0.78]

P2Y = 2.00    # Phase-2 row bottom y
P3Y = 0.85    # Phase-3 row bottom y

# ── Phase-2 row ────────────────────────────────────────────────────────────
ax.text(CX[0], P2Y + BH + 0.13, 'Phase-2 pipeline (built):',
        ha='left', va='bottom', fontsize=9, color=C_GREY, fontweight='bold')

p2_boxes = [
    ('Damaged\npage',       C_GREY),
    ('1. YOLO\ndetector',   C_BLUE),
    ('2. CNN-BiLSTM\n-CTC', C_BLUE),
    (None,                  None),      # spacer — no box
    ('Text\noutput',        C_GREEN),
]
for i, (lbl, col) in enumerate(p2_boxes):
    if col:
        box(ax, CX[i], P2Y, CW[i], BH, lbl, col)

harrow(ax, CX[0]+CW[0]+GAP, CX[1],  P2Y+BH/2)
harrow(ax, CX[1]+CW[1]+GAP, CX[2],  P2Y+BH/2)
harrow(ax, CX[2]+CW[2]+GAP, CX[4],  P2Y+BH/2, ls='dashed')

# ── Phase-3 row ────────────────────────────────────────────────────────────
ax.text(CX[0], P3Y + BH + 0.13, 'Phase-3 pipeline (this work):',
        ha='left', va='bottom', fontsize=9, color=C_ORANGE, fontweight='bold')

p3_boxes = [
    ('Damaged\npage',            C_GREY),
    ('1. YOLO\ndetector',        C_BLUE),
    ('2. TrOCR\nrecogniser',     C_ORANGE),
    ('3. Pix2Pix\nrestoration',  C_ORANGE),
    ('Clean image\n+ text',      C_GREEN),
]
for i, (lbl, col) in enumerate(p3_boxes):
    box(ax, CX[i], P3Y, CW[i], BH, lbl, col)
for i in range(len(p3_boxes) - 1):
    harrow(ax, CX[i]+CW[i]+GAP, CX[i+1], P3Y+BH/2)

# ── Vertical transition arrows ─────────────────────────────────────────────
varrow(ax, CX[0]+CW[0]/2, P2Y,       P3Y+BH, 'kept', C_GREY)
varrow(ax, CX[1]+CW[1]/2, P2Y,       P3Y+BH, 'kept', C_GREY)
varrow(ax, CX[2]+CW[2]/2, P2Y,       P3Y+BH, 'swap', C_ORANGE)

# Pix2Pix "new" dotted arrow
ax.annotate('', xy=(CX[3]+CW[3]/2, P3Y+BH),
            xytext=(CX[3]+CW[3]/2, P2Y+BH/2),
            arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.1,
                            linestyle='dotted'), zorder=2)
ax.text(CX[3]+CW[3]/2+0.07, (P2Y+BH/2+P3Y+BH)/2,
        'new', ha='left', va='center', fontsize=8, color=C_ORANGE, style='italic')

# ── TrOCR → Pix2Pix reference annotation ──────────────────────────────────
ax.annotate('',
    xy    =(CX[3]+0.04,       P3Y+0.13),
    xytext=(CX[2]+CW[2]-0.04, P3Y+0.09),
    arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.1,
                    connectionstyle='arc3,rad=-0.35',
                    linestyle='dashed'), zorder=5)
ax.text((CX[2]+CX[3])/2 + 0.10, P3Y - 0.10,
        'TrOCR output used as printed reference in Pix2Pix',
        ha='center', va='top', fontsize=7.5, color='#6A1B9A', style='italic')

out = r'D:\collects\final year thesis template\images\fig_pivot.png'
plt.savefig(out, dpi=DPI, facecolor='white')
print('Saved:', out)

from PIL import Image
im = Image.open(out)
print(f'Dimensions: {im.size[0]}x{im.size[1]} px  |  '
      f'Print size: {im.size[0]/DPI:.2f}" x {im.size[1]/DPI:.2f}"')
