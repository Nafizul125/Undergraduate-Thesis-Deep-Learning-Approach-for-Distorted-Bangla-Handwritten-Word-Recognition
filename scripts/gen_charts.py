"""
gen_charts.py
=============
Regenerate fig_psnr_bar.png and fig_loss_family.png including the enhanced model.
Also regenerate fig_dataset.png and fig_lambda.png to confirm data is correct.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = r'D:\collects\final year thesis template\images'
DPI = 180

# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 9,
})

C_BEST    = '#2E7D32'   # dark green  – best / stable
C_STABLE  = '#1565C0'   # blue        – stable but not best
C_COLLAPSE= '#C62828'   # red         – mode collapse
C_DIVERGE = '#BF360C'   # deep orange – diverged
C_ENHANCED= '#F57F17'   # amber       – enhanced model (special)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PSNR BAR CHART  (fig_psnr_bar.png)
# ─────────────────────────────────────────────────────────────────────────────
# (label, psnr, color, note)
runs = [
    ('Enhanced model\n(online aug + VGG)',   26.62, C_ENHANCED, 'test-set'),
    ('L1 (λ=150)',                           20.78, C_BEST,     'ablation best'),
    ('VGG + attention',                      18.68, C_STABLE,   ''),
    ('L1 (λ=70) wide gen',                   18.33, C_STABLE,   ''),
    ('Edge (L1=70)',                          18.31, C_STABLE,   ''),
    ('Feature matching',                     17.83, C_STABLE,   ''),
    ('Baseline (L1=100)',                    17.64, C_STABLE,   ''),
    ('Charbonnier',                          17.30, C_STABLE,   ''),
    ('L1 (λ=70) warmup',                     17.22, C_STABLE,   ''),
    ('L1 (λ=40)',                            17.19, C_STABLE,   ''),
    ('VGG + window-attn',                    13.05, C_COLLAPSE,  'collapse'),
    ('Edge + SSIM + VGG',                    13.02, C_COLLAPSE,  'collapse'),
    ('OCR semantic',                          7.74, C_DIVERGE,  'diverged'),
]

labels = [r[0] for r in runs]
psnrs  = [r[1] for r in runs]
colors = [r[2] for r in runs]
notes  = [r[3] for r in runs]

fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.barh(range(len(runs)), psnrs, color=colors, height=0.65,
               edgecolor='white', linewidth=0.8)

# value labels at end of each bar
for i, (bar, psnr, note) in enumerate(zip(bars, psnrs, notes)):
    txt = f'{psnr:.2f} dB'
    if note:
        txt += f'  [{note}]'
    ax.text(psnr + 0.2, i, txt, va='center', fontsize=8,
            color='#333333')

# separator line between enhanced model and ablation runs
ax.axhline(y=11.5, color='#999999', lw=0.8, linestyle='--')
ax.text(1.5, 11.65, 'Ablation study (validation split)', fontsize=7.5,
        color='#555555', style='italic')
ax.text(1.5, 12.30, 'Enhanced model (held-out test set)', fontsize=7.5,
        color=C_ENHANCED, style='italic', fontweight='bold')

ax.set_yticks(range(len(runs)))
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel('PSNR (dB)', fontsize=9)
ax.set_xlim(0, 33)
ax.set_title('Restoration Quality by Configuration', fontsize=10, fontweight='bold', pad=8)

# legend
legend_items = [
    mpatches.Patch(color=C_ENHANCED, label='Enhanced model (test-set PSNR)'),
    mpatches.Patch(color=C_BEST,     label='Best ablation (val PSNR)'),
    mpatches.Patch(color=C_STABLE,   label='Stable run (val PSNR)'),
    mpatches.Patch(color=C_COLLAPSE, label='Mode collapse'),
    mpatches.Patch(color=C_DIVERGE,  label='Diverged'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=7.5, framealpha=0.85)

plt.tight_layout(pad=0.8)
out1 = f'{OUT}/fig_psnr_bar.png'
plt.savefig(out1, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FAMILY BAR  (fig_loss_family.png)
# ─────────────────────────────────────────────────────────────────────────────
# group by loss objective family; show best PSNR per family
families = [
    ('Online Aug + L1 + VGG\n(enhanced model)',  26.62, C_ENHANCED, 'test-set PSNR'),
    ('Plain adversarial + L1\n(best: λ=150)',     20.78, C_BEST,     ''),
    ('L1 + VGG perceptual\n(static data)',        18.68, C_STABLE,   ''),
    ('L1 + Sobel edge',                           18.31, C_STABLE,   ''),
    ('L1 + feature matching',                     17.83, C_STABLE,   ''),
    ('Charbonnier',                               17.30, C_STABLE,   ''),
    ('L1 + VGG + edge + SSIM\n(combo)',           13.02, C_COLLAPSE,  'collapse'),
    ('OCR semantic loss',                          7.74, C_DIVERGE,  'diverged'),
]

fam_labels = [f[0] for f in families]
fam_psnrs  = [f[1] for f in families]
fam_colors = [f[2] for f in families]
fam_notes  = [f[3] for f in families]

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.barh(range(len(families)), fam_psnrs, color=fam_colors,
               height=0.60, edgecolor='white', linewidth=0.8)

for i, (bar, psnr, note) in enumerate(zip(bars, fam_psnrs, fam_notes)):
    txt = f'{psnr:.2f} dB'
    if note:
        txt += f'  [{note}]'
    ax.text(psnr + 0.2, i, txt, va='center', fontsize=8, color='#333333')

ax.axhline(y=6.5, color='#999999', lw=0.8, linestyle='--')
ax.set_yticks(range(len(families)))
ax.set_yticklabels(fam_labels, fontsize=8.5)
ax.set_xlabel('Best PSNR achieved (dB)', fontsize=9)
ax.set_xlim(0, 33)
ax.set_title('Best PSNR by Loss Objective Family', fontsize=10, fontweight='bold', pad=8)

legend_items = [
    mpatches.Patch(color=C_ENHANCED, label='Enhanced model (test-set PSNR)'),
    mpatches.Patch(color=C_BEST,     label='Best ablation (val PSNR)'),
    mpatches.Patch(color=C_STABLE,   label='Stable (val PSNR)'),
    mpatches.Patch(color=C_COLLAPSE, label='Unstable / collapse'),
    mpatches.Patch(color=C_DIVERGE,  label='Diverged'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=7.5, framealpha=0.85)

plt.tight_layout(pad=0.8)
out2 = f'{OUT}/fig_loss_family.png'
plt.savefig(out2, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out2)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LAMBDA SWEEP  (fig_lambda.png) — confirm data is correct
# ─────────────────────────────────────────────────────────────────────────────
# from Table tab:lambda: λ=40→17.19, λ=70→18.33, λ=100→17.64 (100ep), λ=150→20.78
lambdas = [40, 70, 100, 150]
psnr_l  = [17.19, 18.33, 17.64, 20.78]
styles  = ['o-', 'o-', 'o--', 'o-']   # dashed for 100ep (shorter run)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lambdas, psnr_l, marker='o', color=C_BEST, linewidth=1.8,
        markersize=7, markeredgecolor='white', markeredgewidth=1.2)
ax.scatter([100], [17.64], marker='o', color='#78909C', s=70, zorder=5)
ax.annotate('100 ep\n(shorter run)', xy=(100, 17.64),
            xytext=(105, 17.0), fontsize=7.5, color='#78909C',
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=1))
for lam, p in zip(lambdas, psnr_l):
    ax.text(lam, p + 0.15, f'{p:.2f}', ha='center', fontsize=8, color='#333333')
ax.set_xlabel('L1 weight (λ$_{L1}$)', fontsize=9)
ax.set_ylabel('Validation PSNR (dB)', fontsize=9)
ax.set_title('Effect of L1 Weight on Validation PSNR', fontsize=10, fontweight='bold', pad=8)
ax.set_xticks(lambdas)
ax.set_ylim(16, 22)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(pad=0.8)
out3 = f'{OUT}/fig_lambda.png'
plt.savefig(out3, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out3)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET SPLIT PIE  (fig_dataset.png)
# ─────────────────────────────────────────────────────────────────────────────
# from text: 14,312 train / 1,789 val / 1,790 test
sizes  = [14312, 1789, 1790]
labels_pie = ['Training\n14,312 pairs\n(79.9%)',
              'Validation\n1,789 pairs\n(10.0%)',
              'Test\n1,790 pairs\n(10.0%)']
colors_pie = [C_BEST, C_STABLE, '#E65100']
explode    = (0.03, 0.05, 0.05)

fig, ax = plt.subplots(figsize=(6, 4.5))
wedges, texts = ax.pie(sizes, colors=colors_pie, explode=explode,
                       startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax.legend(wedges, labels_pie, loc='lower center', bbox_to_anchor=(0.5, -0.12),
          fontsize=8.5, ncol=3, framealpha=0.9)
ax.set_title('Dataset Split: Training / Validation / Test',
             fontsize=10, fontweight='bold', pad=10)
total = sum(sizes)
ax.text(0, 0, f'Total\n{total:,}', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white')

plt.tight_layout(pad=0.5)
out4 = f'{OUT}/fig_dataset.png'
plt.savefig(out4, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out4)

print('\nAll charts regenerated.')
