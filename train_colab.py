"""
train_colab.py  —  One-file launcher for Google Colab / Kaggle
===============================================================
Paste this into a notebook cell-by-cell, or run as a plain script.

Steps it performs:
  1. Install dependencies
  2. Connect to Kaggle dataset (or Google Drive)
  3. Symlink data/ to the mounted dataset
  4. Run verify_setup → train → evaluate

Usage on Kaggle:
  - Add this repo as a "Code" input to your notebook
  - Add your dataset as a "Data" input
  - Run the cells below in order

Usage on Colab:
  - Upload the project folder to Google Drive
  - Mount Drive and run the cells below
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

pip('torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu118')
pip('scikit-image', 'pyyaml', 'tqdm', 'tensorboard', 'psutil')

print("✅ Dependencies installed")

# ============================================================
# CELL 2 — Mount dataset  (edit paths for your environment)
# ============================================================
import os, shutil
from pathlib import Path

# ---------- KAGGLE ----------
# If running on Kaggle, dataset is already at /kaggle/input/<dataset-name>/
# Update KAGGLE_DATASET_PATH to match your dataset slug
KAGGLE_DATASET_PATH = '/kaggle/input/bangla-handwriting-pix2pix'   # ← change this

# ---------- COLAB -----------
# If on Colab, mount Drive first:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   COLAB_DATASET_PATH = '/content/drive/MyDrive/pix2pix/data'

# Auto-detect environment
if os.path.exists('/kaggle'):
    DATASET_PATH = KAGGLE_DATASET_PATH
    WORK_DIR     = '/kaggle/working/pix2pix'
elif os.path.exists('/content'):
    DATASET_PATH = '/content/drive/MyDrive/pix2pix'   # ← change for Colab
    WORK_DIR     = '/content/pix2pix'
else:
    # Local — already set up
    DATASET_PATH = './data'
    WORK_DIR     = '.'

os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f"Working directory: {WORK_DIR}")
print(f"Dataset path:      {DATASET_PATH}")

# Create data/ symlink pointing at the mounted dataset
data_link = Path(WORK_DIR) / 'data'
if not data_link.exists():
    if os.path.exists(DATASET_PATH):
        os.symlink(DATASET_PATH, str(data_link))
        print(f"✅ data/ → {DATASET_PATH}")
    else:
        print(f"⚠️  Dataset not found at {DATASET_PATH}")
        print("   Update DATASET_PATH above and re-run this cell.")

# ============================================================
# CELL 3 — Copy project scripts (only needed on Kaggle/Colab)
# ============================================================
# On Kaggle add the repo as a Code input; scripts appear at /kaggle/input/<repo>/
REPO_PATH = '/kaggle/input/bangla-pix2pix-code'   # ← change to your code input slug

if os.path.exists(REPO_PATH):
    for f in os.listdir(REPO_PATH):
        if f.endswith('.py') or f.endswith('.yaml'):
            shutil.copy(os.path.join(REPO_PATH, f), WORK_DIR)
    print(f"✅ Scripts copied from {REPO_PATH}")
else:
    print("ℹ️  Scripts not copied — assumed to be in current directory.")

# ============================================================
# CELL 4 — Verify setup
# ============================================================
os.system('python verify_setup.py')

# ============================================================
# CELL 5 — Choose a config and train
# ============================================================
# Quick smoke-test (2000 samples, 1 epoch, ~20-40 min on GPU):
#   CONFIG = 'config_quick_test.yaml'

# Full training (14 356 samples, 10 epochs, ~4-8 h on T4):
CONFIG = 'config.yaml'

# Override epochs from CLI for flexibility:
# EPOCHS = 10

os.system(f'python train.py --config {CONFIG}')

# ============================================================
# CELL 6 — Evaluate latest checkpoint
# ============================================================
import glob

checkpoints = sorted(glob.glob('checkpoints/checkpoint_epoch_*.pth'))
if checkpoints:
    latest = checkpoints[-1]
    print(f"Evaluating checkpoint: {latest}")
    os.system(f'python evaluate.py --config {CONFIG} --checkpoint {latest} '
              f'--output eval_results --num-comparisons 30')
else:
    print("No checkpoints found yet.")

# ============================================================
# CELL 7 — Launch TensorBoard  (Colab only)
# ============================================================
# Uncomment in a Colab notebook:
# %load_ext tensorboard
# %tensorboard --logdir ./runs
