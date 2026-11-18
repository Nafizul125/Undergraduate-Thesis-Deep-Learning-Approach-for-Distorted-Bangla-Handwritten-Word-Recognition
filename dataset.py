"""
dataset.py  —  Dataset classes for Bangla Handwriting Restoration
=================================================================

Two modes controlled by config['dataset']['online_damage']:

  online_damage: true   (RECOMMENDED)
    - Loads clean word crops from crops_dir
    - Applies fresh random damage every __getitem__ call
    - Model sees different noise/blur/stain each epoch
    - Prevents overfitting to fixed damage patterns
    - Points to: data_raw/word_crops/

  online_damage: false  (legacy offline mode)
    - Loads pre-generated AB images (512x256)
    - Same damaged image seen every epoch
    - Can overfit — only use for debugging
    - Points to: data_minimal/train|val|test/
"""

import os
import random
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from damage_utils import make_damaged, prepare_crop, PRESETS


# ---------------------------------------------------------------------------
# Online Dataset  (recommended — fresh damage each epoch)
# ---------------------------------------------------------------------------

class OnlineRestorationDataset(Dataset):
    """
    Loads clean word crops. Applies fresh random damage on every __getitem__.

    Because damage is re-randomised every call:
      - Epoch 1 image_0: damaged with blur kernel=7, fade=0.61, stain at (120,80)
      - Epoch 2 image_0: damaged with blur kernel=9, fade=0.55, stain at (200,40)
    The model cannot memorise — it must learn the general reconstruction rule.
    """

    def __init__(self, root_dir, split='train', image_size=256,
                 damage_preset='distorted', max_samples=None,
                 train_ratio=0.80, val_ratio=0.10, seed=42,
                 augment=False):

        self.root_dir   = root_dir
        self.image_size = image_size
        self.augment    = augment
        self.preset     = dict(PRESETS[damage_preset])
        self.preset['name'] = damage_preset

        # Collect all image files
        all_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not all_files:
            raise ValueError(f"No images found in {root_dir}")

        # Deterministic split — same every run
        rng = random.Random(seed)
        rng.shuffle(all_files)

        n       = len(all_files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        if split == 'train':
            self.image_files = all_files[:n_train]
        elif split == 'val':
            self.image_files = all_files[n_train:n_train + n_val]
        else:
            self.image_files = all_files[n_train + n_val:]

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]

        print(f"  [{split:5s}] {len(self.image_files):>6} clean crops  "
              f"| damage preset: {damage_preset}"
              + (f"  | flip aug: ON" if augment else ""))

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img_bgr  = cv2.imread(img_path)

        if img_bgr is None:
            # Fallback for unreadable files — return white canvas
            clean = np.full((256, 256, 3), 255, dtype=np.uint8)
        else:
            clean = prepare_crop(img_bgr)           # uint8 RGB 256x256

        # Fresh random damage every call — this is the key to generalisation
        damaged = make_damaged(clean, self.preset)  # uint8 RGB 256x256

        # Random horizontal flip — applied to BOTH so they stay aligned
        if self.augment and random.random() < 0.5:
            clean   = np.fliplr(clean).copy()
            damaged = np.fliplr(damaged).copy()

        return {
            'A': self.to_tensor(Image.fromarray(damaged)),  # input  (damaged)
            'B': self.to_tensor(Image.fromarray(clean)),    # target (clean)
            'filename': self.image_files[idx]
        }


# ---------------------------------------------------------------------------
# Offline Dataset  (legacy — loads pre-generated AB images)
# ---------------------------------------------------------------------------

class Pix2PixDataset(Dataset):
    """
    Loads pre-generated 512x256 AB images from disk.
    Left 256px = damaged A, right 256px = clean B.
    Use only for debugging or when online mode is not available.
    """

    def __init__(self, root_dir, image_size=256, augment=False, max_samples=None):
        self.root_dir   = root_dir
        self.image_size = image_size
        self.augment    = augment

        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith('.png')
        ])

        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]

        if not self.image_files:
            raise ValueError(f"No PNG images found in {root_dir}")

        print(f"  Found {len(self.image_files)} images in {root_dir}")

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        ab_image = np.array(Image.open(img_path).convert('RGB'))

        a = Image.fromarray(ab_image[:, :256, :]).resize(
            (self.image_size, self.image_size), Image.BICUBIC)
        b = Image.fromarray(ab_image[:, 256:, :]).resize(
            (self.image_size, self.image_size), Image.BICUBIC)

        if self.augment:
            from PIL import ImageEnhance
            if random.random() < 0.4:
                angle = random.uniform(-3.0, 3.0)
                a = a.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
                b = b.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

        return {
            'A': self.to_tensor(a),
            'B': self.to_tensor(b),
            'filename': self.image_files[idx]
        }


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloaders(config):
    """
    Build train and validation DataLoaders.
    Reads config['dataset']['online_damage'] to choose the dataset class.
    """
    ds_cfg     = config['dataset']
    online     = ds_cfg.get('online_damage', False)
    batch_size = ds_cfg['batch_size']
    num_workers = ds_cfg.get('num_workers', 2)
    pin_memory  = ds_cfg.get('pin_memory', True)

    if online:
        crops_dir     = ds_cfg['crops_dir']
        damage_preset = ds_cfg.get('damage_preset', 'distorted')
        max_train     = ds_cfg.get('max_train_samples', None)
        aug_cfg       = config.get('augmentation', {})
        do_flip       = aug_cfg.get('enable', False) and aug_cfg.get('random_flip', False)

        train_dataset = OnlineRestorationDataset(
            root_dir=crops_dir, split='train',
            image_size=ds_cfg['image_size'],
            damage_preset=damage_preset,
            max_samples=max_train,
            augment=do_flip,        # flip only during training
        )
        val_dataset = OnlineRestorationDataset(
            root_dir=crops_dir, split='val',
            image_size=ds_cfg['image_size'],
            damage_preset=damage_preset,
            augment=False,          # never augment validation
        )

    else:
        max_train = ds_cfg.get('max_train_samples', None)
        train_dataset = Pix2PixDataset(
            root_dir=ds_cfg['train_dir'],
            image_size=ds_cfg['image_size'],
            augment=config['augmentation']['enable'],
            max_samples=max_train,
        )
        val_dataset = Pix2PixDataset(
            root_dir=ds_cfg['val_dir'],
            image_size=ds_cfg['image_size'],
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
