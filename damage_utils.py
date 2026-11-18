"""
damage_utils.py  —  Shared damage functions for dataset generation and online training
======================================================================================
Used by:
  - generate_dataset.py  (offline: pre-generate damaged images to disk)
  - dataset.py           (online:  apply fresh damage each epoch during training)

Keeping damage logic in one place means both paths always use the same preset.
"""

import random
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Damage presets
# ---------------------------------------------------------------------------

PRESETS = {
    'minimal': {
        'sigma':          18.0,
        'fade':           False,
        'heavy_blur':     False,
        'erosion':        False,
        'stains':         False,
        'yellowing':      False,
        'salt_pepper':    False,
        'blur':           False,
        'description':    'Gaussian noise sigma=18 only — subtle grain',
    },
    'medium': {
        'sigma':          35.0,
        'fade':           True,
        'fade_factor':    (0.60, 0.80),
        'heavy_blur':     False,
        'erosion':        False,
        'stains':         False,
        'yellowing':      False,
        'salt_pepper':    False,
        'blur':           True,
        'description':    'Noise sigma=35 + ink fading',
    },
    'demo': {
        'sigma':          40.0,
        'fade':           True,
        'fade_factor':    (0.50, 0.75),
        'heavy_blur':     False,
        'erosion':        False,
        'stains':         True,
        'stain_count':    (3, 6),
        'stain_radius':   (8, 20),
        'yellowing':      True,
        'yellow_strength':(0.10, 0.22),
        'salt_pepper':    True,
        'sp_amount':      0.03,
        'blur':           True,
        'description':    'Noise + fading + stains + yellowing',
    },
    'distorted': {
        'sigma':          20.0,
        'fade':           True,
        'fade_factor':    (0.52, 0.68),
        'heavy_blur':     True,
        'blur_kernel':    (7, 9),
        'erosion':        True,
        'erosion_kernel': (2, 2),
        'stains':         True,
        'stain_count':    (2, 3),
        'stain_radius':   (8, 16),
        'yellowing':      True,
        'yellow_strength':(0.15, 0.25),
        'salt_pepper':    False,
        'blur':           False,
        'description':    'Moderate blur + balanced fading + erosion — aged document',
    },
}


# ---------------------------------------------------------------------------
# Individual damage functions
# ---------------------------------------------------------------------------

def apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_ink_fading(img: np.ndarray, factor_range: tuple) -> np.ndarray:
    """Fade dark ink strokes — simulates old dried ink."""
    factor = random.uniform(*factor_range)
    out = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    stroke_mask = (gray < 180).astype(np.float32)
    for c in range(3):
        faded = out[:, :, c] + (255 - out[:, :, c]) * (1 - factor)
        out[:, :, c] = out[:, :, c] * (1 - stroke_mask) + faded * stroke_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_water_stains(img: np.ndarray,
                       count_range: tuple,
                       radius_range: tuple) -> np.ndarray:
    """Add semi-transparent dark blotches — water damage / aging."""
    out = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    for _ in range(random.randint(*count_range)):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        r  = random.randint(*radius_range)
        alpha = random.uniform(0.15, 0.40)
        color = random.randint(140, 200)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (r | 1, r | 1), r / 3)
        for c in range(3):
            out[:, :, c] = out[:, :, c] * (1 - mask * alpha) + color * (mask * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_yellowing(img: np.ndarray, strength_range: tuple) -> np.ndarray:
    """Tint background yellow — aged paper."""
    strength = random.uniform(*strength_range)
    out = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bg_mask = (gray > 200).astype(np.float32)
    out[:, :, 0] = np.clip(out[:, :, 0] - 10 * strength * bg_mask, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] - 15 * strength * bg_mask, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] - 60 * strength * bg_mask, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_salt_pepper(img: np.ndarray, amount: float) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    n = int(h * w * amount)
    coords = [np.random.randint(0, i, n // 2) for i in (h, w)]
    out[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, n // 2) for i in (h, w)]
    out[coords[0], coords[1]] = 0
    return out


def apply_blur(img: np.ndarray, kernel: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def apply_heavy_blur(img: np.ndarray, kernel_range: tuple) -> np.ndarray:
    """Heavy Gaussian blur — strokes become fuzzy/smeared."""
    k = random.randint(kernel_range[0] // 2, kernel_range[1] // 2) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), k / 3.0)


def apply_erosion(img: np.ndarray, kernel_range: tuple) -> np.ndarray:
    """Morphological erosion — thins ink strokes."""
    k = random.randint(*kernel_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inverted = cv2.bitwise_not(img)
    eroded   = cv2.erode(inverted, kernel, iterations=1)
    return cv2.bitwise_not(eroded)


# ---------------------------------------------------------------------------
# Master damage function
# ---------------------------------------------------------------------------

def make_damaged(img: np.ndarray, preset: dict) -> np.ndarray:
    """Apply all damage layers defined by a preset. img must be uint8 RGB."""
    out = img.copy()
    if preset.get('erosion'):
        out = apply_erosion(out, preset['erosion_kernel'])
    if preset.get('heavy_blur'):
        out = apply_heavy_blur(out, preset['blur_kernel'])
    if preset.get('fade'):
        out = apply_ink_fading(out, preset['fade_factor'])
    out = apply_gaussian_noise(out, preset['sigma'])
    if preset.get('stains'):
        out = apply_water_stains(out, preset['stain_count'], preset['stain_radius'])
    if preset.get('yellowing'):
        out = apply_yellowing(out, preset['yellow_strength'])
    if preset.get('salt_pepper'):
        out = apply_salt_pepper(out, preset['sp_amount'])
    if preset.get('blur'):
        out = apply_blur(out, kernel=3)
    return out


# ---------------------------------------------------------------------------
# Image preparation (shared between offline generation and online loading)
# ---------------------------------------------------------------------------

TARGET_SIZE = (256, 256)


def prepare_crop(img_bgr: np.ndarray) -> np.ndarray:
    """Resize BGR crop to 256x256 on white background, return RGB uint8."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((TARGET_SIZE[1], TARGET_SIZE[0], 3), 255, dtype=np.uint8)
    x_off = (TARGET_SIZE[0] - new_w) // 2
    y_off = (TARGET_SIZE[1] - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
