"""
BUSIDataset — Breast Ultrasound Image Segmentation Dataset loader.

Dataset structure expected:
    <data_dir>/
        benign/
            benign (1).png
            benign (1)_mask.png
            ...
        malignant/
            malignant (1).png
            malignant (1)_mask.png
            ...
        normal/
            normal (1).png
            normal (1)_mask.png   <- all-zero for normal cases
            ...

Multiple masks per image (e.g. benign (5)_mask_1.png, benign (5)_mask_2.png)
are merged with a logical-OR into one binary mask.
"""
import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ── Stats computed on the full BUSI dataset (grayscale, [0,1]) ─────────────
BUSI_MEAN = 0.3329
BUSI_STD  = 0.2196


def _load_gray(path: str) -> np.ndarray:
    """Load image as float32 grayscale in [0, 1]."""
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def _load_mask(path: str) -> np.ndarray:
    """Load mask as uint8 binary (0 / 1)."""
    return (np.array(Image.open(path).convert('L'), dtype=np.float32) > 127).astype(np.uint8)


def find_busi_pairs(data_dir: str,
                    categories: tuple = ('benign', 'malignant', 'normal')):
    """
    Scan a BUSI dataset folder and return a list of
        (image_path, [mask_path_1, ...], category)
    tuples.  Multiple mask files for the same image are listed together.
    """
    pairs = []
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        files = sorted(os.listdir(cat_dir))
        images = [f for f in files if '_mask' not in f and f.lower().endswith('.png')]
        for img_name in images:
            stem     = os.path.splitext(img_name)[0]
            img_path = os.path.join(cat_dir, img_name)
            masks    = sorted(glob.glob(os.path.join(cat_dir, f"{stem}_mask*.png")))
            pairs.append((img_path, masks, cat))
    return pairs


class BUSIDataset(Dataset):
    """
    PyTorch Dataset for binary BUSI segmentation.

    Args:
        list_file : Path to a .txt file, one *relative* image path per line.
                    Relative to ``data_dir``.
        data_dir  : Root directory of the BUSI dataset.
        mode      : ``'train'``, ``'val'``, or ``'test'``.
        img_size  : Square spatial size after resize (default 256).
    """

    def __init__(self, list_file: str, data_dir: str,
                 mode: str = 'train', img_size: int = 256):
        self.data_dir = data_dir
        self.mode     = mode
        self.img_size = img_size

        with open(list_file) as fh:
            rel_paths = [l.strip() for l in fh if l.strip()]

        self.samples = []
        for rel in rel_paths:
            img_path = os.path.join(data_dir, rel)
            stem     = os.path.splitext(img_path)[0]
            masks    = sorted(glob.glob(stem + '_mask*.png'))
            self.samples.append((img_path, masks))

    # ── public interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_paths = self.samples[idx]

        # --- load --------------------------------------------------------
        image = _load_gray(img_path)
        H, W  = image.shape

        if mask_paths:
            merged = np.zeros((H, W), dtype=np.uint8)
            for mp in mask_paths:
                merged = np.logical_or(merged, _load_mask(mp)).astype(np.uint8)
            mask = merged
        else:
            mask = np.zeros((H, W), dtype=np.uint8)

        # --- to PIL -------------------------------------------------------
        img_pil  = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask * 255)

        # --- augment / resize --------------------------------------------
        img_pil, mask_pil = self._transform(img_pil, mask_pil)

        # --- to tensor + normalise ----------------------------------------
        img_t  = TF.to_tensor(img_pil)                                # [1, H, W]  float32
        img_t  = TF.normalize(img_t, mean=[BUSI_MEAN], std=[BUSI_STD])

        mask_t = torch.from_numpy(
            (np.array(mask_pil) > 127).astype(np.int64)               # [H, W]  long
        )
        return img_t, mask_t

    # ── transforms ─────────────────────────────────────────────────────────

    def _transform(self, img: Image.Image, mask: Image.Image):
        # 1. Resize
        img  = TF.resize(img,  [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.mode != 'train':
            return img, mask

        # 2. Random horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # 3. Random vertical flip
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # 4. Random rotation ±15 degrees
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,  fill=0)

        # 5. Brightness / contrast jitter (image only)
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, 1.0 + random.uniform(-0.2, 0.2))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img,   1.0 + random.uniform(-0.2, 0.2))

        return img, mask
