# geoseg/datasets/biodiversity_tiff_dataset.py
from __future__ import annotations

from .transform import *  # provides Compose, RandomScale, SmartCropV1, etc.

import os
import os.path as osp
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as albu
from PIL import Image

# Optional: if rasterio isn't installed, we fall back to PIL for TIFF reading
try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None

# Default tile size (your tiles are 512×512)
ORIGIN_IMG_SIZE: Tuple[int, int] = (512, 512)

# --- taxonomy (6 classes, with Background at 0) ---
CLASSES = (
    "Background",
    "Forest land",
    "Grassland",
    "Cropland",
    "Settlement",
    "Seminatural Grassland",
)

PALETTE = [
    [0, 0, 0],         # Background
    [250, 62, 119],    # Forest
    [168, 232, 84],    # Grassland
    [242, 180, 92],    # Cropland
    [116, 116, 116],   # Settlement
    [255, 214, 33],    # Seminatural
]


def get_training_transform():
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.25
            ),
            albu.Normalize(),
        ]
    )


def get_val_transform():
    return albu.Compose([albu.Normalize()])


def train_aug_random(img: Image.Image, mask: Image.Image):
    """
    Stage 1–3: purely random crop + standard aug (no minority targeting).
    Assumes you use ignore_index=255 in SmartCropV1, but your biodiversity masks
    do NOT use 255 by default (background is a real class 0).
    """
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False),
        ]
    )
    img, mask = crop_aug(img, mask)

    img_np, mask_np = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img_np.copy(), mask=mask_np.copy())
    return aug["image"], aug["mask"]


def train_aug_minority(img: Image.Image, mask: Image.Image):
    """
    Stage 4+: 70% minority-centred crop, 30% random crop + standard aug.
    Minority classes: Settlement=4, Seminatural=5
    """
    crop_size = 512
    use_strategic = random.random() < 0.7

    # scale first so coords match
    scale_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value")])
    img, mask = scale_aug(img, mask)

    if use_strategic:
        mask_np = np.array(mask)
        minority_coords = np.argwhere(np.isin(mask_np, [4, 5]))

        if len(minority_coords) == 0:
            return train_aug_random(img, mask)

        center_y, center_x = minority_coords[random.randint(0, len(minority_coords) - 1)]

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        y1 = max(0, center_y - crop_size // 2)
        x1 = max(0, center_x - crop_size // 2)
        y2 = min(h, y1 + crop_size)
        x2 = min(w, x1 + crop_size)

        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)

        img_np = img_np[y1:y2, x1:x2]
        mask_np = mask_np[y1:y2, x1:x2]

        # pad if needed
        if img_np.shape[0] < crop_size or img_np.shape[1] < crop_size:
            pad_h = max(0, crop_size - img_np.shape[0])
            pad_w = max(0, crop_size - img_np.shape[1])
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask_np = np.pad(
                mask_np, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

        aug = get_training_transform()(image=img_np.copy(), mask=mask_np.copy())
        return aug["image"], aug["mask"]

    return train_aug_random(img, mask)


def val_aug(img: Image.Image, mask: Image.Image):
    img_np, mask_np = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img_np.copy(), mask=mask_np.copy())
    return aug["image"], aug["mask"]


# Backward-compatible alias (some wrappers import train_aug)
train_aug = train_aug_random


# -----------------------------
# TIFF helpers
# -----------------------------
def _normalize_per_band_percentile(img_data: np.ndarray) -> np.ndarray:
    """
    Normalize each band to 0–1 using 2nd–98th percentile on nonzero, non-nan pixels.
    img_data: H x W x C
    """
    norm = np.zeros_like(img_data, dtype=np.float32)
    for c in range(img_data.shape[2]):
        band = img_data[:, :, c].astype(np.float32)
        valid = band[~np.isnan(band)]
        valid = valid[valid != 0]

        if valid.size > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            band = np.clip(band, p2, p98)
            if p98 > p2:
                band = (band - p2) / (p98 - p2)

        norm[:, :, c] = band
    return norm


def _read_tif_as_rgb_uint8(path: str) -> Image.Image:
    """
    Reads a .tif and returns an RGB PIL Image (uint8).
    - If 4 bands exist (e.g., RGB+NIR), we keep first 3.
    - Uses rasterio if available; falls back to PIL otherwise.
    """
    if rasterio is not None:
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
            data = np.transpose(data, (1, 2, 0))  # (H, W, C)

            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, 0, data)

            data = np.where(np.isnan(data), 0, data)
            data = _normalize_per_band_percentile(data)
            data = (data * 255.0).clip(0, 255).astype(np.uint8)

            if data.shape[2] >= 3:
                data = data[:, :, :3]
            else:
                data = np.repeat(data, 3, axis=2)

            return Image.fromarray(data)

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# -----------------------------
# Dataset base (image+mask)
# -----------------------------
class _BiodiversityTiffSegDataset(Dataset):
    """
    Expects:
      data_root/
        images/*.tif
        masks/*.png
    """

    def __init__(
        self,
        data_root: str,
        img_dir: str = "images",
        mask_dir: str = "masks",
        img_suffix: str = ".tif",
        mask_suffix: str = ".png",
        transform=None,
        mosaic_ratio: float = 0.0,
        img_size: Tuple[int, int] = ORIGIN_IMG_SIZE,
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size

        self.img_ids = self.get_img_ids()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
        # mosaic hook (not implemented; kept for API compatibility)
        img, mask = self.load_img_and_mask(index)

        if self.transform:
            img_np, mask_np = self.transform(img, mask)
        else:
            img_np, mask_np = np.array(img), np.array(mask)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask_np).long()

        img_id = self.img_ids[index]
        return {"img": img_t, "gt_semantic_seg": mask_t, "img_id": img_id}

    def get_img_ids(self) -> List[str]:
        img_path = osp.join(self.data_root, self.img_dir)
        mask_path = osp.join(self.data_root, self.mask_dir)

        if not osp.isdir(img_path):
            raise FileNotFoundError(f"Missing images dir: {img_path}")
        if not osp.isdir(mask_path):
            raise FileNotFoundError(f"Missing masks dir: {mask_path}")

        img_files = sorted([f for f in os.listdir(img_path) if f.lower().endswith(self.img_suffix)])
        mask_files = set([f for f in os.listdir(mask_path) if f.lower().endswith(self.mask_suffix)])

        ids: List[str] = []
        for f in img_files:
            stem = osp.splitext(f)[0]
            if f"{stem}{self.mask_suffix}" in mask_files:
                ids.append(stem)

        print(f"[Biodiversity] Found {len(ids)} matching image-mask pairs in {self.data_root}")
        return ids

    def load_img_and_mask(self, index: int) -> Tuple[Image.Image, Image.Image]:
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = _read_tif_as_rgb_uint8(img_path)
        mask = Image.open(mask_path).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        # IMPORTANT: Biodiversity uses 0 as a real class (Background),
        # so we DO NOT remap 0 -> 255.
        return img, mask


# -----------------------------
# Public datasets (use your new split roots by default)
# -----------------------------
class BiodiversityTiffTrainDataset(_BiodiversityTiffSegDataset):
    def __init__(
        self,
        data_root: str = "data/biodiversity_split/train",
        transform=train_aug_random,
        mosaic_ratio: float = 0.25,
        img_size: Tuple[int, int] = ORIGIN_IMG_SIZE,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            transform=transform,
            mosaic_ratio=mosaic_ratio,
            img_size=img_size,
            **kwargs,
        )


class BiodiversityTiffValDataset(_BiodiversityTiffSegDataset):
    def __init__(
        self,
        data_root: str = "data/biodiversity_split/val",
        transform=val_aug,
        mosaic_ratio: float = 0.0,
        img_size: Tuple[int, int] = ORIGIN_IMG_SIZE,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            transform=transform,
            mosaic_ratio=mosaic_ratio,
            img_size=img_size,
            **kwargs,
        )


class BiodiversityTiffTestDataset(Dataset):
    """
    Expects:
      data_root/
        images/*.tif
    (No masks)
    """

    def __init__(
        self,
        data_root: str = "data/biodiversity_split/test",
        img_dir: str = "images",
        img_suffix: str = ".tif",
        transform=None,
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.transform = transform if transform is not None else get_val_transform()

        img_path = osp.join(self.data_root, self.img_dir)
        if not osp.isdir(img_path):
            raise FileNotFoundError(f"Missing images dir: {img_path}")

        img_files = sorted([f for f in os.listdir(img_path) if f.lower().endswith(self.img_suffix)])
        self.img_ids = [osp.splitext(f)[0] for f in img_files]

        print(f"[BiodiversityTest] Found {len(self.img_ids)} images in {self.data_root}")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)

        img = _read_tif_as_rgb_uint8(img_path)
        img_np = np.array(img)

        if self.transform:
            aug = self.transform(image=img_np)
            img_np = aug["image"]

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return {"img": img_t, "img_id": img_id}



# -----------------------------
# Compatibility aliases (so configs can keep using old names)
# -----------------------------
BiodiversityTrainDataset = BiodiversityTiffTrainDataset
BiodiversityValDataset = BiodiversityTiffValDataset
BiodiversityTestDataset = BiodiversityTiffTestDataset
