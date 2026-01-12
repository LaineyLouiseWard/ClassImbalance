# geoseg/datasets/openearthmap_teacher_dataset.py
from __future__ import annotations

import os
import os.path as osp
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Reuse the SAME TIFF->RGB conversion used everywhere else in your repo
from geoseg.datasets.biodiversity_dataset import _read_tif_as_rgb_uint8

# Optional rasterio for reading TIFF masks reliably
try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None

import albumentations as albu
from geoseg.datasets.transform import Compose, RandomScale, SmartCropV1  # adjust import if your path differs

def _oem_get_training_transform():
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
            albu.Normalize(),
        ]
    )

def _oem_get_val_transform():
    return albu.Compose([albu.Normalize()])

def oem_train_aug(img: Image.Image, mask: Image.Image):
    # Scale + crop to 512 (fixed), then normalize
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False),
        ]
    )
    img, mask = crop_aug(img, mask)

    img_np, mask_np = np.array(img), np.array(mask)
    aug = _oem_get_training_transform()(image=img_np.copy(), mask=mask_np.copy())
    return aug["image"], aug["mask"]

def oem_val_aug(img: Image.Image, mask: Image.Image):
    # Center-ish crop is not implemented; simplest is: just resize to 512 for val
    img_np, mask_np = np.array(img), np.array(mask)

    # resize deterministically to 512×512
    img_np = np.array(img.resize((512, 512), Image.BICUBIC))
    mask_np = np.array(mask.resize((512, 512), Image.NEAREST))

    aug = _oem_get_val_transform()(image=img_np.copy(), mask=mask_np.copy())
    return aug["image"], aug["mask"]
   


# OpenEarthMap "native" taxonomy (you saw labels 0..8)
# Keep names simple; they're only for logging/metrics display.
OEM_CLASSES_9 = (
    "Background",   # 0
    "Bareland",     # 1
    "Rangeland",    # 2
    "Developed",    # 3
    "Road",         # 4
    "Tree",         # 5
    "Water",        # 6
    "Agriculture",  # 7
    "Building",     # 8  (some OEM variants have 9 total incl. building)
)


def _read_mask_tif_as_label(path: str) -> Image.Image:
    """
    Read a TIFF mask as single-channel label image (uint8/uint16 -> we keep ints).
    Returns PIL Image in mode 'L' (0..255) if possible; otherwise uses numpy->PIL.
    """
    if rasterio is not None:
        with rasterio.open(path) as src:
            m = src.read(1)  # first band
            m = np.where(np.isnan(m), 0, m)
            # keep integer labels
            if m.dtype != np.uint8:
                # if labels are small ints, uint8 is fine
                if m.max() <= 255:
                    m = m.astype(np.uint8)
                else:
                    # still okay, but PIL L is 8-bit; clamp defensively
                    m = np.clip(m, 0, 255).astype(np.uint8)
            return Image.fromarray(m, mode="L")

    # fallback
    m = Image.open(path)
    # some tifs open as I;16 etc -> convert by numpy
    m_np = np.array(m)
    if m_np.max() <= 255:
        m_np = m_np.astype(np.uint8)
    else:
        m_np = np.clip(m_np, 0, 255).astype(np.uint8)
    return Image.fromarray(m_np, mode="L")


class _OEMTeacherDataset(Dataset):
    """
    Expects:
      data_root/
        images/*.tif
        masks/*.tif   (or .tiff)
    """

    def __init__(
        self,
        data_root: str,
        img_dir: str = "images",
        mask_dir: str = "masks",
        img_suffix: str = ".tif",
        mask_suffix: str = ".tif",
        transform=None,
        ignore_index: int = 255,
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.ignore_index = ignore_index

        self.img_ids = self._get_img_ids()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img, mask = self._load_img_and_mask(index)

        if self.transform:
            img_np, mask_np = self.transform(img, mask)
        else:
            img_np, mask_np = np.array(img), np.array(mask)

        # ---- critical: make numpy arrays contiguous (and owned) ----
        img_np = np.ascontiguousarray(img_np)
        mask_np = np.ascontiguousarray(mask_np)

        # ---- critical: make torch tensors contiguous too ----
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(mask_np).contiguous().long()

        return {"img": img_t, "gt_semantic_seg": mask_t, "img_id": self.img_ids[index]}

    def _get_img_ids(self) -> List[str]:
        img_path = osp.join(self.data_root, self.img_dir)
        mask_path = osp.join(self.data_root, self.mask_dir)

        if not osp.isdir(img_path):
            raise FileNotFoundError(f"Missing images dir: {img_path}")
        if not osp.isdir(mask_path):
            raise FileNotFoundError(f"Missing masks dir: {mask_path}")

        img_files = sorted([f for f in os.listdir(img_path) if f.lower().endswith(self.img_suffix)])

        # allow .tif or .tiff masks if needed
        mask_files = set(os.listdir(mask_path))

        ids: List[str] = []
        for f in img_files:
            stem = osp.splitext(f)[0]

            cand1 = f"{stem}{self.mask_suffix}"
            cand2 = f"{stem}.tiff" if self.mask_suffix.lower() == ".tif" else f"{stem}.tif"

            if cand1 in mask_files or cand2 in mask_files:
                ids.append(stem)

        print(f"[OEMTeacher] Found {len(ids)} pairs in {self.data_root}")
        return ids

    def _load_img_and_mask(self, index: int) -> Tuple[Image.Image, Image.Image]:
        stem = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, stem + self.img_suffix)

        mask_path_1 = osp.join(self.data_root, self.mask_dir, stem + self.mask_suffix)
        mask_path_2 = osp.join(self.data_root, self.mask_dir, stem + (".tiff" if self.mask_suffix == ".tif" else ".tif"))
        mask_path = mask_path_1 if osp.exists(mask_path_1) else mask_path_2

        img = _read_tif_as_rgb_uint8(img_path)
        mask = _read_mask_tif_as_label(mask_path)

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        return img, mask


# Public splits (match your script output)
class OpenEarthMapTeacherTrainDataset(_OEMTeacherDataset):
    def __init__(
        self,
        data_root: str = "data/openearthmap_teacher/train",
        transform=None,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(data_root=data_root, transform=transform, ignore_index=ignore_index, **kwargs)


class OpenEarthMapTeacherValDataset(_OEMTeacherDataset):
    def __init__(
        self,
        data_root: str = "data/openearthmap_teacher/val",
        transform=None,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(data_root=data_root, transform=transform, ignore_index=ignore_index, **kwargs)

#do I need this?
def _oem_train_albu():
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
            albu.Normalize(),
        ]
    )

def _oem_val_albu():
    return albu.Compose([albu.Normalize()])

