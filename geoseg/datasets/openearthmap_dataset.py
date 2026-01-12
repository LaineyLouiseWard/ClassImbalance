# geoseg/datasets/openearthmap_dataset.py
from __future__ import annotations

"""
OpenEarthMap teacher dataset (raw OEM labels).

Expects:
  data_root/
    images/*.tif
    masks/*.tif   (raw OEM label tiles, with 255 as ignore/void)

Used for training the teacher on OEM.
"""

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as albu

from geoseg.datasets.biodiversity_dataset import _read_tif_as_rgb_uint8
from geoseg.datasets.transform import Compose, RandomScale, SmartCropV1

try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None


def _train_tf():
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
            albu.Normalize(),
        ]
    )


def _val_tf():
    return albu.Compose([albu.Normalize()])


def oem_train_aug(img: Image.Image, mask: Image.Image):
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False),
        ]
    )
    img, mask = crop_aug(img, mask)
    img_np, mask_np = np.array(img), np.array(mask)
    out = _train_tf()(image=img_np.copy(), mask=mask_np.copy())
    return out["image"], out["mask"]


def oem_val_aug(img: Image.Image, mask: Image.Image):
    img = img.resize((512, 512), Image.BICUBIC)
    mask = mask.resize((512, 512), Image.NEAREST)
    img_np, mask_np = np.array(img), np.array(mask)
    out = _val_tf()(image=img_np.copy(), mask=mask_np.copy())
    return out["image"], out["mask"]


def _read_mask_tif_as_label(path: str) -> Image.Image:
    if rasterio is not None:
        with rasterio.open(path) as src:
            m = src.read(1)

        # Only handle NaNs if the array is floating
        if np.issubdtype(m.dtype, np.floating):
            m = np.where(np.isnan(m), 255, m)

        # OEM labels are small ints + 255 ignore; keep in uint8
        if m.dtype != np.uint8:
            m = np.clip(m, 0, 255).astype(np.uint8)

        return Image.fromarray(m, mode="L")

    # PIL fallback
    m = Image.open(path)
    m_np = np.array(m)
    if np.issubdtype(m_np.dtype, np.floating):
        m_np = np.where(np.isnan(m_np), 255, m_np)
    if m_np.dtype != np.uint8:
        m_np = np.clip(m_np, 0, 255).astype(np.uint8)
    return Image.fromarray(m_np, mode="L")


class _OEMTeacherDataset(Dataset):
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

        img_t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).contiguous().long()

        return {"img": img_t, "gt_semantic_seg": mask_t, "img_id": self.img_ids[index]}

    def _get_img_ids(self) -> List[str]:
        img_path = osp.join(self.data_root, self.img_dir)
        mask_path = osp.join(self.data_root, self.mask_dir)

        if not osp.isdir(img_path):
            raise FileNotFoundError(f"Missing images dir: {img_path}")
        if not osp.isdir(mask_path):
            raise FileNotFoundError(f"Missing masks dir: {mask_path}")

        img_files = sorted([f for f in os.listdir(img_path) if f.lower().endswith(self.img_suffix)])
        mask_files = set(os.listdir(mask_path))

        ids: List[str] = []
        for f in img_files:
            stem = osp.splitext(f)[0]
            cand1 = f"{stem}{self.mask_suffix}"
            cand2 = f"{stem}.tiff" if self.mask_suffix.lower() == ".tif" else f"{stem}.tif"
            if cand1 in mask_files or cand2 in mask_files:
                ids.append(stem)

        return ids

    def _load_img_and_mask(self, index: int) -> Tuple[Image.Image, Image.Image]:
        stem = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, stem + self.img_suffix)

        mask_path_1 = osp.join(self.data_root, self.mask_dir, stem + self.mask_suffix)
        mask_path_2 = osp.join(
            self.data_root,
            self.mask_dir,
            stem + (".tiff" if self.mask_suffix.lower() == ".tif" else ".tif"),
        )
        mask_path = mask_path_1 if osp.exists(mask_path_1) else mask_path_2

        img = _read_tif_as_rgb_uint8(img_path)
        mask = _read_mask_tif_as_label(mask_path)

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        return img, mask


class OpenEarthMapTeacherTrainDataset(_OEMTeacherDataset):
    def __init__(self, data_root: str = "data/openearthmap_teacher/train", **kwargs):
        super().__init__(data_root=data_root, transform=oem_train_aug, ignore_index=255, **kwargs)


class OpenEarthMapTeacherValDataset(_OEMTeacherDataset):
    def __init__(self, data_root: str = "data/openearthmap_teacher/val", **kwargs):
        super().__init__(data_root=data_root, transform=oem_val_aug, ignore_index=255, **kwargs)
