# geoseg/datasets/biodiversity_oem_dataset.py
from __future__ import annotations

"""
Combined dataset for OEM-pretraining:
- Biodiversity split + OEM relabelled into the same 6-class taxonomy (0..5)

Expected layout:
  data/biodiversity_oem_combined/
    train/images/*.tif
    train/masks/*.png
    val/images/*.tif
    val/masks/*.png
"""

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from geoseg.datasets.biodiversity_dataset import (
    ORIGIN_IMG_SIZE,
    train_aug_random,
    val_aug,
    _read_tif_as_rgb_uint8,
)

class _CombinedSegDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        img_dir: str = "images",
        mask_dir: str = "masks",
        img_suffix: str = ".tif",
        mask_suffix: str = ".png",
        transform=None,
        img_size: Tuple[int, int] = ORIGIN_IMG_SIZE,
        **kwargs,
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size

        self.img_ids = self._get_img_ids()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img, mask = self._load_img_and_mask(index)

        if self.transform:
            img_np, mask_np = self.transform(img, mask)
        else:
            img_np, mask_np = np.array(img), np.array(mask)

        img_np = np.ascontiguousarray(img_np)
        mask_np = np.ascontiguousarray(mask_np)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask_np).long()

        img_id = self.img_ids[index]
        return {"img": img_t, "gt_semantic_seg": mask_t, "img_id": img_id}

    def _get_img_ids(self) -> List[str]:
        img_path = osp.join(self.data_root, self.img_dir)
        mask_path = osp.join(self.data_root, self.mask_dir)

        if not osp.isdir(img_path):
            raise FileNotFoundError(f"Missing images dir: {img_path}")
        if not osp.isdir(mask_path):
            raise FileNotFoundError(f"Missing masks dir: {mask_path}")

        img_files = sorted(f for f in os.listdir(img_path) if f.lower().endswith(self.img_suffix))
        mask_files = set(f for f in os.listdir(mask_path) if f.lower().endswith(self.mask_suffix))

        ids: List[str] = []
        for f in img_files:
            stem = osp.splitext(f)[0]
            if f"{stem}{self.mask_suffix}" in mask_files:
                ids.append(stem)

        print(f"[BiodiversityOEM] Found {len(ids)} pairs in {self.data_root}")
        return ids

    def _load_img_and_mask(self, index: int) -> Tuple[Image.Image, Image.Image]:
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = _read_tif_as_rgb_uint8(img_path)
        mask = Image.open(mask_path).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        return img, mask


class BiodiversityOEMTrainDataset(_CombinedSegDataset):
    def __init__(
        self,
        data_root: str = "data/biodiversity_oem_combined/train",
        transform=train_aug_random,
        **kwargs,
    ):
        super().__init__(data_root=data_root, transform=transform, **kwargs)


class BiodiversityOEMValDataset(_CombinedSegDataset):
    def __init__(
        self,
        data_root: str = "data/biodiversity_oem_combined/val",
        transform=val_aug,
        **kwargs,
    ):
        super().__init__(data_root=data_root, transform=transform, **kwargs)
