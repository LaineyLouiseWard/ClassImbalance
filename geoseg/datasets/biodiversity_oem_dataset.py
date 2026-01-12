# geoseg/datasets/biodiversity_oem_combined_dataset.py
from __future__ import annotations

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Reuse your Biodiversity TIFF utilities + transforms + taxonomy
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    PALETTE,
    ORIGIN_IMG_SIZE,
    train_aug_random,
    val_aug,
    _read_tif_as_rgb_uint8,
)


class _CombinedTiffSegDataset(Dataset):
    """
    Combined dataset for OEM-pretraining (Biodiversity + OEM already harmonised to 0..5).

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
        self.mosaic_ratio = mosaic_ratio  # kept for config compatibility
        self.img_size = img_size

        self.img_ids = self.get_img_ids()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
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

        print(f"[CombinedOEM] Found {len(ids)} matching image-mask pairs in {self.data_root}")
        return ids

    def load_img_and_mask(self, index: int):
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = _read_tif_as_rgb_uint8(img_path)
        mask = Image.open(mask_path).convert("L")

        # Safety: keep alignment
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        # IMPORTANT:
        # We assume OEM masks were ALREADY mapped into 0..5 (Background=0 is real class)
        # so we DO NOT do any remapping here.
        return img, mask


class BiodiversityOEMCombinedTiffTrainDataset(_CombinedTiffSegDataset):
    """
    Train split for OEM pretraining.
    Default root matches the Stage 5A description:
      data/biodiversity_combined_tiff/Train
    """

    def __init__(
        self,
        data_root: str = "data/biodiversity_combined_tiff/Train",
        transform=train_aug_random,
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


class BiodiversityOEMCombinedTiffValDataset(_CombinedTiffSegDataset):
    """
    Optional combined-val split (if you ever make one).
    Not required for your current pipeline, but provided for symmetry.
    """

    def __init__(
        self,
        data_root: str = "data/biodiversity_combined_tiff/Val",
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


# For config convenience / consistency with your other modules:
BiodiversityCombinedTiffTrainDataset = BiodiversityOEMCombinedTiffTrainDataset
BiodiversityCombinedTiffValDataset = BiodiversityOEMCombinedTiffValDataset
