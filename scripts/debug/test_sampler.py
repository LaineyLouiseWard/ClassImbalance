from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, train_aug_random

# load dataset
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train_rep",
    transform=train_aug_random,
)

# load weights
w_map = {}
with open("artifacts/sample_weights_diff_x_minority.txt") as f:
    for line in f:
        k, w = line.strip().split("\t")
        w_map[k] = float(w)

sample_weights = [w_map[i] for i in train_dataset.img_ids]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# draw samples
idxs = list(iter(sampler))
counts = Counter(idxs)

print("Most sampled indices:", counts.most_common(10))
print("Corresponding img_ids:", [train_dataset.img_ids[i] for i,_ in counts.most_common(10)])
