import sys

import torch

from src.datasets import get_dataset

dataset_name = sys.argv[0]
output_path = sys.argv[1]

assert dataset_name in ["avazu", "criteo", "kdd"]

dataset = get_dataset(dataset_name, "train")
freq = torch.zeros(dataset.field_dims.sum(), dtype=torch.int32)

for x, y in dataset:
    freq[x] += 1

dataset = get_dataset(dataset_name, "val")

for x, y in dataset:
    freq[x] += 1

torch.save(freq, output_path)
