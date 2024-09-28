from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.base import CTRDataset
from src.const import DATASET_INFO

from .avazu import AvazuDataset
from .criteo import CriteoDataset
from .kdd_dataset import KddDataset


def get_dataset(
    dataset_name: Literal["criteo", "avazu", "kdd"],
    split: Literal["val", "train", "test"],
) -> CTRDataset:
    """Hard coded to get dataset instance quickly"""
    dataset_config = DATASET_INFO[dataset_name].copy()
    dataset_config["dataset_name"] = split
    if dataset_name == "criteo":
        return CriteoDataset(**dataset_config)
    elif dataset_name == "avazu":
        return AvazuDataset(**dataset_config)
    elif dataset_name == "kdd":
        return KddDataset(**dataset_config)
    else:
        raise NotImplementedError()


def get_train_val_merge(
    dataset_name: Literal["criteo", "avazu", "kdd"],
    p_train: float,
    p_val: float = 1.0,
    reverse_val: bool = False,
    seed=2024,
) -> Dataset:
    """Quick utils function to provide merge between train and test"""
    assert p_train >= 0 and p_train <= 1
    train_only = (p_train == 1) and (p_val == 0)
    if train_only:
        train_dataset = get_dataset(dataset_name, "train")
        return train_dataset

    if p_val < 1:
        val_dataset = get_partial_val(dataset_name, p_val, reverse_val, seed)
    else:
        val_dataset = get_dataset(dataset_name, "val")

    if p_train == 0:
        return val_dataset

    train_dataset = get_dataset(dataset_name, "train")
    if p_train == 1:
        val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    elif p_train > 0:
        rng = np.random.default_rng(seed=seed)
        n_data = int(len(train_dataset) * p_train)
        indices = rng.choice(len(train_dataset), size=n_data)
        subset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.ConcatDataset([subset, val_dataset])
    return val_dataset


def get_partial_val(dataset_name, ratio, reverse=False, seed=2024):
    dataset = get_dataset(dataset_name, "val")

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator, dtype=int)

    if reverse:
        perm = perm[::-1]

    n_data = int(len(dataset) * ratio)
    subset = torch.utils.data.Subset(dataset, perm[:n_data])
    return subset
