from typing import Dict

import torch

from src.base import CTRModel
from src.utils import validate_epoch


def validate_feat_remove(
    loader,
    model: CTRModel,
    feat_to_remove: torch.LongTensor,
    device="cuda",
) -> Dict[str, float]:
    model.remove_feat(feat_to_remove)
    result = validate_epoch(loader, model, device)
    model.recover()
    return result


def validate_feat_keep(
    loader,
    model: CTRModel,
    feat_to_keep: tuple[int, ...],
    device="cuda",
) -> Dict[str, float]:
    all_feats = model.field_dims.sum().item()
    all_feats = torch.ones(all_feats, dtype=torch.bool)

    if len(feat_to_keep) > 0:
        feat_to_keep = torch.tensor(feat_to_keep)
        all_feats[feat_to_keep] = 0

    feat_to_remove = torch.where(all_feats)[0]
    result = validate_feat_remove(loader, model, feat_to_remove, device)

    return result


def validate_field_remove(
    loader,
    model: CTRModel,
    field_to_remove,
    device="cuda",
) -> Dict[str, float]:
    feat_to_removes = map_from_field_to_feats(model, field_to_remove)
    return validate_feat_remove(loader, model, feat_to_removes, device)


def validate_field_keep(
    loader,
    model: CTRModel,
    field_to_keep,
    device="cuda",
) -> Dict[str, float]:
    feat_to_keep = map_from_field_to_feats(model, field_to_keep)
    return validate_feat_keep(loader, model, feat_to_keep, device)


def map_from_field_to_feats(model, fields):
    field_ends = torch.cumsum(model.field_dims, 0)

    feats = []
    for i in fields:
        end = field_ends[i]
        start = 0
        if i > 0:
            start = field_ends[i - 1]
        feats.extend(range(start, end))
    return feats
