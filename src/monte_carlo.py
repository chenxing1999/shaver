"""Source code in calculate CTRModel"""

from typing import Optional, cast

import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .base import CTRModel
from .imputers import DefaultImputer, Imputer
from .utils import get_mask, validate_epoch


def monte_carlo_shapley(
    model: CTRModel,
    loader: DataLoader,
    n_iters: int,
    imputer: Optional[Imputer] = None,
    device: Optional[torch.device] = None,
    verbose=True,
    metric="log_loss",
) -> torch.Tensor:
    """
    Algorithm using Monte Carlo sampling + Permutation to estimate
        SAGE for loader


    Args:

    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    device = cast(torch.device, device)

    model.to(device)
    model.eval()

    if imputer is None:
        imputer = DefaultImputer(model)

    level = "feat"

    # num_rows, num_cols = emb_table.shape[0], emb_table.shape[1]
    num_rows, num_cols = model.get_emb_size()

    if level == "feat":
        total_element = num_rows
    elif level == "weight":
        total_element = num_rows * num_cols
    else:
        raise NotImplementedError

    # calculate v_n
    init_v = validate_epoch(loader, model, device)[metric]
    sum_val = torch.zeros(total_element)
    count = torch.zeros(total_element, dtype=torch.int64)

    try:
        for i in tqdm.tqdm(
            range(n_iters), total=n_iters, ascii=True, disable=not verbose
        ):
            prev_v = init_v

            # outer loop
            permutation = torch.randperm(total_element, device=device)
            for j in range(total_element):
                # the next feat not in list set
                model.remove_feat(permutation[: j + 1])
                next_v = validate_epoch(loader, model, device)[metric]
                model.recover()

                diff_v = prev_v - next_v

                cur_idx = permutation[j].item()
                sum_val[cur_idx] += diff_v
                count[cur_idx] += 1

                prev_v = next_v
    except KeyboardInterrupt:
        print("interupt")

    return sum_val / (count + 1e-12)


def monte_carlo_shapley2(
    model: CTRModel,
    loader: DataLoader,
    n_iters: int,
    imputer: Optional[Imputer] = None,
    device: Optional[torch.device] = None,
    verbose=True,
    metric: str = "log_loss",
) -> torch.Tensor:
    """
    Truncated MonteCarlo with assumption that when value < Null mean that it is not significant.

    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    device = cast(torch.device, device)

    model.to(device)
    model.eval()

    if imputer is None:
        imputer = DefaultImputer(model)

    level = "feat"

    # num_rows, num_cols = emb_table.shape[0], emb_table.shape[1]
    num_rows, num_cols = model.get_emb_size()

    if level == "feat":
        total_element = num_rows
    elif level == "weight":
        total_element = num_rows * num_cols
    else:
        raise NotImplementedError

    # calculate v_n
    init_v = validate_epoch(loader, model, device)[metric]
    sum_val = torch.zeros(total_element)
    count = torch.zeros(total_element, dtype=torch.int64)

    # Get set features
    # feat_set = torch.empty(0, dtype=torch.int64)
    # for x, y in loader:
    #     feat_set = torch.concat([feat_set, torch.unique(x)])
    #     feat_set = torch.unique(feat_set)

    # feat_list = feat_set.tolist()
    # print("Num feat", len(feat_list))

    permutation = torch.randperm(total_element, device=device)
    model.remove_feat(permutation)
    null = validate_epoch(loader, model, device)[metric]
    threshold = null
    model.recover()

    main_bar = tqdm.tqdm(range(n_iters), total=n_iters, ascii=True, disable=not verbose)
    for i in main_bar:
        prev_v = init_v

        # outer loop
        permutation = torch.randperm(total_element, device=device)
        for j in tqdm.tqdm(range(total_element), ascii=True, disable=True):
            diff_v = 1

            if (metric == "log_loss" and prev_v > threshold) or (
                metric == "auc" and prev_v < threshold
            ):
                # if diff_v != 0:
                #     print(threshold, prev_v, j)
                next_v = prev_v
                diff_v = 0
            else:
                model.remove_feat(permutation[: j + 1])
                next_v = validate_epoch(loader, model, device)[metric]
                model.recover()

                diff_v = prev_v - next_v

            cur_idx = permutation[j].item()
            sum_val[cur_idx] += diff_v
            count[cur_idx] += 1

            prev_v = next_v

    return sum_val / (count + 1e-12)
    # return sum_val


def monte_carlo_shapley3(
    model: CTRModel,
    loader: DataLoader,
    n_iters: int,
    imputer: Optional[Imputer] = None,
    device: Optional[torch.device] = None,
    verbose=True,
) -> torch.Tensor:
    """
    Algorithm using Monte Carlo sampling + Permutation to estimate
        SAGE for loader


    Args:

    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    device = cast(torch.device, device)

    model.to(device)
    model.eval()

    if imputer is None:
        imputer = DefaultImputer(model)

    level = "feat"

    # num_rows, num_cols = emb_table.shape[0], emb_table.shape[1]
    num_rows, num_cols = model.get_emb_size()

    if level == "feat":
        total_element = num_rows
    elif level == "weight":
        total_element = num_rows * num_cols
    else:
        raise NotImplementedError

    # calculate v_n
    init_v = validate(model, loader, device)
    sum_val = torch.zeros(total_element, device=device)

    num_fields = len(model.field_dims)
    sum_fields = torch.zeros(num_fields, device=device)

    count = torch.zeros(total_element, dtype=torch.int64, device=device)

    for i in tqdm.tqdm(range(n_iters), total=n_iters, ascii=True, disable=not verbose):
        prev_v = init_v

        # outer loop
        permutation = torch.randperm(num_fields, device=device)
        # permutation = torch.tensor([0,1,2], device=device)
        for j in range(num_fields):
            cur_idx = permutation[j].item()

            n_fields = model.field_dims[cur_idx].item()
            loss_vec = torch.zeros(n_fields, device=device)
            offsets = model.field_dims[:cur_idx].sum().item()

            n_data = 0
            count = torch.zeros(n_fields, dtype=torch.int64, device=device)

            for x, y in loader:
                x, y = x.to(device), y.to(device)

                S = torch.ones_like(x, dtype=torch.bool, device=device)
                S[:, permutation[: j + 1]] = 0

                y_pred = imputer(x, S)
                loss = F.binary_cross_entropy(y_pred, y, reduction="none")
                loss_vec.index_add_(dim=0, index=x[:, cur_idx] - offsets, source=loss)
                count.index_add_(
                    dim=0,
                    index=x[:, cur_idx] - offsets,
                    source=torch.ones_like(loss, dtype=torch.int64),
                )

                n_data += x.shape[0]

            # import pdb; pdb.set_trace()
            next_v = loss_vec.sum().item() / n_data
            # loss_vec /= count
            # loss_vec = loss_vec - (prev_v * n_data) / count

            loss_vec /= n_data
            loss_vec = loss_vec - prev_v / n_fields

            sum_val[offsets : offsets + n_fields] += loss_vec
            sum_fields[cur_idx] += next_v - prev_v
            prev_v = next_v

    print(sum_fields / n_iters)
    return sum_val.cpu() / (n_iters + 1e-12)


def value_function(
    model,
    loader,
    feat_to_keeps: torch.Tensor,
    imputer: Imputer,
    device: torch.device,
) -> float:
    """Feature level pruning value function for Shapley value

    Args:
        model
        loader
        feat_to_keeps: torch.IntTensor - Shape: K
            List of feature to be kept (S)

        samples: torch.LongTensor - Shape: N x #Fields

    Return: v(feat_to_keeps)
        Estimate loss if we only use feature in feat_to_keeps
    """

    loss: float = 0
    n_data = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        S = get_mask(x, feat_to_keeps, level="feat")

        y_pred = imputer(x, S)

        loss += F.binary_cross_entropy(y_pred, y, reduction="sum").item()
        n_data += x.shape[0]
    return loss / n_data
