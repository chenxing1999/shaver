from typing import Optional, cast

import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .base import CTRModel
from .imputers import Imputer
from .utils import ImportanceTracker


class RowImputerMixup(Imputer):
    """Convert masked feature to base value"""

    def __init__(
        self,
        model: CTRModel,
        use_sigmoid=True,
        base_value=None,
    ):
        super().__init__()

        self.model = model
        self.use_sigmoid = use_sigmoid

        field_dims = self.model.field_dims

        if base_value is None:
            base_value = []
            start = 0
            end = 0
            for field in field_dims:
                end = end + field
                base_value.append(self.model.embedding.weight[start:end].mean(dim=0))
                start = end

            # shape: 1 x num_fields x hidden_size
            self.base_value = torch.stack(base_value, dim=0).unsqueeze(0)
        else:
            self.base_value = base_value

    def __call__(self, x, S, mixup_indices=None, alpha=1.):
        """
        Args:
            x: shape batch x n_fields
            S: Shape batch x (n_fields x n_cols)
                S=False --> Remove feature
        """

        batch_size, n_fields = x.shape
        if mixup_indices is None:
            mixup_indices = torch.arange(batch_size)
            alpha = 1.

        with torch.no_grad():
            emb = self.model.get_emb(x)
            S = S.view(*emb.shape)

            if isinstance(self.base_value, (float, int)):
                emb[~S] = self.base_value
            else:
                base_value = self.base_value.repeat(batch_size, 1, 1)
                emb[~S] = base_value[~S]

            emb = alpha * emb + (1 - alpha) * emb[mixup_indices]
            y_pred = self.model.head(emb, x)

        if self.use_sigmoid:
            y_pred = torch.sigmoid_(y_pred)
        return y_pred

    @property
    def total_element(self):
        return self.model.get_emb_size()[1] * len(self.model.field_dims)


def sage_shapley(
    model: CTRModel,
    loader: DataLoader,
    n_iters: int,
    imputer: Optional[Imputer] = None,
    device: Optional[torch.device] = None,
    threshold: Optional[float] = None,
    verbose=True,
    mode_marginal=False,
    min_epochs=-1,
) -> torch.Tensor:
    """
    Algorithm using SAGE sampling + Permutation to estimate
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
        imputer = RowDefaultImputer(model)

    # num_rows, num_cols = emb_table.shape[0], emb_table.shape[1]
    total_element = imputer.total_element

    sum_val = torch.zeros(total_element, device=device)
    sum_sq = torch.zeros(total_element, device=device)

    iter_loader = iter(loader)
    std = ratio = 1.0
    bar = tqdm.tqdm(
        range(n_iters),
        total=n_iters,
        ascii=True,
        disable=not verbose,
        desc=f"std={std:.2f} - ratio={ratio:.2f}",
    )
    count = 0
    num_epochs = 0

    for i in bar:
        # sample x and y
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x, y = next(iter_loader)
            num_epochs += 1

        x, y = x.to(device), y.to(device)
        S = torch.zeros((x.shape[0], total_element), dtype=torch.bool, device=device)
        prev_v = F.binary_cross_entropy(
            imputer(x, S),
            y,
            reduction="none",
        )

        permutation = torch.stack(
            [torch.randperm(total_element) for i in range(x.shape[0])]
        )

        for j in range(total_element):
            S[torch.arange(x.shape[0]), permutation[:, j]] = 1
            y_pred = imputer(x, S)
            next_v = F.binary_cross_entropy(y_pred, y, reduction="none")

            diff_v = next_v - prev_v

            cur_idx = permutation[:, j].to(device)

            sum_val.index_add_(dim=0, index=cur_idx, source=diff_v)
            sum_sq.index_add_(dim=0, index=cur_idx, source=diff_v * diff_v)

            prev_v = next_v

        count += x.shape[0]
        if threshold is not None:
            values = sum_val / count
            sq = sum_sq / count
            std_arr = ((sq - values**2) / count).sqrt()
            std = std_arr.max().item()

            gap: float = max(values.max().item() - values.min().item(), 1e-12)
            ratio = std / gap

            if ratio < threshold and num_epochs >= min_epochs:
                return values.cpu(), std_arr.cpu()
            bar.set_description(f"std={std:.2f} - ratio={ratio:.2f}")

    values = sum_val / count
    sq = sum_sq / count
    std_arr = ((sq - values**2) / count).sqrt()
    return values.cpu(), std_arr.cpu()


def sage_shapley_field_ver2(
    model: CTRModel,
    loader: DataLoader,
    n_iters: int,
    imputer: Optional[Imputer] = None,
    device: Optional[torch.device] = None,
    threshold: Optional[float] = None,
    verbose=True,
    mode_marginal=False,
    min_epochs=-1,
    criterion=None,
) -> torch.Tensor:
    """
    Algorithm using SAGE sampling + Permutation to estimate
        SAGE for loader (with Welford aggregrate)


    Args:

    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    device = cast(torch.device, device)
    if criterion is None:
        criterion = torch.nn.BCELoss(reduction="none")

    model.to(device)
    model.eval()

    if imputer is None:
        imputer = RowDefaultImputer(model)

    # num_rows, num_cols = emb_table.shape[0], emb_table.shape[1]
    total_element = imputer.total_element

    emb_size = model.get_emb_size()
    weight_size = emb_size[0] * emb_size[1]
    n_cols = emb_size[1]
    # sum_val = torch.zeros(weight_size, device=device)
    # sum_sq = torch.zeros(weight_size, device=device)

    iter_loader = iter(loader)
    std = ratio = 1.0
    bar = tqdm.tqdm(
        range(n_iters),
        total=n_iters,
        ascii=True,
        disable=not verbose,
        desc=f"std={std:.2f} - ratio={ratio:.2f}",
    )
    count = 0
    num_epochs = 0
    max_diff = 0

    tracker = ImportanceTracker()

    for i in bar:
        # sample x and y
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x, y = next(iter_loader)
            num_epochs += 1

        x, y = x.to(device), y.to(device)
        S = torch.ones((x.shape[0], total_element), dtype=torch.bool, device=device)
        # S = torch.zeros((x.shape[0], total_element), dtype=torch.bool, device=device)
        prev_v = criterion(
            imputer(x, S),
            y,
        )

        permutation = torch.stack(
            [torch.randperm(total_element) for i in range(x.shape[0])]
        )

        scores = torch.zeros((x.shape[0], weight_size), device=device)
        for j in range(total_element):
            # Remove feature perm[j]
            S[torch.arange(x.shape[0]), permutation[:, j]] = 0
            # S[torch.arange(x.shape[0]), permutation[:, j]] = 1

            y_pred = imputer(x, S)
            next_v = criterion(y_pred, y)

            diff_v = next_v - prev_v
            max_diff = max(diff_v.max().item(), max_diff)

            cur_idx = permutation[:, j].to(device)
            cur_field_idx = cur_idx // n_cols
            cur_cols = cur_idx % n_cols

            # permutation: weight * 16
            # cur_idx: batch x 16
            cur_field = x[torch.arange(x.shape[0]), cur_field_idx]
            cur_idx = cur_field * n_cols + cur_cols

            # sum_val.index_add_(dim=0, index=cur_idx, source=diff_v)
            # sum_sq.index_add_(dim=0, index=cur_idx, source=diff_v * diff_v)
            scores[torch.arange(x.shape[0]), cur_idx] = diff_v

            prev_v = next_v

        tracker.update(scores)
        count += x.shape[0]
        if threshold is not None:
            values = tracker.values
            std_arr = tracker.std
            std = std_arr.max().item()

            gap: float = max(values.max().item() - values.min().item(), 1e-12)
            ratio = std / gap

            if ratio < threshold and num_epochs >= min_epochs:
                return values.cpu(), std_arr.cpu()
            bar.set_description(f"std={std:.2f} - ratio={ratio:.2f}")

    values = tracker.values
    std_arr = tracker.std
    std = std_arr.max().item()

    return values.cpu(), std_arr.cpu()


def sage_shapley_field_ver(
    model: CTRModel,
    loader: DataLoader,
    n_iters: int,
    imputer: RowImputerMixup,
    device: Optional[torch.device] = None,
    threshold: Optional[float] = None,
    verbose=True,
    mode_marginal=False,
    min_epochs=-1,
    criterion=None,
) -> torch.Tensor:
    """
    Algorithm using SAGE sampling + Permutation to estimate
        SAGE for loader


    Args:

    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    device = cast(torch.device, device)
    if criterion is None:
        criterion = torch.nn.BCELoss(reduction="none")

    model.to(device)
    model.eval()

    # num_rows, num_cols = emb_table.shape[0], emb_table.shape[1]
    total_element = imputer.total_element

    emb_size = model.get_emb_size()
    weight_size = emb_size[0] * emb_size[1]
    n_cols = emb_size[1]
    sum_val = torch.zeros(weight_size, device=device)
    sum_sq = torch.zeros(weight_size, device=device)

    iter_loader = iter(loader)
    std = ratio = 1.0
    bar = tqdm.tqdm(
        range(n_iters),
        total=n_iters,
        ascii=True,
        disable=not verbose,
        desc=f"std={std:.2f} - ratio={ratio:.2f}",
    )
    count = 0
    num_epochs = 0
    max_diff = 0

    for i in bar:
        # sample x and y
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x, y = next(iter_loader)
            num_epochs += 1

        x, y = x.to(device), y.to(device)

        batch_size = x.shape[0]
        mixup_indices = torch.randperm(batch_size)

        # a batch share the same alpha
        alpha = torch.rand(1, device=device)

        # S[i, j] = 0 --> Prune (i, j)
        S = torch.ones((x.shape[0], total_element), dtype=torch.bool, device=device)

        y_pred = imputer(x, S, mixup_indices, alpha)
        # prev_v = alpha * criterion(y_pred, y) + (1 - alpha) * criterion(y_pred, y[mixup_indices])
        prev_v1 = criterion(y_pred, y)
        prev_v2 = criterion(y_pred, y[mixup_indices])

        permutation = torch.stack(
            [torch.randperm(total_element) for i in range(x.shape[0])]
        ).T.to(device)

        batch_size = x.shape[0]
        batch_idx = torch.arange(batch_size, device=device)

        for j in range(total_element):
            # Remove feature perm[j]
            cur_idx = permutation[j]
            S[batch_idx, cur_idx] = 0

            y_pred = imputer(x, S, mixup_indices, alpha)
            # next_v = alpha * criterion(y_pred, y) + (1 - alpha) * criterion(y_pred, y[mixup_indices])
            next_v1 = criterion(y_pred, y)
            next_v2 = criterion(y_pred, y[mixup_indices])

            diff_v = alpha * (next_v1 - prev_v1)
            diff_v[mixup_indices] += (1 - alpha) * (next_v2 - prev_v2)

            max_diff = max(diff_v.max().item(), max_diff)

            cur_field_idx = cur_idx // n_cols
            cur_cols = cur_idx % n_cols

            # permutation: weight * 16
            # cur_idx: batch x 16
            cur_field = x[torch.arange(x.shape[0]), cur_field_idx]
            cur_idx = cur_field * n_cols + cur_cols

            sum_val.index_add_(dim=0, index=cur_idx, source=diff_v)
            sum_sq.index_add_(dim=0, index=cur_idx, source=diff_v * diff_v)

            prev_v1 = next_v1
            prev_v2 = next_v2

        count += x.shape[0]
        if threshold is not None:
            values = sum_val / count
            sq = sum_sq / count
            std_arr = ((sq - values**2) / count).sqrt()
            std = std_arr.max().item()

            gap: float = max(values.max().item() - values.min().item(), 1e-12)
            ratio = std / gap

            if ratio < threshold and num_epochs >= min_epochs:
                return values.cpu(), std_arr.cpu()
            bar.set_description(f"std={std:.2f} - ratio={ratio:.2f}")

    values = sum_val / count
    sq = sum_sq / count

    std_arr = ((sq - values**2) / count).sqrt()

    print("max diff", max_diff)
    mean_diff = sum_val.sum().item() / count / total_element
    print("mean diff", mean_diff)

    std = sq.sum().item() / count / total_element - mean_diff**2
    print("std diff", std)
    print("total", count * total_element)

    return values.cpu(), std_arr.cpu()
