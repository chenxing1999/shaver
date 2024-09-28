from src.base import CTRModel
from src.imputers import Imputer
import tqdm 
import torch


class Estimator:
    def __init__(self, model: CTRModel, imputer: Imputer):

        self.device = "cuda"
        model.to(self.device)
        model.eval()

        self.model = model
        self.imputer = imputer


    def __call__(self, loader, n_iters):

        # get input information from self
        model = self.model
        imputer = self.imputer
        device = self.device
        total_element = imputer.total_element

        emb_size = model.get_emb_size()
        weight_size = emb_size[0] * emb_size[1]
        n_cols = emb_size[1]
        sum_val = torch.zeros(weight_size, device=device)
        sum_sq = torch.zeros(weight_size, device=device)

        iter_loader = iter(loader)
        std = ratio = 1
        bar = tqdm.tqdm(
            range(n_iters),
            total=n_iters,
            ascii=True,
            disable=not verbose,
            desc=f"std={std:.2f} - ratio={ratio:.2f}",
        )

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

            for j in range(total_element):
                # Remove feature perm[j]
                S[torch.arange(x.shape[0]), permutation[:, j]] = 0

                y_pred = imputer(x, S)
                next_v = criterion(y_pred, y)

                diff_v = next_v - prev_v

                cur_idx = permutation[:, j].to(device)
                cur_field_idx = cur_idx // n_cols
                cur_cols = cur_idx % n_cols

                # permutation: weight * 16
                # cur_idx: batch x 16
                cur_field = x[torch.arange(x.shape[0]), cur_field_idx]
                cur_idx = cur_field * n_cols + cur_cols

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
