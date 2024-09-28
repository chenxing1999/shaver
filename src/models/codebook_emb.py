from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import get_feat_to_field, get_freq_avg, post_training_quantization


class CodebookEmb(nn.Module):
    def __init__(
        self,
        codebook_mask,
        weight,
        codebook: Optional[torch.FloatTensor] = None,
        n_bits: int = 32,
    ):
        """
        Args:
            codebook_mask: torch.Bool
                0 for not pruned
                1 for pruned

            weight: torch.FloatTensor (NumFeat x HiddenSize)
                original model weight

            codebook: torch.FloatTensor (NumField x HiddenSize)
                if None --> Set all to zero and disable gradient

            n_bits: 8, 16, 32
                If n_bits in [8, 16] --> Use int with corresponding bits
        """
        super().__init__()
        assert n_bits in [4, 8, 16, 32], f"{n_bits=}. Not supported"

        self.codebook = None
        if codebook is not None:
            self.codebook = nn.Parameter(codebook)
        self.codebook_mask = nn.Parameter(codebook_mask, requires_grad=False)

        self.n_bits = n_bits
        if self.n_bits in [4, 8, 16]:
            weight = post_training_quantization(weight, self.n_bits)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        """
        Args:
            x: torch.LongTensor (Batch x NumField)

        Returns:
            emb: torch.FloatTensor (Batch x NumField x HiddenSize)
        """

        mask = F.embedding(x, self.codebook_mask)
        w1 = F.embedding(x, self.weight)

        if self.codebook is None:
            return torch.logical_not(mask) * w1

        emb = torch.logical_not(mask) * w1 + mask * self.codebook
        return emb


class ReparametrizationSTE(torch.autograd.Function):
    """Perform STE to Emb, no update to Codebook"""

    @staticmethod
    def forward(ctx, emb, codebook, mask):
        """
        Args:
            emb (torch.FloateTensor - Batch x N_Field x Hidden Size)
            codebook (torch.FloateTensor - N_Field x Hidden Size)
            mask (torch.BoolTensor - Batch x N_Field x Hidden Size)
        Returns:
            emb (torch.FloateTensor - Batch x N_Field x Hidden Size)
        """
        emb_result = torch.logical_not(mask) * emb + mask * codebook

        # ctx.save_for_backward(q_w, q_w_float, n_bits)
        return emb_result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: 2 or 3D array
        Returns:
            grad_output (STE) directly to weight
            codebook -- No update
            mask -- No update
        """
        # (res, q_w, n_bits) = ctx.saved_tensors

        return grad_output, torch.tensor(0), torch.tensor(0)


class CodebookEmb2(nn.Module):
    def __init__(
        self,
        codebook_mask,
        weight,
        field_dims,
        codebook: Optional[torch.FloatTensor] = None,
        n_bits: int = 32,
        ema: float = 0.0,
    ):
        """Instead of optimizing Codebook by Gradient, use reparameterization trick

        Args:
            codebook_mask: torch.Bool
                0 for not pruned
                1 for pruned

            weight: torch.FloatTensor (NumFeat x HiddenSize)
                original model weight

            codebook: torch.FloatTensor (NumField x HiddenSize)
                if None --> Set all to zero and disable gradient

            n_bits: 8, 16, 32
                If n_bits in [8, 16] --> Use int with corresponding bits
        """
        super().__init__()
        assert n_bits in [8, 16, 32], f"{n_bits=}. Not supported"

        self.codebook = None
        if codebook is not None:
            self.codebook = nn.Parameter(codebook, requires_grad=False)
        self.codebook_mask = nn.Parameter(codebook_mask, requires_grad=False)

        self.n_bits = n_bits
        if self.n_bits in [8, 16]:
            weight = post_training_quantization(weight, self.n_bits)
        self.weight = nn.Parameter(weight)
        self.field_dims = field_dims
        self.register_buffer("feat_to_fields", get_feat_to_field(field_dims))

        self.n_cols = weight.shape[1]
        self.ema = ema

    def forward(self, x):
        """
        Args:
            x: torch.LongTensor (Batch x NumField)

        Returns:
            emb: torch.FloatTensor (Batch x NumField x HiddenSize)
        """
        if self.training:
            w1 = F.embedding(x, self.weight)
            mask = F.embedding(x, self.codebook_mask)
            emb = ReparametrizationSTE.apply(w1, self.codebook, mask)
            return emb
            # return F.embedding(x, self.weight)
        else:
            mask = F.embedding(x, self.codebook_mask)
            w1 = F.embedding(x, self.weight)

            if self.codebook is None:
                return torch.logical_not(mask) * w1

            # emb = torch.logical_not(mask) * w1 + mask * self.codebook
            emb = ReparametrizationSTE.apply(w1, self.codebook, mask)
            return emb

    def reparametrization(self, freq):
        weight = self.weight.data
        base_value = get_freq_avg(weight, freq.to(weight.device), self.field_dims)

        if not self.ema:
            self.codebook.data = base_value
        else:
            self.codebook.data = self.codebook.data * self.ema + base_value * (
                1 - self.ema
            )

        # mask = self.codebook_mask
        # idx1, idx2 = torch.where(mask)
        # idx_in_base = self.feat_to_fields[idx1] * self.n_cols + idx2
        # self.weight.data[idx1, idx2] = base_value.flatten()[idx_in_base]


class CodebookEmb_Opt(nn.Module):
    def __init__(
        self,
        codebook_mask,
        weight,
        field_dims,
        codebook: Optional[torch.FloatTensor] = None,
        n_bits: int = 32,
        ema: float = 0.0,
    ):
        """Optimizing Codebook

        Args:
            codebook_mask: torch.Bool
                0 for not pruned
                1 for pruned

            weight: torch.FloatTensor (NumFeat x HiddenSize)
                original model weight

            codebook: torch.FloatTensor (NumField x HiddenSize)
                if None --> Set all to zero and disable gradient

            n_bits: 8, 16, 32
                If n_bits in [8, 16] --> Use int with corresponding bits
        """
        super().__init__()
        assert n_bits in [8, 16, 32], f"{n_bits=}. Not supported"

        self.codebook = None
        if codebook is not None:
            self.codebook = nn.Parameter(codebook)

        self.n_bits = n_bits
        if self.n_bits in [8, 16]:
            weight = post_training_quantization(weight, self.n_bits)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.field_dims = field_dims
        self.register_buffer("feat_to_fields", get_feat_to_field(field_dims))

        self.n_cols = weight.shape[1]
        self.ema = ema

    def forward(self, x):
        """
        Args:
            x: torch.LongTensor (Batch x NumField)

        Returns:
            emb: torch.FloatTensor (Batch x NumField x HiddenSize)
        """
        if self.training:
            w1 = F.embedding(x, self.weight)
            mask = torch.randint_like(w1, 0, 1, dtype=bool)
            emb = torch.logical_not(mask) * w1 + mask * self.codebook
            return emb
            # return F.embedding(x, self.weight)
        else:
            w1 = F.embedding(x, self.weight)
            mask = torch.zeros_like(w1, dtype=bool)

            if self.codebook is None:
                return torch.logical_not(mask) * w1

            # emb = torch.logical_not(mask) * w1 + mask * self.codebook
            emb = ReparametrizationSTE.apply(w1, self.codebook, mask)
            return emb

    def reparametrization(self, freq):
        return
        weight = self.weight.data
        base_value = get_freq_avg(weight, freq.to(weight.device), self.field_dims)

        if not self.ema:
            self.codebook.data = base_value
        else:
            self.codebook.data = self.codebook.data * self.ema + base_value * (
                1 - self.ema
            )

        # mask = self.codebook_mask
        # idx1, idx2 = torch.where(mask)
        # idx_in_base = self.feat_to_fields[idx1] * self.n_cols + idx2
        # self.weight.data[idx1, idx2] = base_value.flatten()[idx_in_base]


if __name__ == "__main__":
    field_dims = [2, 2, 3]
    n_fields = len(field_dims)
    n_feats = sum(field_dims)

    hidden_size = 4

    w = torch.rand(n_feats, hidden_size)

    # 7, 4
    codebook_mask = torch.tensor(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )

    # 3, 4
    codebook = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
        ],
        dtype=float,
    )

    x = torch.tensor([[1, 3, 5]])
    emb = CodebookEmb(codebook_mask, w, codebook)

    res = emb(x)
    res.mean().backward()

    # print(emb.codebook.grad)
    # print(emb.weight.grad)
