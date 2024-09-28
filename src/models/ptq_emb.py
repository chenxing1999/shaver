import torch
from torch import nn
from torch.nn import functional as F


class PTQEmb_Fp16(nn.Module):
    """Post-training Quantization Embedding Float16"""

    def __init__(self, field_dims, num_factor, mode, ori_checkpoint_path):
        """
        Args:
            field_dims, num_factor: Could be anything, only there for API
                format
        """
        super().__init__()

        checkpoint = torch.load(ori_checkpoint_path)
        state = checkpoint["state_dict"]
        emb = state["embedding._emb_module.weight"]
        # Convert emb to
        self.register_buffer("weight", emb.to(torch.float16))

    def forward(self, x):
        return F.embedding(x, self.weight).to(torch.float32)

    def get_weight(self) -> torch.Tensor:
        return self.weight

    def get_num_params(self) -> int:
        return self.weight.shape[0] * self.weight.shape[1]


class PTQEmb_Int(nn.Module):
    """Post-training Quantization Embedding Int8"""

    def __init__(
        self,
        field_dims,
        num_factor,
        mode,
        # ori_checkpoint_path,
        weight,
        n_bits=8,
    ):
        """
        Args:
            field_dims, num_factor: Could be anything, only there for API
                format

            ori_checkpoint_path: Path to original checkpoint to
                quantize
            n_bits: Num bits, 8 or 16
        """
        super().__init__()

        # checkpoint = torch.load(ori_checkpoint_path, map_location="cpu")
        # state = checkpoint["state_dict"]
        # emb: torch.Tensor = state["embedding._emb_module.weight"]
        emb: torch.Tensor = weight

        # calculate scale
        assert n_bits in [8, 16]
        self.n_bits = n_bits

        q_min = (-1) * (1 << (self.n_bits - 1))
        q_max = (1 << (self.n_bits - 1)) - 1
        r_min = emb.min().cpu()
        scale = (emb.max().item() - r_min) / (q_max - q_min)

        dtype = torch.int8
        if self.n_bits == 16:
            dtype = torch.int16

        bias = (q_min - r_min / scale).to(dtype)

        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

        weight = torch.round(emb / scale + bias)
        self.register_buffer("weight", weight.to(dtype))

    def forward(self, x):
        res = F.embedding(x, self.weight)

        return (res - self.bias) * self.scale

    def get_weight(self) -> torch.Tensor:
        return (self.weight - self.bias) * self.scale

    def get_num_params(self) -> int:
        return self.weight.shape[0] * self.weight.shape[1]

    @property
    def num_embeddings(self):
        return self.weight.shape[0]

    @property
    def embedding_dim(self):
        return self.weight.shape[1]
