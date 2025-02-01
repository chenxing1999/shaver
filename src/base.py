from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset


class CTRModel(nn.Module):
    @abstractmethod
    def get_emb(self, x: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        ...

    @abstractmethod
    def head(self, emb: torch.FloatTensor, x: torch.LongTensor) -> torch.FloatTensor:
        ...

    @abstractmethod
    def get_emb_size(self) -> tuple[int, int]:
        ...

    def forward(self, x=None) -> torch.FloatTensor:
        emb = self.get_emb(x)
        return self.head(emb, x)

    def remove_feat(self, feat_to_remove) -> None:
        ...

    def recover(self) -> None:
        ...

    @property
    def field_dims(self) -> torch.Tensor:
        ...


class CTRDataset(Dataset[Tuple[torch.Tensor, float]], ABC):
    field_dims: Iterable[int]

    @abstractmethod
    def pop_info(self):
        ...

    @abstractmethod
    def describe(self):
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """get item function of CTR Dataset

        Returns:
            feature: ADDED offsets
            label: 1 clicked, 0 not clicked
        """
        ...
