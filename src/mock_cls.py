"""Class used for logic mocking"""

import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .base import CTRDataset, CTRModel


class MockDataset(CTRDataset):
    """Dataset to generate random data for CTR task
    Used for mocking input and output flow
    """

    def __init__(
        self,
        field_dims: list[int] | int,
        num_items: int = 10,
        *,
        include_offsets: bool = False,
        seed: int = 2023,
        distribution="uniform",
        label_distribution="equal",
    ):
        if isinstance(field_dims, int):
            field_dims = [field_dims]

        assert distribution in ["uniform", "long-tail"]

        self.field_dims = field_dims
        self.num_items = sum(field_dims)
        self._include_offsets = include_offsets

        rng = random.Random(seed)
        seed = seed

        data = []

        for _ in range(num_items):
            result = []
            offsets = 0

            for field in self.field_dims:
                if distribution == "uniform":
                    item = rng.randrange(0, field)
                elif distribution == "long-tail":
                    item = rng.choices(
                        range(field),
                        weights=range(field, 0, -1),
                        k=1,
                    )[0]

                if self._include_offsets:
                    item += offsets
                    offsets += field

                result.append(item)

            if label_distribution == "equal":
                label = sum(result) % 2
            else:
                label = (sum(result) % 4) == 0

            data.append(
                (torch.tensor(result), torch.tensor(label, dtype=torch.float32))
            )

        self.data = data
        self._num_items = num_items

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self._num_items

    def pop_info(self):
        return

    def describe(self):
        desc = (
            "MockDataset("
            f"field_dims={self.field_dims}"
            f"include_offsets={self._include_offsets}"
            f"num_item={self._num_items}"
            ")"
        )

        print(desc)

    def get_frequency(self):
        freq = torch.zeros(sum(self.field_dims), dtype=torch.long)
        for x, y in self.data:
            freq[x] += 1
        return freq


class MockModel(CTRModel):
    def __init__(
        self,
        field_dims: list[int] | int,
        hidden_size: int = 4,
        include_offsets: bool = True,
    ):
        super().__init__()

        if isinstance(field_dims, int):
            field_dims = [field_dims]

        self._field_dims: torch.Tensor = torch.tensor(field_dims)
        self.embedding = nn.Embedding(sum(field_dims), hidden_size)

        field_dims_tensor = torch.tensor(field_dims)
        field_dims_tensor = torch.cat(
            [torch.tensor([0], dtype=torch.long), field_dims_tensor]
        )
        offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
        self.register_buffer("offsets", offsets)

        self.include_offsets = include_offsets

    def forward(self, x):
        """
        Args:
            x: torch.FloatTensor - Shape: Batch x #Fields

        Returns:
            y: Logit, Shape: Batch x 1
        """
        emb = self.get_emb(x)
        return self.head(emb, x)

    def head(self, emb, x):
        return emb.sum((1, 2))

    def get_emb(self, x=None):
        if x is None:
            return self.embedding.weight

        if self.include_offsets:
            x += self.offsets
        return self.embedding(x)

    def get_emb_size(self):
        return self.embedding.weight.shape

    @property
    def field_dims(self):
        return self._field_dims

    def remove_feat(self, feat_to_remove):
        self._backup = self.embedding.weight.data.clone()
        self.embedding.weight.data[feat_to_remove] = 0

    def recover(self):
        self.embedding.weight.data = self._backup


if __name__ == "__main__":
    field_dims = [1, 2, 3]
    dataset = MockDataset(field_dims)
    model = MockModel(field_dims)

    loader = DataLoader(dataset, 4)

    loss = 0.0
    for x, y in loader:
        y_pred = model(x)
        loss += F.binary_cross_entropy_with_logits(y_pred, y).item()

    print(loss)
