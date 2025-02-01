from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn import functional as F

from .base import CTRModel


class Imputer(ABC):
    @abstractmethod
    def __call__(self, x, S):
        """Get prediction based on input x and masked feature S

        Args:
            x: torch.LongTensor, shape: Batch x Num Field
            S: mask for which feature to keep
                shape: Batch x Num Fields
                S = 1 --> Prune

        Returns:
            y: torch.FloatTensor - Same with original model output
                shape: Batch for CTRModel
        """
        ...

    @property
    @abstractmethod
    def total_element(self) -> int:
        ...


class DefaultImputer(Imputer):
    """Convert masked feature to base value"""

    def __init__(
        self,
        model: CTRModel,
        base_value=0,
        use_sigmoid=True,
        level: Optional[str] = None,
    ):
        super().__init__()

        if level is None:
            level = "feat"
        self.level = level

        self.model = model
        self.base_value = base_value
        self.use_sigmoid = use_sigmoid
        self._loss = 0

    @property
    def total_element(self):
        if self.level == "feat":
            return sum(self.model.field_dims)
        else:
            raise NotImplementedError()

    def __call__(self, x, S):
        if self.level == "weight":
            return self.weight_call(x, S)

        with torch.no_grad():
            emb = self.model.get_emb(x)
            if isinstance(self.base_value, (float, int)):
                emb[~S] = self.base_value
            else:
                batch_size = emb.shape[0]
                base_value = self.base_value.repeat(batch_size, 1, 1)

                emb[~S] = base_value[~S]

            y_pred = self.model.head(emb, x)

        if self.use_sigmoid:
            y_pred = torch.sigmoid_(y_pred)
        return y_pred

    def weight_call(self, x, neuron_to_keeps):
        """
        Args:
        """
        with torch.no_grad():
            new_emb_table = self.model.get_emb().clone()

        num_cols = new_emb_table.shape[1]
        r, c = neuron_to_keeps // num_cols, neuron_to_keeps % num_cols

        mask = torch.ones_like(new_emb_table, dtype=torch.bool)
        mask[r, c] = 0
        new_emb_table[mask] = 0

        emb = F.embedding(x, new_emb_table)
        with torch.no_grad():
            y_pred = self.model.head(emb, x)

        if self.use_sigmoid:
            y_pred = torch.sigmoid_(y_pred)
        return y_pred


class DefaultOOVImputer(Imputer):
    """Convert masked feature to its corresponding OOV value"""

    def __init__(
        self,
        model: CTRModel,
        use_sigmoid=True,
        level: Optional[str] = None,
    ):
        super().__init__()

        self.model = model
        self.level = level

        self.use_sigmoid = use_sigmoid
        self.oov_idx = torch.cumsum(model.field_dims, 0) - 1
        print("oov idx", self.oov_idx)

    def __call__(self, x, S):
        if self.level == "weight":
            raise NotImplementedError()

        oov_idx = self.oov_idx.to(x.device)
        with torch.no_grad():
            # Convert x to its OOV value
            oov_idx = oov_idx.repeat(x.shape[0]).view(*x.shape)
            x_clone = x.clone()
            x_clone[~S] = oov_idx[~S]

            emb_oov = self.model.get_emb(x_clone)
            y_pred = self.model.head(emb_oov, x)

        if self.use_sigmoid:
            y_pred = torch.sigmoid_(y_pred)
        return y_pred

    def weight_call(self, x, neuron_to_keeps):
        """
        Args:
        """
        with torch.no_grad():
            new_emb_table = self.model.get_emb().clone()

        num_cols = new_emb_table.shape[1]
        r, c = neuron_to_keeps // num_cols, neuron_to_keeps % num_cols

        mask = torch.ones_like(new_emb_table, dtype=torch.bool)
        mask[r, c] = 0
        new_emb_table[mask] = 0

        emb = F.embedding(x, new_emb_table)
        with torch.no_grad():
            y_pred = self.model.head(emb, x)

        if self.use_sigmoid:
            y_pred = torch.sigmoid_(y_pred)
        return y_pred


class MarginalImputer(Imputer):
    def __init__(
        self,
        model,
        samples,
        use_sigmoid=True,
        sample_size=2048,
        dataset=None,
    ):
        self.model = model
        self.samples = samples
        self.use_sigmoid = use_sigmoid
        self.sample_size = sample_size
        self.num_samples = len(samples)
        self.dataset = dataset

    def __call__(self, x, S):
        """Get prediction based on input x and masked feature S

        Args:
            x: torch.LongTensor, shape: Batch x Num Field
            S: mask for which feature to keep
                shape: Batch x Num Fields

        Returns:
            y: torch.FloatTensor - Same with original model output
                shape: Batch for CTRModel
        """
        batch_size = x.shape[0]
        tmp = torch.ones(self.num_samples).unsqueeze(0).repeat(batch_size, 1)
        indices = torch.multinomial(tmp, self.sample_size)
        samples = self.samples[indices]

        # samples = random.choices(
        #     self.dataset,
        #     k=self.sample_size,
        # )
        # samples = torch.stack([torch.from_numpy(s[0]) for s in samples])
        # samples = samples.unsqueeze(0).repeat(batch_size, 1, 1)

        samples = samples.to(x.device)

        # x: batch_size, sample_size, num_fields
        new_x = x.unsqueeze(1).repeat(1, self.sample_size, 1)
        S = S.unsqueeze(1).repeat(1, self.sample_size, 1)
        new_x[~S] = samples[~S]
        with torch.no_grad():
            emb = self.model.get_emb(new_x)

            # batch_size, sample_size, num_fields, hidden_size
            emb = emb.view(batch_size * self.sample_size, emb.shape[2], -1)
            y_pred = self.model.head(emb, x.repeat(self.sample_size, 1))

        if self.use_sigmoid:
            y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.view(x.shape[0], self.sample_size).mean(1)
        return y_pred
