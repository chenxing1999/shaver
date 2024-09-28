from src.mock_cls import MockModel
from src.utils import get_freq_avg
import torch


def test_init_codebook():

    field_dims = [2,4,2]
    hidden_size = 4

    model = MockModel(field_dims, hidden_size)
    freq = torch.tensor([5,4,5,2,1,1,8,1])

    codebook = get_freq_avg(model.embedding.weight, freq, field_dims)
    assert codebook.shape == (len(field_dims), hidden_size)


