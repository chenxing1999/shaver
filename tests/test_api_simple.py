from src.sage_row import RowDefaultImputer, sage_shapley_field_ver
from src.mock_cls import MockModel, MockDataset
from src.utils import set_seed
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch

set_seed(2023)

def test_algo1_api():

    field_dims = [1,2,3]
    num_data = 10

    # Model should assume offsets included from Dataset
    model = MockModel(field_dims, hidden_size=4, include_offsets=False)
    dataset = MockDataset(
        field_dims,
        num_data,
        include_offsets=True,
    )

    loader = DataLoader(dataset, batch_size=4)


    n_iters = 1000
    device = "cpu"

    imputer = RowDefaultImputer(
        model,
        use_sigmoid=True, # Return sigmoid output,
        base_value=0, # Codebook value
    )
    value, std = sage_shapley_field_ver(
        model,
        loader,
        n_iters,
        imputer,
        device,
        # threshold=1e-2, # converge threshold
        # min_epochs=1, # ensure loop through the whole dataset
    )

    assert len(value.shape) == 1
    assert value.shape[0] == sum(field_dims) * 4

    model.eval()
    loss1 = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)

        loss1 += F.binary_cross_entropy_with_logits(y_pred, y, reduction="sum")

    loss2 = 0
    model.embedding.weight.data[:] = 0 # set S to empty
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)

        loss2 += F.binary_cross_entropy_with_logits(y_pred, y, reduction="sum")

    loss = loss2 - loss1
    loss = loss / num_data

    # ensure efficiency
    assert (value.sum() - loss).abs() < 1e-3, f'{value.sum()} != {loss / num_data}'
