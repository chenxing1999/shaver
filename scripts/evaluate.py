import argparse
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from src.const import get_parser, parse_args_const
from src.datasets import get_dataset
from src.models.codebook_emb import CodebookEmb
from src.models.dcn import DCN_Mix, DCNv2
from src.models.deepfm import DeepFM
from src.utils import validate_epoch


# input
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = get_parser()

    # ratios = [0.5, 0.8, 0.95, 0.99, 0.999, 0.9999]
    ratios = [0.5, 0.8, 0.95]
    full_ratios = ratios.copy()
    full_ratios.extend([0.2, 0.4, 0.6, 0.9])
    full_ratios.sort()

    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="list of sparsity rates",
        default=ratios,
    )
    parser.add_argument(
        "--full-ratio",
        action="store_true",
        help="Run big list of predefined ratio",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split used for evaluation",
        choices=["val", "test", "train"],
    )

    args = parser.parse_args(argv)
    args = parse_args_const(args)
    if args.full_ratio:
        args.ratios = full_ratios
    return args


args = parse_args()
# input parameters
DATASET_NAME = args.dataset_name
MODEL_NAME = args.model_name
device = "cuda"
# shapley_path = args.shapley_value_path
# shapley_path = "./cols_shapley_field.pth"
# shapley_path = "./artifacts/criteo_zerobaseline/cols_shapley_field.pth"
# -------------------

dataset = get_dataset(args.dataset_name, args.split)

checkpoint = args.checkpoint_path
freq = torch.load(args.freq_path).to(device)


# checkpoint = "./random_init.pth"
if MODEL_NAME == "dcn":
    model = DCN_Mix.load(checkpoint)
    model = model._orig_mod
elif MODEL_NAME == "deepfm":
    model = DeepFM.load(checkpoint)
elif MODEL_NAME == "dcnv2":
    model = DCNv2.load(checkpoint)
w = model.embedding.weight.data.clone()

# shapley_path = "./cols_shapley_field.pth"
# shapley_path = "./artifacts/avazu_res_v5/cols_shapley_field.pth"
# shapley_value = torch.load(shapley_path)
# shapley_value = shapley_value.abs()
shapley_value = model.embedding.weight.data.abs().flatten()


# shapley_path = "./taylor.pth"
# shapley_value = torch.load(shapley_path)
# shapley_value = shapley_value.flatten()
shapley_value = shapley_value.flatten().abs()
# print(shapley_value.shape)

# shapley_path = "./sseds.pth"
# shapley_value = torch.load(shapley_path).flatten()


class SubsetSampler2(Sampler):
    def __init__(self, num_data):
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def __iter__(self):
        yield from range(self.num_data)


n_data = int(4e3)
loader = DataLoader(dataset, 2048, num_workers=4)
# feat_to_keep = torch.argsort(shapley_value, descending=False)

n_rows, n_cols = w.shape
ratios = args.ratios
# ratios = [0.2, 0.4, 0.6, 0.8, 0.9]

# ratios.extend([0.2, 0.4, 0.6, 0.9])
# ratios.sort()

weight = model.embedding.weight.data

for ratio in ratios:
    print(ratio)
    num_ele = int(n_rows * n_cols * ratio)

    # shapley_value = shapley_value.view(w.shape[0], w.shape[1])
    idx = torch.argsort(shapley_value)
    idx = idx[:num_ele]
    idx1 = idx // n_cols
    idx2 = idx % n_cols

    weight[idx1, idx2] = 0

    print(len(idx1))
    mask = torch.zeros_like(weight, dtype=bool)
    mask[idx1, idx2] = 1

    emb = CodebookEmb(mask, weight, None, n_bits=32)
    model.embedding = emb

    print(validate_epoch(loader, model))

    # print(validate_feat_keep(loader, model, feat_to_keep[:num_ele]))
    # print(validate_feat_keep(loader, model, feat_to_keep))
    # print(validate_feat_keep(loader, model, feat_to_keep[num_ele:]))
