import argparse
import copy
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from src.const import get_parser, parse_args_const
from src.datasets import get_dataset
from src.models.codebook_emb import CodebookEmb
from src.models.dcn import DCN_Mix, DCNv2
from src.models.deepfm import DeepFM
from src.utils import get_feat_to_field, get_freq_avg, validate_epoch


# input
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = get_parser()

    ratios = [0.5, 0.8, 0.95, 0.99, 0.999, 0.9999]
    full_ratios = ratios.copy()
    full_ratios.extend([0.2, 0.4, 0.6, 0.9])

    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="list of sparsity rates",
        default=ratios,
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=32,
        help="number of bits used per param",
        choices=[8, 16, 32],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split used for evaluation",
        choices=["val", "test", "train"],
    )

    parser.add_argument(
        "--kldiv",
        action="store_true",
        help="Enable KLDiv Loss to compare original model and pruned model",
    )
    parser.add_argument(
        "--full-ratio",
        action="store_true",
        help="Run big list of predefined ratio",
    )

    parser.add_argument(
        "--codebook_path",
        default="",
        help="Path to codebook file to replace original codebook",
    )
    args = parser.parse_args(argv)
    args = parse_args_const(args)
    if args.full_ratio:
        args.ratios = full_ratios
    return args


args = parse_args()
# input parameters
device = "cuda"
shapley_path = args.shapley_value_path

dataset = get_dataset(args.dataset_name, args.split)

checkpoint = args.checkpoint_path
freq = torch.load(args.freq_path).to(device)

# checkpoint = "./random_init.pth"
if args.model_name == "dcn":
    model = DCN_Mix.load(checkpoint)
    model = model._orig_mod
elif args.model_name == "deepfm":
    model = DeepFM.load(checkpoint)
elif args.model_name == "dcnv2":
    model = DCNv2.load(checkpoint)

model = model.to(device)

w = model.embedding.weight.data.clone()

# shapley_path = "./artifacts/avazu_res_v5/cols_shapley_field.pth"
# shapley_path = "./cols_shapley_field.pth"
shapley_value = torch.load(shapley_path)
shapley_value = shapley_value.to(device)
shapley_value = shapley_value.abs()
# shapley_value = shapley_value + 1e-5 * model.embedding.weight.data.abs().flatten()

# shapley_path = "./taylor.pth"
# shapley_value = torch.load(shapley_path)
# shapley_value = shapley_value.flatten().abs()

if args.codebook_path:
    base_value = torch.load(args.codebook_path)
else:
    base_value = get_freq_avg(w.to(device), freq, model.field_dims)
    base_value = base_value.to(device)
feat_to_fields = get_feat_to_field(model.field_dims).to(device)


class SubsetSampler2(Sampler):
    def __init__(self, num_data):
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def __iter__(self):
        yield from range(self.num_data)


loader = DataLoader(dataset, 2048, num_workers=4)
# feat_to_keep = torch.argsort(shapley_value, descending=False)

n_rows, n_cols = w.shape
# ratios = [0.5, 0.8, 0.95, 0.99, 0.999, 0.9999]
ratios = args.ratios
# ratios = [0.2, 0.4, 0.6, 0.8, 0.9]

# ratios.extend([0.2, 0.4, 0.6, 0.9])
# ratios.extend([0.0])
ratios.sort()

weight = model.embedding.weight.data

original_model = None
if args.kldiv:
    original_model = copy.deepcopy(model)


for ratio in ratios:
    print(ratio)
    num_ele = int(n_rows * n_cols * ratio)

    # shapley_value = shapley_value.view(w.shape[0], w.shape[1])
    idx = torch.argsort(shapley_value)
    idx = idx[:num_ele]
    idx1 = idx // n_cols
    idx2 = idx % n_cols
    idx_in_base = feat_to_fields[idx1] * n_cols + idx2
    # idx = torch.where(shapley_value > 0)
    # weight[idx1, idx2] = base_value.flatten()[idx_in_base]

    print(len(idx1))
    # model.embedding.weight.data[idx1, idx2] = base_value.flatten()[idx_in_base]
    mask = torch.zeros_like(model.embedding.weight, dtype=bool)
    mask[idx1, idx2] = 1

    emb = CodebookEmb(mask, weight, base_value, n_bits=args.n_bits)
    model.embedding = emb

    print(validate_epoch(loader, model, device, original_model))
