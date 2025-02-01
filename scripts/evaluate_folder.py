import argparse
import os
from collections import defaultdict
from typing import Dict, Final, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.const import get_parser, parse_args_const
from src.datasets import get_dataset
from src.models.codebook_emb import CodebookEmb
from src.models.dcn import DCN_Mix, DCNv2
from src.models.deepfm import DeepFM
from src.utils import get_feat_to_field, get_freq_avg, validate_epoch


def get_model(checkpoint: str, model_name: str):
    if model_name == "dcn":
        model = DCN_Mix.load(checkpoint)
        model = model._orig_mod
    elif model_name == "deepfm":
        model = DeepFM.load(checkpoint)
    elif model_name == "dcnv2":
        model = DCNv2.load(checkpoint)
    return model


def evaluate(
    loader,
    model_name: str,
    checkpoint_path: str,
    shapley_value_path: str,
    ratios: Final[List[float]],
    base_value: Optional[torch.Tensor] = None,
) -> Dict[float, Dict[str, float]]:
    """ """
    model = get_model(checkpoint_path, model_name)
    shapley_value = torch.load(shapley_value_path).flatten().abs()

    weight = model.embedding.weight.data
    n_rows, n_cols = weight.shape
    results = {}
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

        emb = CodebookEmb(mask, weight, base_value, n_bits=32)
        model.embedding = emb

        val_result = validate_epoch(loader, model)
        print(val_result)
        results[ratio] = val_result
    return results


def parse_args(argv: Optional = None) -> argparse.Namespace:
    parser = get_parser()

    parser.add_argument("shapley_folder", type=str)
    parser.add_argument("output_path", type=str)

    # ratios = [0.5, 0.8, 0.95, 0.99, 0.999, 0.9999]
    ratios = [0.5, 0.8, 0.95, 0.99, 0.999, 0.9999]
    full_ratios = ratios.copy()
    full_ratios.extend([0.2, 0.4, 0.6, 0.9])
    full_ratios.sort()

    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="list of sparsity rates",
        default=full_ratios,
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="instead of write, add value",
    )

    args = parser.parse_args(argv)
    args = parse_args_const(args)
    return args


args = parse_args()

save_as_csv = args.output_path.endswith(".csv")

model = get_model(args.checkpoint_path, args.model_name)
w = model.embedding.weight.data
device = "cuda"
freq = torch.load(args.freq_path).to(device)

base_value = 0
base_value = get_freq_avg(w.to(device), freq, model.field_dims)
base_value = base_value.to(device)
feat_to_fields = get_feat_to_field(model.field_dims).to(device)
del model

dataset = get_dataset(args.dataset_name, "test")
loader = DataLoader(dataset, 2048, num_workers=4)

output = {}
for shapley_value_name in os.listdir(args.shapley_folder):
    shapley_value_path = os.path.join(args.shapley_folder, shapley_value_name)
    print(shapley_value_path)

    if "codebook" not in shapley_value_name:
        b = None
    else:
        b = base_value
    output[shapley_value_name] = evaluate(
        loader,
        args.model_name,
        args.checkpoint_path,
        shapley_value_path,
        args.ratios,
        b,
    )

if not save_as_csv:
    # Save as binary file
    print(f"Save as binary file {args.output_path}")
    if args.add and os.path.exists(args.output_path):
        o = torch.load(args.output_path)
        output.update(o)
    torch.save(output, args.output_path)
else:
    print(f"Save as csv file {args.output_path}")
    # Format to data frame
    rate_to_record = defaultdict(dict)
    for name, info in output.items():
        for sparse_rate, result in info.items():
            row = rate_to_record[sparse_rate]
            row["sparse-rate"] = sparse_rate
            row[name] = result["auc"]

    df = pd.DataFrame.from_records(list(rate_to_record.values()))
    if args.add and os.path.exists(args.output_path):
        old_df = pd.read_csv(args.output_path, index_col="sparse-rate")
        cols_union = old_df.columns.union(df.columns)
        if len(cols_union):
            print(
                f"{cols_union} already exists... Overwrite old result with new result"
            )
            cols_to_use = old_df.columns.difference(df.columns)
            old_df = old_df[cols_to_use]

        df = df.set_index("sparse-rate").join(old_df, how="outer").reset_index()

    cols = df.columns
    cols: list = cols.tolist()
    cols.sort()
    ind = cols.index("sparse-rate")
    cols[0], cols[ind] = cols[ind], cols[0]
    df.to_csv(
        args.output_path,
        sep=",",
        na_rep="",
        header=True,
        index=False,
        columns=cols,
    )
