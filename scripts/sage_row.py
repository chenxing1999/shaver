import argparse
import os
from pprint import pprint
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.const import get_parser, parse_args_const
from src.datasets import get_dataset, get_train_val_merge
from src.models.dcn import DCN_Mix, DCNv2
from src.models.deepfm import DeepFM
from src.models.ptq_emb import PTQEmb_Int
from src.sage_row import RowDefaultImputer, sage_shapley_field_ver
from src.utils import get_freq_avg, set_seed, validate_epoch


set_seed(2023)


def str2bool(inp: str) -> bool:
    return inp.lower() in ["1", "true"]


# input
def parse_args(argv: Optional = None) -> argparse.Namespace:
    parser = get_parser()
    parser.add_argument(
        "--p_train",
        default=0,
        type=float,
        help="percent of training to be used in train",
    )
    parser.add_argument("--p_val", type=float, help="Part of val dataset", default=1.0)
    parser.add_argument("--codebook", action="store_true", help="Enable codebook mode")
    parser.add_argument("--codebook_path", default="", help="Using new codebook format")

    parser.add_argument("--debug", action="store_true", help="Run only 100 iterations")
    parser.add_argument("--output_path", default="cols_shapley_field.pth")
    parser.add_argument(
        "--n_bits", choices=[8, 16, 32], default=32, help="Run quantize mode or not"
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        help="Num workers",
        type=int,
    )
    args = parser.parse_args(argv)
    args = parse_args_const(args)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    p_train = args.p_train  # amount of train to include to val
    assert args.model_name in ["deepfm", "dcn", "dcnv2"]
    assert args.dataset_name in ["criteo", "avazu", "kdd"]

    device = "cuda"

    val_dataset = get_train_val_merge(
        args.dataset_name,
        p_train,
        p_val=args.p_val,
    )

    checkpoint = args.checkpoint_path
    freq = torch.load(args.freq_path).to(device)

    batch_size = 8192

    num_workers = args.num_workers
    metric = "log_loss"

    if args.model_name == "deepfm":
        model = DeepFM.load(checkpoint)
    elif args.model_name == "dcn":
        model = DCN_Mix.load(checkpoint)
    elif args.model_name == "dcnv2":
        model = DCNv2.load(checkpoint)

    if args.n_bits in [8, 16]:
        model.embedding = PTQEmb_Int(
            model.field_dims,
            16,
            None,
            model.embedding.weight,
            n_bits=args.n_bits,
        )

    model.to(device)
    model.eval()

    for idx, (x, y) in enumerate(val_dataset):
        if y == 1:
            x = torch.tensor([x])
            y = torch.tensor([y])
            break

    val_loader = DataLoader(
        val_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    # Create Imputer

    base_value = get_freq_avg(
        model.embedding.weight.data,
        freq,
        model.field_dims,
    )
    if args.codebook:
        if args.codebook_path and os.path.exists(args.codebook_path):
            base_value = torch.load(args.codebook_path)
        imputer = RowDefaultImputer(model, True, base_value)
    else:
        imputer = RowDefaultImputer(model, True, 0)

    # Test code
    n_fields = len(model.field_dims)
    S = torch.ones(n_fields * 16, device="cuda", dtype=bool)
    x = x.to(device)
    y = y.to(device)

    print("ori_pred", imputer(x, S))

    # Run algorithm
    criterion = None

    if args.debug:
        value, std = sage_shapley_field_ver(
            model,
            val_loader,
            100,
            min_epochs=0,
            criterion=criterion,
            imputer=imputer,
        )
    else:
        value, std = sage_shapley_field_ver(
            model,
            val_loader,
            int(1e20),
            threshold=1e-2,
            min_epochs=1,
            criterion=criterion,
            imputer=imputer,
        )
    torch.save(value, args.output_path)

    # testing
    ori_weight = model.embedding.weight.data.clone()
    print("no correction")
    num_rows, num_cols = model.get_emb_size()
    total_ele = num_rows * num_cols
    n_remove = int(total_ele * 0.8)

    # s_value = value.view(*ori_weight.shape)
    s_value = value.abs()
    idx = torch.argsort(s_value)

    idx1 = idx // num_cols
    idx2 = idx % num_cols

    print(f"Remove {n_remove} smallest shapley value")
    model.embedding.weight.data[idx1[:n_remove], idx2[:n_remove]] = 0
    print("new_pred", imputer(x, S))

    print("Test result")
    test_dataset = get_dataset(args.dataset_name, "test")
    val_loader = DataLoader(test_dataset, 8192, shuffle=False, num_workers=8)
    print(validate_epoch(val_loader, model))
    model.embedding.weight.data = ori_weight.clone()
    print()

    peak_cuda_mem = torch.cuda.max_memory_allocated()
    cur_cuda_mem = torch.cuda.memory_allocated()

    print(
        {
            "cur_cuda_mem": cur_cuda_mem,
            "peak_cuda_mem": peak_cuda_mem,
        }
    )
