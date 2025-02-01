import argparse
import os
from typing import Literal, Optional, Sequence

import loguru
import torch
from torch.utils.data import DataLoader

from src.const import get_parser, parse_args_const
from src.datasets import get_dataset
from src.loggers import Logger
from src.models.codebook_emb import CodebookEmb
from src.models.dcn import DCN_Mix
from src.models.deepfm import DeepFM
from src.trainer import train_epoch
from src.utils import get_freq_avg, set_seed, validate_epoch

set_seed(2023)


def str2bool(inp: str) -> bool:
    return inp.lower() in ["1", "true"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = get_parser()
    parser.add_argument(
        "ratio", help="Model Prune Ratio -- Higher = Prune more", type=float
    )

    parser.add_argument(
        "--run_name", help="Name of run. Used to store checkpoint name", default=None
    )

    # Other Optional Argument
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epoches", type=int, default=15)
    parser.add_argument("--log_step", type=int, default=1000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--disable_codebook", action="store_true")
    parser.add_argument("--freeze_head", action="store_true")
    parser.add_argument("--freeze_codebook", type=str2bool, default=True)
    parser.add_argument("--codebook_path", default="")

    args = parser.parse_args(argv)
    args = parse_args_const(args)

    if args.run_name is None:
        args.run_name = f"{args.dataset_name}-{args.model_name}-{args.ratio}"

    for name, value in vars(args).items():
        assert value is not None, f"{name} argument is not set"

    return args


def get_model(
    name: Literal["deepfm", "dcn"],
    checkpoint_path: str,
    shapley_value_path: str,
    freq_path: str,
    ratio: float,
    device: str = "cuda",
):
    """
    Args:
        name: Model name
        checkpoint_path: Path to the checkpoint model
        shapley_value_path: Path to Shapley value
        freq_path: Path to frequency file
        ratio: Prune ratio, the higher ratio, prune more
    """
    assert ratio <= 1 and ratio >= 0
    assert name in ["deepfm", "dcn"]

    for path in [checkpoint_path, shapley_value_path, freq_path]:
        assert os.path.exists(path), f"{path=} is not exists."

    checkpoint = torch.load(checkpoint_path, device)

    if name == "deepfm":
        model = DeepFM.load(checkpoint)
    elif name == "dcn":
        model = DCN_Mix.load(checkpoint)
        model = model._orig_mod
    else:
        raise NotImplementedError()

    with torch.no_grad():
        weight = model.get_emb()

    mask = torch.zeros_like(weight, dtype=torch.bool, device=device)

    freq = torch.load(freq_path).to(device)
    if args.codebook_path:
        base_value = torch.load(args.codebook_path)
    else:
        base_value = get_freq_avg(weight.to(device), freq, model.field_dims)
        base_value = base_value.to(device)
    # base_value = get_freq_avg(weight.to(device), freq, model.field_dims)
    # base_value = base_value.to(device)
    # codebook = get_feat_to_field(model.field_dims).to(device)

    # Sort
    shapley_value = torch.load(shapley_value_path)
    shapley_value = shapley_value.abs().flatten()

    n_rows, n_cols = model.get_emb_size()
    num_ele = int(n_rows * n_cols * ratio)

    idx = torch.argsort(shapley_value)
    idx = idx[:num_ele]
    idx1 = idx // n_cols
    idx2 = idx % n_cols

    # 1 is pruned
    mask[idx1, idx2] = 1
    loguru.logger.info(f"Num Params: {n_rows * n_cols - num_ele}")

    if args.disable_codebook:
        loguru.logger.info("No Codebook")
        emb = CodebookEmb(mask, weight, None)
    else:
        loguru.logger.info("Shapley - Codebook")
        emb = CodebookEmb(mask, weight, base_value)

    model.embedding = emb
    return model


if __name__ == "__main__":
    args = parse_args()
    name = args.run_name

    logger = Logger(log_name=name)
    logger.info(args)

    logger.info(f"{args.dataset_name=}-{args.model_name=}-{args.ratio=}")
    checkpoint_path = f"checkpoints/{name}.pth"

    device = "cuda"

    model = get_model(
        args.model_name,
        args.checkpoint_path,
        args.shapley_value_path,
        args.freq_path,
        args.ratio,
        device,
    )
    model.to(device)

    logger.info("Load train dataset...")
    train_dataset = get_dataset(args.dataset_name, "train")
    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    logger.info("Load val dataset...")
    val_dataset = get_dataset(args.dataset_name, "val")
    val_loader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    logger.info("Successfully load val dataset")
    val_metrics = validate_epoch(val_loader, model, device)
    for metric, value in val_metrics.items():
        logger.log_metric(f"val/{metric}", value, -1)

    params = []
    for name, p in model.named_parameters():
        if args.freeze_head and "embedding" not in name:
            continue
        if "codebook" not in name:
            params.append(p)

    params = [
        {"params": params, "weight_decay": args.wd, "lr": args.lr},
    ]
    if not args.freeze_codebook:
        params.append(
            {"params": model.embedding.codebook, "weight_decay": args.wd, "lr": args.lr}
        )
    optimizer = torch.optim.Adam(
        params,
        args.lr,
        weight_decay=args.wd,
    )

    best_auc = 0
    early_stop_count = 0

    test_dataset = get_dataset(args.dataset_name, "test")
    test_loader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info("Successfully load test dataset")
    val_metrics = validate_epoch(test_loader, model, device)
    print(val_metrics)

    for epoch_idx in range(args.num_epoches):
        logger.info(f"{epoch_idx=}")

        train_metrics = train_epoch(
            train_loader,
            model,
            optimizer,
            device,
            args.log_step,
            freeze_head=args.freeze_head,
        )
        for metric, value in train_metrics.items():
            logger.log_metric(f"train/{metric}", value, epoch_idx)

        val_metrics = validate_epoch(val_loader, model, device)
        # val_metrics = validate_epoch(test_loader, model, device)
        for metric, value in val_metrics.items():
            logger.log_metric(f"val/{metric}", value, epoch_idx)

        if best_auc < val_metrics["auc"]:
            logger.info("New best, saving model...")
            best_auc = val_metrics["auc"]

            checkpoint = {
                "state_dict": model.state_dict(),
                "run_config": vars(args),
                "val_metrics": val_metrics,
                "field_dims": train_dataset.field_dims,
            }
            torch.save(checkpoint, checkpoint_path)
            early_stop_count = 0
        else:
            early_stop_count += 1
            logger.debug(f"{early_stop_count=}")

            if args.early_stop > 0 and early_stop_count >= args.early_stop:
                break

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    test_dataset = get_dataset(args.dataset_name, "test")
    test_loader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logger.info("Validate Test set")

    test_result = validate_epoch(test_loader, model, device)
    logger.info(test_result)
    logger.info(f"AUC: {test_result['auc']}")
