import argparse
from typing import Final

DEFAULT_DICTIONARY: Final[dict] = {
    "criteo": {
        "deepfm": dict(
            checkpoint_path="/home/xing/workspace/phd/shapley//checkpoints/criteo-deepfm-best.pth",
            shapley_value_path="./artifacts/criteo_res_v5/cols_shapley_field.pth",
            freq_path="/home/xing/workspace/phd/shapley/my_code/artifacts/criteo/criteo_freq.bin",
            # wd=1e-5,
            # lr=1e-4,
            wd=1.3876614923531129e-05,
            lr=0.00010399611233634769,
            batch_size=2048,
        ),
        "dcn": dict(
            checkpoint_path="/home/xing/workspace/phd/shapley//checkpoints/criteo-dcn-best.pth",
            shapley_value_path="./artifacts/criteo_res_v5-dcn/cols_shapley_field_dcn_criteo.pth",
            freq_path="/home/xing/workspace/phd/shapley/my_code/artifacts/criteo/criteo_freq.bin",
            wd=4.980568409796324e-05,
            lr=0.00016000142575522478,
            batch_size=8192,
        ),
    },
    "avazu": {
        "deepfm": dict(
            checkpoint_path="/home/xing/workspace/phd/shapley//checkpoints/avazu-best-deepfm.pth",
            shapley_value_path="./artifacts/avazu_res_v5/cols_shapley_field.pth",
            freq_path="/home/xing/workspace/phd/shapley/my_code/artifacts/avazu/avazu_freq.bin",
            # wd=3e-5,
            # lr=2e-5,
            lr=2.760593174284856e-05,
            wd=3.5683572330141996e-05,
            batch_size=2048,
        ),
        "dcn": dict(
            checkpoint_path="/home/xing/workspace/phd/shapley//checkpoints/avazu-dcn-best.pth",
            shapley_value_path="./artifacts/avazu_res_v5-dcn/cols_shapley_field_codebook_mix.pth",
            freq_path="/home/xing/workspace/phd/shapley/my_code/artifacts/avazu/avazu_freq.bin",
            wd=1.0361109028617641e-05,
            lr=1.0361109028617641e-05,
            batch_size=8192,
        ),
    },
    "kdd": {
        "deepfm": dict(
            checkpoint_path="/home/xing/workspace/phd/shapley/checkpoints/kdd-deepfm-best.pth",
            shapley_value_path="./artifacts/kdd-deepfm-v5/codebook.pth",
            freq_path="/home/xing/workspace/phd/shapley/my_code/artifacts/kdd/kdd_freq.bin",
            lr=4.2496200892571215e-05,
            wd=1.1357382431638127e-05,
            batch_size=8192,
        ),
        "dcn": dict(
            checkpoint_path="/home/xing/workspace/phd/shapley/checkpoints/kdd-dcn-best.pth",
            shapley_value_path="./artifacts/kdd-dcn-v5/codebook.pth",
            freq_path="/home/xing/workspace/phd/shapley/my_code/artifacts/kdd/kdd_freq.bin",
            lr=1.0792370095904069e-05,
            wd=3.6565362068177706e-05,
            batch_size=8192,
        ),
    },
}


DATASET_INFO: Final = {
    "criteo": dict(
        train_test_info="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/criteo-common-split/train_test_val_info.bin",
        dataset_name="train",
        dataset_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/train.txt",
        cache_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/criteo/criteo-fm",
    ),
    "avazu": dict(
        train_test_info="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/train_test_info.bin",
        dataset_name="train",
        dataset_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/train",
        cache_path="/home/xing/workspace/phd/recsys-benchmark/dataset/ctr/avazu/avazu-fm",
    ),
    "kdd": dict(
        train_test_info="/home/xing/workspace/phd/run-code/dataset/ctr/kdd/preprocessed/train_test_val_info.bin",
        dataset_name="train",
        dataset_path="/home/xing/workspace/phd/run-code/dataset/ctr/kdd/track2/training.txt",
        cache_path="/home/xing/workspace/phd/run-code/.kdd",
    ),
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    dataset_names = list(DEFAULT_DICTIONARY.keys())
    model_names = list(DEFAULT_DICTIONARY[dataset_names[0]].keys())
    # Compulsory argument
    parser.add_argument(
        "model_name",
        help="Model name",
        choices=model_names,
    )
    parser.add_argument("dataset_name", help="Dataset name", choices=dataset_names)

    # Optional argument -- set based on model name + dataset name
    parser.add_argument(
        "--checkpoint_path", default=None, help="Path to initial checkpoint"
    )
    parser.add_argument(
        "--shapley_value_path", default=None, help="Path to Shapley Value"
    )
    parser.add_argument("--freq_path", default=None, help="Path to frequency file")

    parser.add_argument("--wd", type=float, help="weight decay", default=None)
    parser.add_argument("--lr", type=float, help="learning rate", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser


def parse_args_const(args: argparse.Namespace) -> argparse.Namespace:
    """Parse Args based on DEFAULT_DICTIONARY"""

    # Parse optional argument -- I am lazy to retype these every time
    dataset_name = args.dataset_name
    model_name = args.model_name
    defaults = DEFAULT_DICTIONARY[dataset_name][model_name]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            if value is None:
                raise NotImplementedError(
                    f"Not support empty {name} for {dataset_name} - {model_name}"
                )
            setattr(args, name, value)

    return args
