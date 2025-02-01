# Shaver

The code for paper On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspective (WWW'25 - Oral)

# Installation

```shell
# create an environment. Ensure python >= 3.9
python -m venv env
source env/bin/activate

# install
pip install -e '.[dev]'
```

# Quick start

See [this notebook](notebooks/main.ipynb) for running with sample dataset and sample model.

To run with full dataset, please get the full checkpoints and datasets split from github.com/chenxing1999/recsys-benchmarks.

Then modify `src/const.py` for list of path to dataset and checkpoint accordingly.

First set the dataset and model name:

```shell
DATASET_NAME=<`criteo` or `avazu` or `kdd`>
MODEL_NAME=<`deepfm` or `dcn`>

# Create directory to store step result
mkdir artifacts
mkdir checkpoints
```

If use Codebook, you need to first calculate frequency:

```shell
python scripts/count_freq.py $DATASET_NAME artifacts/freq.bin
```

Set the frequency file in `src/const.py`.

Then calculate Shapley value with the accodingly hyperparameters:

```shell
# Run Algorithm 1 and save output to artifacts/shapley_value.bin
python scripts/sage_row.py $MODEL_NAME $DATASET_NAME \
    --codebook\
    --p_train 1\
    --output_path artifacts/shapley_value.bin


# No codebook
python scripts/sage_row.py $MODEL_NAME $DATASET_NAME \
    --p_train 1\
    --output_path artifacts/shapley_value_zero.bin
```

Run evaluate on test set:

```shell
# If use codebook
python scripts/evaluate_codebook.py $MODEL_NAME $DATASET_NAME \
   --shapley_value_path artifacts/shapley_value.bin

# If not use codebook
python scripts/evaluate.py $MODEL_NAME $DATASET_NAME \
  --shapley_value_path artifacts/shapley_value_zero.bin
```

Run train:

```shell

# By default, checkpoint store at `checkpoints/run_name.pth'

# Use codebook
sparse_rate=0.8
python scripts/train.py $MODEL_NAME $DATASET_NAME $sparse_rate \
    --shapley_value_path artifacts/shapley_value.bin\
    --run_name codebook80

# If not use codebook
python scripts/train.py $MODEL_NAME $DATASET_NAME $sparse_rate \
    --shapley_value_path artifacts/shapley_value_zero.bin\
    --disable_codebook\
    --run_name zero80
```

# Citation
If you find this repo helpful, please cite the below paper:
```
@inproceedings{
tran2025ondevice,
title={On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspective},
author={Hung Vinh Tran and Tong Chen and Guanhua Ye and Quoc Viet Hung Nguyen and Kai Zheng and Hongzhi Yin},
booktitle={THE WEB CONFERENCE 2025},
year={2025},
url={https://openreview.net/forum?id=k03hiubX3F}
}
```
