import argparse
import copy
import os
from typing import Literal, Optional, Sequence

import loguru
import numba
import numpy as np
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from numba import cuda
from torch import nn
from torch.utils.data import DataLoader

from src.const import get_parser, parse_args_const
from src.datasets import get_dataset
from src.loggers import Logger
from src.models.dcn import DCN_Mix
from src.models.deepfm import DeepFM
from src.utils import get_freq_avg


class SparseCodebookEmb(nn.Module):
    """Efficient Implementation for CodebookEmb used for On-device"""

    def __init__(
        self,
        codebook_mask=None,
        weight=None,
        codebook: Optional[torch.FloatTensor] = None,
        n_bits: int = 32,
    ):
        """
        Args:
            codebook_mask: torch.Bool
                0 for not pruned
                1 for pruned

            weight: torch.FloatTensor (NumFeat x HiddenSize)
                original model weight

            codebook: torch.FloatTensor (NumField x HiddenSize)
                if None --> Set all to zero and disable gradient

            n_bits: 8, 16, 32
                If n_bits in [8, 16] --> Use int with corresponding bits
        """
        super().__init__()
        assert n_bits in [4, 8, 16, 32], f"{n_bits=}. Not supported"

        self.codebook = codebook.data.cpu().numpy()
        if weight is not None:
            weight: torch.Tensor = weight * torch.logical_not(codebook_mask)

            weight = weight.to_sparse_csr().cpu()

            self.values = weight.values().numpy()
            self.crow_indices = weight.crow_indices().numpy().astype(np.int32)

            # col_indices from 0 to 16
            self.col_indices = weight.col_indices().numpy().astype(np.uint8)

        self.hidden_size = codebook.shape[1]

        # Let's only support CPU for now
        self.is_cuda = False

    def get_extra_state(self):
        state = {}
        state["codebook"] = self.codebook
        state["values"] = self.values
        state["crow_indices"] = self.crow_indices
        state["col_indices"] = self.col_indices
        state["is_cuda"] = self.is_cuda
        return state

    def set_extra_state(self, state):
        self.codebook = state["codebook"]
        self.values = state["values"]
        self.crow_indices = state["crow_indices"]
        self.col_indices = state["col_indices"]
        self.is_cuda = state["is_cuda"]
        self.hidden_size = self.codebook.shape[1]

    def forward(self, x):
        original_shape = x.shape

        if self.is_cuda:
            x = x.flatten()
            tensor_size = x.shape[0]
            x = numba.cuda.as_cuda_array(x)
            hidden_size = self.hidden_size

            n_threads = 32
            n_blocks = (tensor_size - 1) // 32 + 1
            results = np.repeat(self.codebook, tensor_size, 0).copy()
            results = numba.cuda.as_cuda_array(results)

            csr_embedding_lookup[n_blocks, n_threads](
                self.values,
                self.crow_indices,
                self.col_indices,
                x,
                results,
                tensor_size,
                hidden_size,
            )
            results = torch.as_tensor(results, device="cuda")
            return results.reshape(*original_shape, self.hidden_size)
        else:
            batch_size = x.shape[0]
            x = x.flatten().numpy()
            tensor_size = x.shape[0]
            hidden_size = self.hidden_size

            # results = np.empty((tensor_size, hidden_size), dtype=np.float32)
            results = np.repeat(self.codebook, batch_size, 0).copy()

            csr_embedding_lookup_cpu(
                self.values,
                self.crow_indices,
                self.col_indices,
                x,
                results,
                tensor_size,
                hidden_size,
            )
            results = torch.as_tensor(results)
            return results.reshape(*original_shape, self.hidden_size)

    def forward_torch(self, x):
        batch_size = x.shape[0]
        results = self.codebook.repeat(batch_size, 0).copy()
        x.shape[0]
        self.hidden_size

        csr_embedding_lookup_cpu(
            self.values,
            self.crow_indices,
            self.col_indices,
            x,
            results,
        )
        return results


class DcnModelHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.cross_head = model.cross_head
        self._dnn = model._dnn

    def forward(self, emb: torch.FloatTensor, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            emb: torch.FloatTensor (Batch x NumField x HiddenSize)
            x: torch.LongTensor (Batch x NumField)

        Return:
            scores: torch.FloatTensor (Batch): Logit result before sigmoid
        """
        bs = emb.shape[0]
        emb = emb.view(bs, -1)
        cross_logit = self.cross_head(emb)

        scores = self._dnn(cross_logit)
        scores = scores.squeeze(-1)
        return scores


@cuda.jit
def csr_embedding_lookup(
    values,
    crow_indices,
    col_indices,
    ids,
    outputs,
    size,
    hidden_size,
):
    """
    Get value from

    Args:
        values
        crow_indices
        col_indices
        ids: Index to select from
        outputs: to store output
            shape: N x D

        size: ids size
        hidden_size: Size of final dimension of outputs
    """
    index = cuda.grid(1)

    if index >= size:
        return

    rowid = ids[index]
    left = crow_indices[rowid]
    right = crow_indices[rowid + 1]

    # for i in range(hidden_size):
    #     outputs[index][i] = 0

    for i in range(left, right):
        outputs[index][col_indices[i]] = values[i]


nb_typehint = numba.void(
    numba.float32[:],
    # numba.int64[::1],
    # numba.int64[::1],
    numba.int32[::1],
    numba.uint8[::1],
    numba.int64[::1],
    numba.float32[:, :],
    numba.uint32,
    numba.uint32,
)


from numba.pycc import CC

cc = CC("my_module")


# @numba.jit(nb_typehint, nopython=True)
@cc.export("csr_embedding_lookup_cpu", nb_typehint)
def csr_embedding_lookup_cpu(
    values,
    crow_indices,
    col_indices,
    ids,
    outputs,
    size,
    hidden_size,
):
    left = crow_indices[ids]
    right = crow_indices[ids + 1]

    # outputs[:] = 0

    for index in range(size):
        for i in range(left[index], right[index]):
            outputs[index][col_indices[i]] = values[i]


### I am lazy here. Fix Later
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
        weight = model.get_emb().to(device)

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
    mask = mask.to(device)

    if args.disable_codebook:
        loguru.logger.info("No Codebook")
        emb = SparseCodebookEmb(mask, weight, None)
    else:
        loguru.logger.info("Shapley - Codebook")
        emb = SparseCodebookEmb(mask, weight, base_value)

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
    logger.info(f"Num params: {len(model.embedding.values)}")

    generator = torch.Generator()
    generator.manual_seed(42)

    x = torch.randint(model.field_dims.sum(), size=(64, 11), generator=generator)
    train_dataset = get_dataset(args.dataset_name, "train")
    train_loader = DataLoader(
        train_dataset,
        # args.batch_size,
        64,
        shuffle=args.shuffle,
        num_workers=0,
    )
    x, y = next(iter(train_loader))
    torch.save(x, "outs/sample_inp.pth")

    # with torch.no_grad():
    #     emb = model.embedding(x)

    type_dict = {
        "int8": "b",
        "uint8": "B",
        "float32": "f",
        "int32": "l",
        "int16": "i",
    }

    emb_state = model.embedding.state_dict()["_extra_state"]
    # for k, v in emb_state.items():
    #     if isinstance(v, np.ndarray):
    #         arr_type = type_dict[str(v.dtype)]
    #         emb_state[k] = array.array(arr_type, v.flatten().tolist())
    torch.save(emb_state, "outs/emb_sparse.pkl")
    torch.save(DcnModelHead(model).state_dict(), "outs/head.pkl")
    torch.save(model.state_dict(), f"outs/model_{args.ratio}.pkl")
    exit()

    print("Infer normal")
    model = DcnModelHead(model)
    model.cpu()
    model.eval()
    # print(timeit.timeit(lambda: model(x), number=100))
    # model = torch.compile(model)
    # print("Infer compile")
    # print(timeit.timeit(lambda: model(x), number=100))
    # torch.onnx.export(model, (x,), "xxx.onnx")
    # exit()

    ep = torch.export.export(
        model,
        (
            emb,
            x,
        ),
    )
    model = ep.module()
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=True,
            _skip_dim_order=True,
            # TODO(T182187531): enable dim order in xnnpack
        ),
    )
    logger.info(f"Exported and lowered graph:\n{edge.exported_program().graph}")

    # this is needed for the ETRecord as lowering modifies the graph in-place
    edge_copy = copy.deepcopy(edge)

    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    # if args.etrecord is not None:
    #     generate_etrecord(args.etrecord, edge_copy, exec_prog)
    #     logger.info(f"Saved ETRecord to {args.etrecord}")

    quant_tag = "fp32"
    model_name = f"{args.model_name}_xnnpack_{quant_tag}"
    save_pte_program(exec_prog, model_name, "outs/")
