from typing import Dict, List, Literal, Optional

import numba
import numpy as np
import torch
from loguru import logger
from numba import cuda
from torch import nn

from src.models.dcn import DCN_Mix, DCN_MixHead
from src.models.deepfm import DeepFM


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

        if codebook is not None:
            self.codebook = codebook.data.cpu().numpy()
            self.hidden_size = codebook.shape[1]

        if weight is not None:
            weight: torch.Tensor = weight * torch.logical_not(codebook_mask)

            weight = weight.to_sparse_csr().cpu()

            self.values = weight.values().numpy()
            self.crow_indices = weight.crow_indices().numpy().astype(np.int32)

            # col_indices from 0 to 16
            self.col_indices = weight.col_indices().numpy().astype(np.uint8)

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
    def __init__(
        self,
        field_dims: List[int],
        num_factor: int,
        hidden_sizes: List[int],
        num_layers: int = 3,
        num_experts: int = 4,
        rank: int = 64,
        activation: Optional[str] = None,
        embedding_config: Optional[Dict] = None,
        p_dropout=0.5,
        empty_embedding=False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"Received {kwargs}")

        if not embedding_config:
            embedding_config = {"name": "vanilla"}

        sum(field_dims)

        inp_size = num_factor * len(field_dims)
        self.cross_head = DCN_MixHead(
            num_experts,
            num_layers,
            rank,
            inp_size,
            activation,
        )
        layers: List[nn.Module] = []
        for size in hidden_sizes:
            layers.append(nn.Linear(inp_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))

            inp_size = size

        layers.append(nn.Linear(inp_size, 1))
        self._dnn = nn.Sequential(*layers)

        # self._field_dims = torch.tensor(field_dims)
        # field_dims_tensor = torch.tensor(field_dims)
        # field_dims_tensor = torch.cat(
        #     [torch.tensor([0], dtype=torch.long), field_dims_tensor]
        # )

        # offsets = torch.cumsum(field_dims_tensor[:-1], 0).unsqueeze(0)
        # self.register_buffer("offsets", offsets)
        # self.offsets = offsets

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


# @cc.export("csr_embedding_lookup_cpu", nb_typehint)
# @numba.jit(nb_typehint, nopython=True)
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


def get_model(
    name: Literal["deepfm", "dcn"],
    checkpoint_path: str,
    device: str = "cpu",
):
    """
    Args:
        name: Model name
        checkpoint_path: Path to the checkpoint model
        shapley_value_path: Path to Shapley value
        freq_path: Path to frequency file
        ratio: Prune ratio, the higher ratio, prune more
    """
    assert name in ["deepfm", "dcn"]
    checkpoint = torch.load(checkpoint_path, device)

    if name == "deepfm":
        model = DeepFM.load(checkpoint)
    elif name == "dcn":
        model = DCN_Mix.load(checkpoint)
        model = model._orig_mod
    else:
        raise NotImplementedError()
    return model


class ModelWrapper(nn.Module):
    def __init__(self, emb, head: DcnModelHead):
        super().__init__()
        self.emb = emb
        self.head = head

    def forward(self, x):
        emb = self.emb(x)
        res = self.head(emb, x)
        return res


def get_model_sparse(
    name: Literal["deepfm", "dcn"],
    config_path: str,
    head_path: str,
    emb_path: str,
    device: str = "cpu",
):
    """
    Args:
        name: Model name

    """
    # assert ratio <= 1 and ratio >= 0
    assert name in ["deepfm", "dcn"]

    config = torch.load(config_path, device)

    field_dims = config["field_dims"]
    model_config = config["model_config"]

    if name == "deepfm":
        raise NotImplementedError()
        # model = DeepFM.load(checkpoint)
    elif name == "dcn":
        model_head = DcnModelHead(field_dims, **model_config)
        emb = SparseCodebookEmb()
        head_state = torch.load(head_path, device)
        model_head.load_state_dict(head_state)

        emb_state = torch.load(emb_path, device)
        emb.load_state_dict(emb_state)
    else:
        raise NotImplementedError()

    model = ModelWrapper(emb, model_head)
    return model


if __name__ == "__main__":
    # args = parse_args()
    # name = args.run_name

    # logger = Logger(log_name=name)
    # logger.info(args)

    # logger.info(f"{args.dataset_name=}-{args.model_name=}-{args.ratio=}")
    # checkpoint_path = f"checkpoints/{name}.pth"

    device = "cpu"

    model = get_model_sparse(
        "dcn",
        "outs/config.pth",
        "outs/head.pkl",
        "outs/emb_sparse_0.8.pkl",
        "cpu",
    )
    sample_inp = torch.load("outs/sample_inp.pth")

    model.eval()
    with torch.no_grad():
        model(sample_inp)
