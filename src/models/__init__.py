from src.base import CTRModel

from .dcn import DCN_Mix, DCNv2
from .deepfm import DeepFM


def get_model(model_name: str, checkpoint: str) -> CTRModel:
    if model_name == "dcn":
        model = DCN_Mix.load(checkpoint)
        model = model._orig_mod
    elif model_name == "deepfm":
        model = DeepFM.load(checkpoint)
    elif model_name == "dcnv2":
        model = DCNv2.load(checkpoint)
    else:
        raise NotImplementedError()
    return model
