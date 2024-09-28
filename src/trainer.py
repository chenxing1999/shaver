import datetime
from typing import Dict, List, Tuple, Union

import loguru
import torch
from torch.utils.data import DataLoader

from src.base import CTRModel

now = datetime.datetime.now

# first is feat (with offset), second is label
CTR_DATA = Tuple[torch.Tensor, float]


def train_epoch(
    dataloader: DataLoader[CTR_DATA],
    model: CTRModel,
    optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    device="cuda",
    log_step=10,
    profiler=None,
    clip_grad=0,
    *,
    freeze_head=False,
) -> Dict[str, float]:
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    if not freeze_head:
        model.train()
    else:
        model.embedding.train()
    model.to(device)

    loss_dict: Dict[str, float] = dict(loss=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    load_data_time = datetime.timedelta()
    train_time = datetime.timedelta()
    first_start = start = now()

    for idx, batch in enumerate(dataloader):
        load_data_time += now() - start

        start_train = now()
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels.float())
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        for optimizer in optimizers:
            optimizer.step()

        loss_dict["loss"] += loss.item()

        # Logging
        if log_step and idx % log_step == 0:
            msg = f"Idx: {idx}"

            for metric, value in loss_dict.items():
                if value > 0:
                    avg = value / (idx + 1)
                    msg += f" - {metric}: {avg:.4}"

            loguru.logger.info(msg)

        if profiler:
            profiler.step()

        end_train = start = now()
        train_time += end_train - start_train

    for metric, value in loss_dict.items():
        avg = value / (idx + 1)
        loss_dict[metric] = avg

    loguru.logger.info(f"train_time: {train_time}")
    loguru.logger.info(f"load_data_time: {load_data_time}")

    total_time = now() - first_start
    loguru.logger.info(f"total_time: {total_time}")

    return loss_dict
