from pathlib import Path
import numpy as np
import torch
import os
from typing import BinaryIO, IO


def data_loader(
    dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    start_ids = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x1 = torch.stack(
        [torch.from_numpy(dataset[start_idx : start_idx + context_length]).to(device) for start_idx in start_ids], dim=0
    )
    x2 = torch.stack(
        [
            torch.from_numpy(dataset[start_idx + 1 : start_idx + context_length + 1]).to(device)
            for start_idx in start_ids
        ],
        dim=0,
    )
    return x1, x2


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


class Dataset:
    def __init__(
        self, dataset_dir: os.PathLike, context_length: int, batch_size: int, device: torch.device, **kwargs
    ) -> None:
        dataset_dir = Path(dataset_dir)
        self.train_data = np.memmap(dataset_dir / "train.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.val_data = np.memmap(dataset_dir / "test.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        return data_loader(data, self.batch_size, self.context_length, self.device)
