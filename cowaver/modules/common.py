import torch.nn as nn
from torch import Tensor


def unpack_batch(batch):
    if len(batch) == 2:
        (x, y), labels = batch
        task_ids = None
    else:
        (x, y), labels, task_ids = batch
    return x, y.squeeze(1), labels, task_ids


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)
