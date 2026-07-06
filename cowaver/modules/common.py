import torch.nn as nn
from torch import Tensor


def unpack_batch(batch):
    (x, y), labels, task_ids = batch
    return x, y.squeeze(1), labels, task_ids

class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        return self.act(x + y)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return self.act(x + y)
