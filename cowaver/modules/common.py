import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


def unpack_batch(batch):
    (x, y), labels, task_ids, ctc_targets, ctc_lengths = batch
    return x, y.squeeze(1), labels, task_ids, ctc_targets, ctc_lengths


class CTCHead(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int, blank: int = 0):
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must include blank plus at least one symbol.")
        self.proj = nn.Linear(latent_dim, vocab_size)
        self.loss = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)

    def training_loss(self, h: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
        if target_lengths.max().item() > h.size(1):
            raise ValueError(
                f"CTC target length exceeds encoder steps: max target is {target_lengths.max().item()}, "
                f"but the encoder produced {h.size(1)} steps."
            )

        logits = self(h)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full(
            size=(h.size(0),),
            fill_value=h.size(1),
            dtype=torch.long,
            device=h.device,
        )
        return self.loss(
            log_probs,
            targets.to(h.device),
            input_lengths,
            target_lengths.to(h.device),
        )

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
