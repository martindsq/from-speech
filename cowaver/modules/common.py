import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


def unpack_batch(batch):
    if len(batch) == 2:
        (x, y), labels = batch
        task_ids = None
    elif len(batch) == 3:
        (x, y), labels, task_ids = batch
    elif len(batch) == 4:
        (x, y), labels, _, _ = batch
        task_ids = None
    else:
        (x, y), labels, task_ids, _, _ = batch
    return x, y.squeeze(1), labels, task_ids


def unpack_ctc_targets(batch):
    if len(batch) == 4:
        return batch[2], batch[3]
    if len(batch) == 5:
        return batch[3], batch[4]
    return None, None


class CTCHeadMixin:
    def _setup_ctc(self, latent_dim: int, ctc_vocab_size: int = 0, ctc_weight: float = 0.0, ctc_blank: int = 0):
        if ctc_weight < 0:
            raise ValueError("ctc_weight must be non-negative.")
        if ctc_vocab_size < 0:
            raise ValueError("ctc_vocab_size must be non-negative.")
        if ctc_weight > 0 and ctc_vocab_size <= 1:
            raise ValueError("ctc_vocab_size must include blank plus at least one symbol when ctc_weight > 0.")

        self.ctc_weight = ctc_weight
        self.ctc_head = None
        self.ctc_loss = None
        if ctc_weight > 0:
            self.ctc_head = nn.Linear(latent_dim, ctc_vocab_size)
            self.ctc_loss = nn.CTCLoss(blank=ctc_blank, zero_infinity=True)

    @property
    def has_ctc(self) -> bool:
        return self.ctc_head is not None

    def ctc_logits(self, h: Tensor) -> Tensor | None:
        if self.ctc_head is None:
            return None
        return self.ctc_head(h)

    def ctc_training_loss(self, h: Tensor, batch) -> Tensor:
        if self.ctc_head is None:
            return h.new_zeros(())

        targets, target_lengths = unpack_ctc_targets(batch)
        if targets is None or target_lengths is None:
            raise ValueError("CTC is enabled, but the batch does not contain CTC targets.")
        if target_lengths.max().item() > h.size(1):
            raise ValueError(
                f"CTC target length exceeds encoder steps: max target is {target_lengths.max().item()}, "
                f"but the encoder produced {h.size(1)} steps."
            )

        logits = self.ctc_logits(h)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full(
            size=(h.size(0),),
            fill_value=h.size(1),
            dtype=torch.long,
            device=h.device,
        )
        return self.ctc_loss(
            log_probs,
            targets.to(h.device),
            input_lengths,
            target_lengths.to(h.device),
        )


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
