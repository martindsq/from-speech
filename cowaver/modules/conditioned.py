import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, LinearLR
from .adapters import build_temporal_adapter
from .common import CTCHead, unpack_batch
from .decoders import build_decoder
from .encoders import AvgPooledITEncoder
from ..models import DataModule, TestResults, TrainProgramme, TrainableModule


class CoWaverConditioned(TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, num_tasks: int = 2, adapter: str = "convolutional", decoder: str = "convolutional", *, ctc_vocab_size: int, ctc_weight: float = 0.0):
        adapter_suffix = "" if adapter == "convolutional" else f"_ad{adapter.replace('-', '_')}"
        decoder_suffix = "" if decoder == "convolutional" else f"_dc{decoder.replace('-', '_')}"
        ctc_suffix = "" if ctc_weight == 0 else f"_ctc{ctc_weight:g}"
        super().__init__(name=f"cowaver_conditioned_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}{adapter_suffix}{decoder_suffix}{ctc_suffix}")
        self.mel_bins = mel_bins
        self.num_tasks = num_tasks
        self.visual_encoder = AvgPooledITEncoder(
            feature_dim=latent_dim,
        )
        self.adapter = build_temporal_adapter(
            adapter,
            input_dim=latent_dim,
            latent_dim=latent_dim,
        )
        self.task_embedding = nn.Embedding(num_tasks, latent_dim)
        self.decoder = build_decoder(
            decoder,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            seq_len=seq_len
        )
        self.ctc_weight = ctc_weight
        self.ctc_head = CTCHead(
            latent_dim=latent_dim,
            vocab_size=ctc_vocab_size,
        )

    def _resolve_task_ids(self, task_ids, device: torch.device) -> Tensor:
        if task_ids is None:
            raise ValueError("CoWaverConditioned requires task_ids from the dataset.")
        task_ids = task_ids.to(device).long().view(-1)
        task_indices = task_ids - 1
        if (task_indices < 0).any() or (task_indices >= self.num_tasks).any():
            raise ValueError(f"task_ids must be in the range 1..{self.num_tasks}.")
        return task_indices

    def forward(self, x: Tensor, task_ids: Tensor | None = None, phase: int = 3) -> tuple[Tensor, Tensor]:
        h = self.visual_encoder(x)
        z = self.adapter(h)
        task_ids = self._resolve_task_ids(task_ids, z.device)
        z = z + self.task_embedding(task_ids).unsqueeze(1)
        mel = self.decoder(z)
        return mel, z

    def training_step(self, batch, batch_idx, phase: int):
        x, y, _, task_ids, ctc_targets, ctc_lengths = unpack_batch(batch)
        h = self.visual_encoder(x)
        z = self.adapter(h)
        task_ids = self._resolve_task_ids(task_ids, z.device)
        z = z + self.task_embedding(task_ids).unsqueeze(1)
        y_hat = self.decoder(z)
        mel_loss = F.l1_loss(y_hat, y)
        if self.ctc_weight == 0:
            return mel_loss
        ctc_loss = self.ctc_head.training_loss(h, ctc_targets, ctc_lengths)
        return mel_loss + self.ctc_weight * ctc_loss

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        x, _, targets, task_ids, _, _ = unpack_batch(batch)
        if task_ids is None and getattr(data, "task_id", None) is not None:
            task_ids = torch.full((targets.size(0),), data.task_id, dtype=torch.long, device=targets.device)
        y_hat, _ = self(x, task_ids=task_ids)
        prototypes = data.mel_prototypes(y_hat.device)

        distances = torch.abs(
            y_hat.unsqueeze(1) - prototypes.unsqueeze(0)
        ).mean(dim=(2, 3))

        k = min(5, distances.size(1))
        topk = distances.topk(k, dim=1, largest=False).indices
        targets = targets.view(-1, 1)

        top1 = (topk[:, :1] == targets).any(dim=1).sum().item()
        top3 = (topk[:, :min(3, k)] == targets).any(dim=1).sum().item()
        top5 = (topk[:, :k] == targets).any(dim=1).sum().item()
        return TestResults(top1=top1, top3=top3, top5=top5)

    def inference_step(self, batch: tuple) -> tuple[Tensor, Tensor]:
        x, _, _, task_ids, _, _ = unpack_batch(batch)
        return self(x, task_ids=task_ids)

    def optimizer(self, phase: int, programme: TrainProgramme) -> torch.optim.Optimizer:
        return AdamW(
            self.parameters(),
            lr=programme.epsilon_zero,
            weight_decay=1e-4
        )

    def scheduler(self, optimizer: Optimizer, phase: int, programme: TrainProgramme) -> LRScheduler:
        start_epoch = programme.epochs_before_phase(phase)
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=programme.end_factor,
            total_iters=programme.decay_epochs(),
            last_epoch=start_epoch - 1,
        )
