import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, StepLR
from .adapters import build_temporal_adapter
from .common import CTCHeadMixin, unpack_batch
from .decoders import build_decoder
from .encoders import ImageToHorizontalFeatures
from ..models import DataModule, TestResults, TrainableModule


class CoWaverConditioned(CTCHeadMixin, TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, width_steps: int = 24, height_bands: int = 4, num_tasks: int = 2, adapter: str = "convolutional", decoder: str = "convolutional", ctc_vocab_size: int = 0, ctc_weight: float = 0.0):
        adapter_suffix = "" if adapter == "convolutional" else f"_ad{adapter.replace('-', '_')}"
        decoder_suffix = "" if decoder == "convolutional" else f"_dc{decoder.replace('-', '_')}"
        height_suffix = f"_hb{height_bands}"
        ctc_suffix = "" if ctc_weight == 0 else f"_ctc{ctc_weight:g}"
        super().__init__(name=f"cowaver_conditioned_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}_ws{width_steps}{height_suffix}{adapter_suffix}{decoder_suffix}{ctc_suffix}")
        self.mel_bins = mel_bins
        self.num_tasks = num_tasks
        self.visual_encoder = ImageToHorizontalFeatures(
            feature_dim=latent_dim,
            width_steps=width_steps,
            height_bands=height_bands,
        )
        self.adapter = build_temporal_adapter(
            adapter,
            input_dim=latent_dim,
            latent_dim=latent_dim,
            width_steps=width_steps,
        )
        self.task_embedding = nn.Embedding(num_tasks, latent_dim)
        self.decoder = build_decoder(
            decoder,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            seq_len=seq_len
        )
        self._setup_ctc(
            latent_dim=latent_dim,
            ctc_vocab_size=ctc_vocab_size,
            ctc_weight=ctc_weight,
        )

    def _resolve_task_ids(self, task_ids, device: torch.device) -> Tensor:
        if task_ids is None:
            raise ValueError("CoWaverConditioned requires task_ids from the dataset.")
        task_ids = task_ids.to(device).long().view(-1)
        task_indices = task_ids - 1
        if (task_indices < 0).any() or (task_indices >= self.num_tasks).any():
            raise ValueError(f"task_ids must be in the range 1..{self.num_tasks}.")
        return task_indices

    def forward(self, x: Tensor, task_ids: Tensor | None = None, phase: int = 3) -> tuple[Tensor, Tensor, Tensor | None]:
        h = self.visual_encoder(x)
        z = self.adapter(h)
        task_ids = self._resolve_task_ids(task_ids, z.device)
        z = z + self.task_embedding(task_ids).unsqueeze(1)
        mel = self.decoder(z)
        return mel, z, self.ctc_logits(h)

    def training_step(self, batch, batch_idx, phase: int):
        x, y, _, task_ids = unpack_batch(batch)
        h = self.visual_encoder(x)
        z = self.adapter(h)
        task_ids = self._resolve_task_ids(task_ids, z.device)
        z = z + self.task_embedding(task_ids).unsqueeze(1)
        y_hat = self.decoder(z)
        mel_loss = F.l1_loss(y_hat, y)
        return mel_loss + self.ctc_weight * self.ctc_training_loss(h, batch)

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        x, _, targets, task_ids = unpack_batch(batch)
        if task_ids is None and getattr(data, "task_id", None) is not None:
            task_ids = torch.full((targets.size(0),), data.task_id, dtype=torch.long, device=targets.device)
        y_hat, _, _ = self(x, task_ids=task_ids)
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
        x, _, _, task_ids = unpack_batch(batch)
        return self(x, task_ids=task_ids)

    def optimizer(self, phase: int) -> torch.optim.Optimizer:
        if phase == 1:
            lrs = (3e-5, 3e-4, 1e-3)
        elif phase == 2:
            lrs = (3e-6, 1e-4, 3e-4)
        else:
            lrs = (3e-6, 5e-5, 1e-4)
        param_groups = [
                {"params": self.visual_encoder.parameters(), "lr": lrs[0]},
                {"params": self.adapter.parameters(), "lr": lrs[1]},
                {"params": self.task_embedding.parameters(), "lr": lrs[1]},
                {"params": self.decoder.parameters(), "lr": lrs[2]},
        ]
        if self.ctc_head is not None:
            param_groups.append({"params": self.ctc_head.parameters(), "lr": lrs[1]})
        return AdamW(
            param_groups,
            weight_decay=1e-4
        )

    def scheduler(self, optimizer: Optimizer, phase: int) -> LRScheduler:
        return StepLR(optimizer, step_size=10, gamma=0.5)
