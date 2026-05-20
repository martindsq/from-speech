import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, StepLR
from .convolutional import (
    HorizontalFeaturesToMel,
    ImageToHorizontalFeatures,
    TemporalAdapter,
    unpack_batch,
)
from ..models import DataModule, TestResults, TrainableModule


class CoWaverDualRoute(TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, width_steps: int = 24, num_tasks: int = 3):
        super().__init__(name=f"cowaver_dual_route_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}_ws{width_steps}")
        self.mel_bins = mel_bins
        self.num_tasks = num_tasks
        self.visual_encoder = ImageToHorizontalFeatures(
            feature_dim=latent_dim,
            width_steps=width_steps
        )
        self.adapter = TemporalAdapter(
            input_dim=latent_dim,
            latent_dim=latent_dim,
        )
        self.decoders = torch.nn.ModuleList([
            HorizontalFeaturesToMel(
                latent_dim=latent_dim,
                hidden_size=hidden_size,
                mel_bins=mel_bins,
                seq_len=seq_len
            )
            for _ in range(num_tasks)
        ])

    @staticmethod
    def _phase_task_id(phase: int) -> int:
        return max(0, min(phase - 1, 2))

    def _resolve_task_ids(self, task_ids, batch_size: int, phase: int, device: torch.device) -> Tensor:
        if task_ids is None:
            task_ids = torch.full((batch_size,), self._phase_task_id(phase), dtype=torch.long, device=device)
        else:
            task_ids = task_ids.to(device).long().view(-1)
        return task_ids.clamp(0, self.num_tasks - 1)

    def forward(self, x: Tensor, task_ids: Tensor | None = None, phase: int = 3) -> tuple[Tensor, Tensor]:
        h = self.visual_encoder(x)
        z = self.adapter(h)
        task_ids = self._resolve_task_ids(task_ids, z.size(0), phase, z.device)
        outputs = torch.stack([decoder(z) for decoder in self.decoders], dim=1)
        batch_indices = torch.arange(z.size(0), device=z.device)
        mel = outputs[batch_indices, task_ids]
        return mel, z

    def training_step(self, batch, batch_idx, phase: int):
        x, y, _, task_ids = unpack_batch(batch)
        y_hat, _ = self(x, task_ids=task_ids, phase=phase)
        return F.l1_loss(y_hat, y)

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        x, _, targets, task_ids = unpack_batch(batch)
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
        x, _, _, task_ids = unpack_batch(batch)
        return self(x, task_ids=task_ids)

    def optimizer(self, phase: int) -> torch.optim.Optimizer:
        if phase == 1:
            lrs = (3e-5, 3e-4, 1e-3)
        elif phase == 2:
            lrs = (3e-6, 1e-4, 3e-4)
        else:
            lrs = (3e-6, 5e-5, 1e-4)
        return AdamW(
            [
                {"params": self.visual_encoder.parameters(), "lr": lrs[0]},
                {"params": self.adapter.parameters(), "lr": lrs[1]},
                {"params": self.decoders.parameters(), "lr": lrs[2]},
            ],
            weight_decay=1e-4
        )

    def scheduler(self, optimizer: Optimizer, phase: int) -> LRScheduler:
        return StepLR(optimizer, step_size=10, gamma=0.5)
