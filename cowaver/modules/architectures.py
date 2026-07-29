import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR

from .adapters import build_temporal_adapter
from .common import unpack_batch, Postnet
from .decoders import build_decoder
from .encoders import AvgPooledITEncoder
from ..models import DataModule, TestResults, TrainProgramme, TrainableModule
from ..utils import distancia_mel


class CoWaver(TrainableModule):
    def __init__(
        self,
        name_prefix: str,
        latent_dim: int = 256,
        hidden_size: int = 256,
        seq_len: int = 49,
        mel_bins: int = 40,
        adapter: str = "convolutional",
        decoder: str = "convolutional",
    ):
        adapter_suffix = "" if adapter == "convolutional" else f"_ad{adapter.replace('-', '_')}"
        decoder_suffix = "" if decoder == "convolutional" else f"_dc{decoder.replace('-', '_')}"
        super().__init__(
            name=f"{name_prefix}_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}"
            f"{adapter_suffix}{decoder_suffix}"
        )
        self.mel_bins = mel_bins
        self.num_tasks = None
        self.visual_encoder = AvgPooledITEncoder()
        self.adapter = build_temporal_adapter(
            adapter,
            input_dim=self.visual_encoder.feature_dim,
            latent_dim=latent_dim,
        )
        self.postnet = Postnet()

    def encode(self, x: Tensor) -> Tensor:
        h = self.visual_encoder(x)
        return self.adapter(h)

    def autoencode(self, x: Tensor) -> Tensor:
        return self.mel_encoder(x)

    def decode(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        mel = self.decode(z)
        mel = mel + self.postnet(mel)
        return mel, z 

    def training_step(self, batch, batch_idx, phase: int):
        x, y, _ = unpack_batch(batch)
        z = self.encode(x)
        y_hat = self.decode(z, y=y)
        refined_y_hat = y_hat + self.postnet(y_hat)
        return F.mse_loss(y_hat, y) + F.mse_loss(refined_y_hat, y)

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        x, _, targets = unpack_batch(batch)
        y_hat, _ = self(x)
        prototypes = data.mel_prototypes(y_hat.device)
        distances = distancia_mel(y_hat, prototypes)

        k = min(5, distances.size(1))
        topk = distances.topk(k, dim=1, largest=False).indices

        targets = targets.view(-1, 1)

        top1 = (topk[:, :1] == targets).any(dim=1).sum().item()
        top3 = (topk[:, :min(3, k)] == targets).any(dim=1).sum().item()
        top5 = (topk[:, :k] == targets).any(dim=1).sum().item()

        return TestResults(top1=top1, top3=top3, top5=top5)

    def inference_step(self, batch: tuple) -> tuple[Tensor, Tensor]:
        x, _, _ = unpack_batch(batch)
        return self(x)

    def optimizer(self, phase: int, programme: TrainProgramme) -> Optimizer:
        return AdamW(params=self.parameters(), lr=programme.epsilon_zero, weight_decay=1e-4)

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

    def _single_task_indices(self, batch_size: int, device: torch.device) -> Tensor:
        if self.num_tasks is None:
            raise ValueError(f"{type(self).__name__} does not use route indices.")
        return torch.zeros(batch_size, dtype=torch.long, device=device)


class CoWaverUnconditioned(CoWaver):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 256,
        seq_len: int = 49,
        mel_bins: int = 40,
        adapter: str = "convolutional",
        decoder: str = "convolutional",
        name_prefix: str = "cowaver_unconditioned",
    ):
        super().__init__(
            name_prefix=name_prefix,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
            mel_bins=mel_bins,
            adapter=adapter,
            decoder=decoder,
        )
        self.decoder = build_decoder(
            decoder,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            seq_len=seq_len,
        )

    def decode(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        mel = self.decoder(z, y)
        return mel


class CoWaverConditioned(CoWaver):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 256,
        seq_len: int = 49,
        mel_bins: int = 40,
        num_tasks: int = 2,
        adapter: str = "convolutional",
        decoder: str = "convolutional",
    ):
        super().__init__(
            name_prefix="cowaver_conditioned",
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
            mel_bins=mel_bins,
            adapter=adapter,
            decoder=decoder,
        )
        self.num_tasks = num_tasks
        self.task_embedding = nn.Embedding(num_tasks, latent_dim)
        self.decoder = build_decoder(
            decoder,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            seq_len=seq_len,
        )

    def decode(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        task_indices = self._single_task_indices(z.size(0), z.device)
        z = z + self.task_embedding(task_indices).unsqueeze(1)
        mel = self.decoder(z, y)
        return mel


class CoWaverDualRoute(CoWaver):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 256,
        seq_len: int = 49,
        mel_bins: int = 40,
        num_tasks: int = 2,
        adapter: str = "convolutional",
        decoder: str = "convolutional",
    ):
        super().__init__(
            name_prefix="cowaver_dual_route",
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
            mel_bins=mel_bins,
            adapter=adapter,
            decoder=decoder,
        )
        self.num_tasks = num_tasks
        self.decoders = nn.ModuleList([
            build_decoder(
                decoder,
                latent_dim=latent_dim,
                hidden_size=hidden_size,
                mel_bins=mel_bins,
                seq_len=seq_len,
            )
            for _ in range(num_tasks)
        ])

    def decode(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        task_indices = self._single_task_indices(z.size(0), z.device)
        outputs = torch.stack([decoder(z, y) for decoder in self.decoders], dim=1)
        batch_indices = torch.arange(z.size(0), device=z.device)
        mel = outputs[batch_indices, task_indices]
        return mel
