import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR

from ..models import DataModule, TestResults, TrainProgramme, TrainableModule
from ..utils import distancia_mel
from .architectures import CoWaverUnconditioned

class MelToMel(nn.Module):
    def __init__(
        self,
        mel_bins: int = 100,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(mel_bins, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_size, mel_bins)

        # Al inicio, la salida es aproximadamente igual al mel de entrada.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, mel):
        """
        mel: [B, mel_bins, T]
        """
        residual = mel
        mel = mel.transpose(1, 2)
        h = self.input_proj(mel)
        h = self.encoder(h)
        delta_mel = self.output_proj(h)
        delta_mel = delta_mel.transpose(1, 2)
        return residual + delta_mel

class MelToMel(nn.Module):
    def __init__(
        self,
        mel_bins: int = 100,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(mel_bins, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.to_delta_mel = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, mel_bins),
        )

        self.refiner = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mel_bins, mel_bins, kernel_size=5, padding=2),
        )

        # El modelo empieza aproximadamente como una identidad.
        nn.init.zeros_(self.to_delta_mel[-1].weight)
        nn.init.zeros_(self.to_delta_mel[-1].bias)

        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, mel: Tensor) -> Tensor:
        """
        Parameters
        ----------
        mel:
            Tensor de forma [B, mel_bins, T].

        Returns
        -------
        output:
            Tensor de forma [B, mel_bins, T].
        """
        residual = mel

        # [B, mel_bins, T] -> [B, T, mel_bins]
        h = mel.transpose(1, 2)

        # [B, T, mel_bins] -> [B, T, hidden_size]
        h = self.input_proj(h)

        # Dependencias globales entre frames.
        h = self.encoder(h)

        # Continuidad temporal.
        h, _ = self.rnn(h)

        # [B, T, hidden_size] -> [B, T, mel_bins]
        delta = self.to_delta_mel(h)

        # [B, T, mel_bins] -> [B, mel_bins, T]
        delta = delta.transpose(1, 2)

        # Primera corrección residual.
        mel_out = residual + delta

        # Corrección espectro-temporal local.
        mel_out = mel_out + self.refiner(mel_out)

        return mel_out

class PhonologicalAwareness(TrainableModule):
    def __init__(self, cowaver: CoWaverUnconditioned, seq_len: int = 49):
        mel_bins = cowaver.mel_bins
        super().__init__(name=f"phonol_awareness_sl{seq_len}_mb{mel_bins}")
        self.mel_bins = mel_bins
        self.seq_len = seq_len
        self.cowaver = cowaver

        for parameter in self.cowaver.parameters():
            parameter.requires_grad = False

        self.decoder = MelToMel()

    def encode(self, x: Tensor) -> Tensor:
        phonetized_mel, z = self.cowaver(x)
        return phonetized_mel

    def decode(self, phonetized_mel: Tensor):
        spoken_mel = self.decoder(phonetized_mel)
        return spoken_mel

    def forward(self, x: Tensor) -> Tensor:
        phonetized_mel = self.encode(x)
        spoken_mel = self.decode(phonetized_mel)
        return spoken_mel

    def training_step(self, batch, batch_idx, phase: int):
        (x, _, y), labels = batch
        y_hat = self(x)
        return F.l1_loss(y_hat, y.squeeze(1))

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        (x, _, y), targets = batch
        y_hat = self(x)
        prototypes = data.mel_prototypes(y_hat.device)
        distances = distancia_mel(y_hat, prototypes)
        k = min(5, distances.size(1))
        topk = distances.topk(k, dim=1, largest=False).indices
        targets = targets.view(-1, 1)
        top1 = (topk[:, :1] == targets).any(dim=1).sum().item()
        top3 = (topk[:, :min(3, k)] == targets).any(dim=1).sum().item()
        top5 = (topk[:, :k] == targets).any(dim=1).sum().item()
        return TestResults(top1=top1, top3=top3, top5=top5)

    def optimizer(self, phase: int, programme: TrainProgramme) -> Optimizer:
        return AdamW(
            params=self.parameters(),
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

