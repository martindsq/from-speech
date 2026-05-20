import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, StepLR
from .common import unpack_batch
from .convolutional import ImageToHorizontalFeatures
from .decoders import build_decoder
from ..models import DataModule, TestResults, TrainableModule


class TransformerAdapter(nn.Module):
    def __init__(self, input_dim: int = 256, latent_dim: int = 256, width_steps: int = 24, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, latent_dim)
        self.positional = nn.Parameter(torch.zeros(1, width_steps, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        x = self.in_proj(h)
        x = x + self.positional[:, :x.size(1)]
        x = self.encoder(x)
        return self.norm(x)


class CoWaverTransformer(TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, width_steps: int = 24, decoder: str = "convolutional"):
        decoder_suffix = "" if decoder == "convolutional" else f"_dc{decoder.replace('-', '_')}"
        super().__init__(name=f"cowaver_transformer_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}_ws{width_steps}{decoder_suffix}")
        self.mel_bins = mel_bins
        self.visual_encoder = ImageToHorizontalFeatures(
            feature_dim=latent_dim,
            width_steps=width_steps
        )
        self.adapter = TransformerAdapter(
            input_dim=latent_dim,
            latent_dim=latent_dim,
            width_steps=width_steps,
        )
        self.decoder = build_decoder(
            decoder,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            seq_len=seq_len
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.visual_encoder(x)
        z = self.adapter(h)
        mel = self.decoder(z)
        return mel, z

    def training_step(self, batch, batch_idx, phase: int):
        x, y, _, _ = unpack_batch(batch)
        y_hat, _ = self(x)
        return F.l1_loss(y_hat, y)

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        x, _, targets, _ = unpack_batch(batch)
        y_hat, _ = self(x)
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
        x, _, _, _ = unpack_batch(batch)
        return self(x)

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
                {"params": self.decoder.parameters(), "lr": lrs[2]},
            ],
            weight_decay=1e-4
        )

    def scheduler(self, optimizer: Optimizer, phase: int) -> LRScheduler:
        return StepLR(optimizer, step_size=10, gamma=0.5)
