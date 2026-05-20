import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, StepLR
from .cornet import CORnet_Z
from ..models import TrainableModule, DataModule, TestResults


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


class ImageToHorizontalFeatures(nn.Module):
    def __init__(self, feature_dim: int = 256, width_steps: int = 24, height_bands: int = 4):
        super().__init__()

        self.feature_dim = feature_dim
        self.width_steps = width_steps
        self.height_bands = height_bands

        self.cornet_z = CORnet_Z()
        self.cornet_z.module.decoder = nn.Identity()
        self.projector = nn.Conv1d(512 * height_bands, feature_dim, kernel_size=1)

    def forward(self, x):
        """Makes an inference.

        Parameters
        ----------
        x: Tensor
            Un tensor de forma [B, C, H, W] donde B es el tamaño del batch, C
            es el número de canales (tipicamente 3), H y W la altura y el ancho
            de las imágenes respectivamente. Se puede omitir B.

        Returns
        ------
        h: Tensor
            Un tensor de forma [B, width_steps, feature_dim]. B es igual a 1 si se
            omitió en x.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.cornet_z(x)
        # Preserve coarse vertical structure before turning width into time.
        features = F.interpolate(
            features,
            size=(self.height_bands, features.size(-1)),
            mode="bilinear",
            align_corners=False
        )
        B, C, H, W = features.shape
        features = features.reshape(B, C * H, W)
        features = F.interpolate(
            features,
            size=self.width_steps,
            mode="linear",
            align_corners=False
        )
        features = self.projector(features)
        return features.transpose(1, 2)


class TemporalAdapter(nn.Module):
    def __init__(self, input_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, latent_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(latent_dim, kernel_size=5, dilation=1),
            ResidualTemporalBlock(latent_dim, kernel_size=5, dilation=2),
            ResidualTemporalBlock(latent_dim, kernel_size=3, dilation=1),
        )

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        x = h.transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        return x.transpose(1, 2)


class HorizontalFeaturesToMel(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, mel_bins: int = 40, seq_len: int = 49):
        super().__init__()
        self.seq_len = seq_len

        self.in_proj = nn.Conv1d(latent_dim, hidden_size, kernel_size=1)
        self.pre_blocks = nn.Sequential(
            ResidualTemporalBlock(hidden_size, kernel_size=5, dilation=1),
            ResidualTemporalBlock(hidden_size, kernel_size=5, dilation=2),
        )
        self.post_blocks = nn.Sequential(
            ResidualTemporalBlock(hidden_size, kernel_size=5, dilation=1),
            ResidualTemporalBlock(hidden_size, kernel_size=3, dilation=1),
        )
        self.out_proj = nn.Conv1d(hidden_size, mel_bins, kernel_size=1)

    def forward(self, z: Tensor):
        """Makes an inference.

        Parameters
        ----------
        z: Tensor
            Un tensor de forma [B, seq_len, latent_dim]. Se puede omitir B.

        Returns
        ------
        mel: Tensor
            Un tensor de forma [B, mel_bins, seq_len]. B es igual a 1 si se
            omitió en x.
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)

        x = z.transpose(1, 2)
        x = self.in_proj(x)
        x = self.pre_blocks(x)
        x = F.interpolate(
            x,
            size=self.seq_len,
            mode="linear",
            align_corners=False
        )
        x = self.post_blocks(x)
        mel = self.out_proj(x)
        return mel


class CoWaverConvolutional(TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, width_steps: int = 24):
        super().__init__(name=f"cowaver_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}_ws{width_steps}")
        self.mel_bins = mel_bins
        self.visual_encoder = ImageToHorizontalFeatures(
            feature_dim=latent_dim,
            width_steps=width_steps
        )
        self.adapter = TemporalAdapter(
            input_dim=latent_dim,
            latent_dim=latent_dim,
        )
        self.decoder = HorizontalFeaturesToMel(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            seq_len=seq_len
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Hace una inferencia.

        Parameters
        ----------
        x: Tensor
            Un tensor de forma [B, C, H, W] donde B es el tamaño del batch, C
            es el número de canales (tipicamente 3), H y W la altura y el ancho
            de las imágenes respectivamente. Se puede omitir B.

        Returns
        ------
        mel: Tensor
            Un tensor de forma [B, mel_bins, seq_len]. B es igual a 1 si se
            omitió en x.
        z: Tensor
            Un tensor de forma [B, seq_len, latent_dim]. B es igual a 1 si se
            omitió en x.
        """
        h = self.visual_encoder(x)
        z = self.adapter(h)
        mel = self.decoder(z)
        return mel, z

    def training_step(self, batch, batch_idx, phase: int):
        x, y, _, _ = unpack_batch(batch)
        y_hat, _ = self(x)
        loss = F.l1_loss(y_hat, y)
        return loss

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
