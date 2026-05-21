import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, StepLR

from .adapters import build_temporal_adapter
from .common import unpack_batch
from .cornet import CORnet_Z
from .decoders import build_decoder
from ..models import DataModule, TestResults, TrainableModule


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


class CoWaverUnconditioned(TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, width_steps: int = 24, adapter: str = "convolutional", decoder: str = "convolutional", name_prefix: str = "cowaver_unconditioned"):
        adapter_suffix = "" if adapter == "convolutional" else f"_ad{adapter.replace('-', '_')}"
        decoder_suffix = "" if decoder == "convolutional" else f"_dc{decoder.replace('-', '_')}"
        super().__init__(name=f"{name_prefix}_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}_ws{width_steps}{adapter_suffix}{decoder_suffix}")
        self.mel_bins = mel_bins
        self.visual_encoder = ImageToHorizontalFeatures(
            feature_dim=latent_dim,
            width_steps=width_steps
        )
        self.adapter = build_temporal_adapter(
            adapter,
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
