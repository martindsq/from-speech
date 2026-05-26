import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, StepLR

from .adapters import build_temporal_adapter
from .common import CTCHead, unpack_batch
from .decoders import build_decoder
from .encoders import AvgPooledITEncoder
from ..models import DataModule, TestResults, TrainableModule


class CoWaverUnconditioned(TrainableModule):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, seq_len: int = 49, mel_bins: int = 40, width_steps: int = 24, height_bands: int = 4, adapter: str = "convolutional", decoder: str = "convolutional", name_prefix: str = "cowaver_unconditioned", *, ctc_vocab_size: int, ctc_weight: float = 0.0):
        adapter_suffix = "" if adapter == "convolutional" else f"_ad{adapter.replace('-', '_')}"
        decoder_suffix = "" if decoder == "convolutional" else f"_dc{decoder.replace('-', '_')}"
        height_suffix = f"_hb{height_bands}"
        ctc_suffix = "" if ctc_weight == 0 else f"_ctc{ctc_weight:g}"
        super().__init__(name=f"{name_prefix}_lt{latent_dim}_hs{hidden_size}_sl{seq_len}_mb{mel_bins}_ws{width_steps}{height_suffix}{adapter_suffix}{decoder_suffix}{ctc_suffix}")
        self.mel_bins = mel_bins
        self.visual_encoder = AvgPooledITEncoder(
            feature_dim=latent_dim,
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
        self.ctc_weight = ctc_weight
        self.ctc_head = CTCHead(
            latent_dim=latent_dim,
            vocab_size=ctc_vocab_size,
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
        x, y, _, _, ctc_targets, ctc_lengths = unpack_batch(batch)
        h = self.visual_encoder(x)
        z = self.adapter(h)
        y_hat = self.decoder(z)
        mel_loss = F.l1_loss(y_hat, y)
        if self.ctc_weight == 0:
            return mel_loss
        ctc_loss = self.ctc_head.training_loss(h, ctc_targets, ctc_lengths)
        return mel_loss + self.ctc_weight * ctc_loss

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        x, _, targets, _, _, _ = unpack_batch(batch)
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
        x, _, _, _, _, _ = unpack_batch(batch)
        return self(x)

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
                {"params": self.decoder.parameters(), "lr": lrs[2]},
        ]
        param_groups.append({"params": self.ctc_head.parameters(), "lr": lrs[1]})
        return AdamW(
            param_groups,
            weight_decay=1e-4
        )

    def scheduler(self, optimizer: Optimizer, phase: int) -> LRScheduler:
        return StepLR(optimizer, step_size=10, gamma=0.5)
