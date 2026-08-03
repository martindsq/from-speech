import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer

from .autoencoders import ConvAutoEncoder
from .cornet import CORnet_Z
from ..models import DataModule, TestResults, TrainProgramme, TrainableMixin
from ..utils import distancia_mel, topk
    
class SpeechAutoEncoder(TrainableMixin, ConvAutoEncoder):
    def __init__(self, h_dim: int = 256, seq_len: int = 49, mel_bins: int = 80):
        super().__init__(
            x_dim=(1, seq_len, mel_bins),
            h_dim=h_dim,
            n_filters=32,
            filter_size=5
        )
        self.h_dim = h_dim
        self.seq_len = seq_len
        self.mel_bins = mel_bins

    @property
    def name(self) -> str:
        return f"speech_autoencoder_hd{self.h_dim}_sl{self.seq_len}_mb{self.mel_bins}"

    def training_step(self, batch: tuple):
        (image, phonetized_mel, spoken_mel), target = batch
        B = image.size(0)
        spoken_mel = spoken_mel.transpose(2, 3)
        y_hat = self(spoken_mel)
        return F.mse_loss(y_hat.reshape(B, -1), target=spoken_mel.reshape(B, -1))

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        (image, phonetized_mel, spoken_mel), target = batch
        spoken_mel = spoken_mel.transpose(2, 3)
        y_hat = self(spoken_mel)
        y_hat = y_hat.squeeze(1).transpose(1, 2)
        prototypes = data.mel_prototypes(y_hat.device)
        distances = distancia_mel(y_hat, prototypes)
        return topk(distances, target)

    def optimizer(self, programme: TrainProgramme) -> Optimizer:
        params = filter(lambda p: p.requires_grad, self.parameters())
        return AdamW(params=params, lr=programme.lr, weight_decay=1e-4)

class PhonologicalAwareness(TrainableMixin, ConvAutoEncoder):
    def __init__(self, h_dim: int = 256, seq_len: int = 49, mel_bins: int = 80):
        super().__init__(
            x_dim=(1, seq_len, mel_bins),
            h_dim=h_dim,
            n_filters=32,
            filter_size=5
        )
        self.h_dim = h_dim
        self.seq_len = seq_len
        self.mel_bins = mel_bins

        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

    @property
    def name(self) -> str:
        return f"phonological_awareness_hd{self.h_dim}_sl{self.seq_len}_mb{self.mel_bins}"

    def training_step(self, batch: tuple):
        (image, phonetized_mel, spoken_mel), target = batch
        B = image.size(0)
        phonetized_mel = phonetized_mel.transpose(2, 3)
        spoken_mel = spoken_mel.transpose(2, 3)
        y_hat = self(phonetized_mel)
        return F.mse_loss(y_hat.reshape(B, -1), target=spoken_mel.reshape(B, -1))

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        (image, phonetized_mel, spoken_mel), target = batch
        phonetized_mel = phonetized_mel.transpose(2, 3)
        spoken_mel = spoken_mel.transpose(2, 3)
        y_hat = self(phonetized_mel)
        y_hat = y_hat.squeeze(1).transpose(1, 2)
        prototypes = data.mel_prototypes(y_hat.device)
        distances = distancia_mel(y_hat, prototypes)
        return topk(distances, target)

    def optimizer(self, programme: TrainProgramme) -> Optimizer:
        params = filter(lambda p: p.requires_grad, self.parameters())
        return AdamW(params=params, lr=programme.lr, weight_decay=1e-4)

class PhonologicalRoute(TrainableMixin, nn.Module):
    def __init__(self, h_dim=256, seq_len: int = 49, mel_bins: int = 80):
        super().__init__()
        self.h_dim = h_dim
        self.seq_len = seq_len
        self.mel_bins = mel_bins
        
        self.cornet = CORnet_Z()
        self.cornet.module.decoder = nn.Identity()

        # Freeze CORnet-Z
        for param in self.cornet.parameters():
            param.requires_grad = False
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 7))
        
        self.rnn = nn.RNN(
            input_size=512,
            hidden_size=256,
            nonlinearity='tanh',
            batch_first=True,
            bidirectional=False
        )
        self.proj = nn.Linear(256, h_dim)
        
        self.phonological_awareness = PhonologicalAwareness(
            h_dim=h_dim,
            seq_len=seq_len,
            mel_bins=mel_bins
        )

        # Freeze Phonological Awareness
        for param in self.phonological_awareness.parameters():
            param.requires_grad = False

        self.register_buffer('h_mean', torch.zeros(h_dim))
        self.register_buffer('h_std', torch.ones(h_dim))

    @property
    def name(self) -> str:
        return f"phonological_route_hd{self.h_dim}_sl{self.seq_len}_mb{self.mel_bins}"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        v = self.cornet(x)
        v = self.avg_pool(v).squeeze(2)
        v = v.transpose(1, 2)
        out, h_n = self.rnn(v)
        h_norm = self.proj(h_n[0])
        return h_norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h_norm = self.encode(x)
        h = h_norm * self.h_std + self.h_mean
        mel = self.phonological_awareness.decoder(h)
        return mel, h

    def training_step(self, batch: tuple):
        (image, phonetized_mel, _), target = batch
        phonetized_mel = phonetized_mel.transpose(2, 3)
        with torch.no_grad():
            h = self.phonological_awareness.encoder(phonetized_mel)
            h_norm = (h - self.h_mean) / (self.h_std + 1e-6)
        h_hat_norm = self.encode(image)
        return F.mse_loss(h_norm, h_hat_norm)

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        (image, _, spoken_mel), target = batch
        predicted_mel, h = self(image)
        predicted_mel = predicted_mel.squeeze(1).transpose(1, 2)
        prototypes = data.mel_prototypes(predicted_mel.device)
        distances = distancia_mel(predicted_mel, prototypes)
        return topk(distances, target)

    def optimizer(self, programme: TrainProgramme) -> Optimizer:
        params = filter(lambda p: p.requires_grad, self.parameters())
        return AdamW(params=params, lr=programme.lr, weight_decay=1e-4)
    
    def set_h_stats(self, h_mean: torch.Tensor, h_std: torch.Tensor):
        """Llamar una vez, antes de entrenar, con las stats reales del
        dataset.
        """
        self.h_mean.copy_(h_mean)
        self.h_std.copy_(h_std)
