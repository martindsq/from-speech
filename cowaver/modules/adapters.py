import torch
import torch.nn as nn
from torch import Tensor

from .common import ResidualTemporalBlock


class TemporalReLUNorm(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        return self.proj(h)


class ConvolutionalTemporalAdapter(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, latent_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(latent_dim, kernel_size=3, dilation=1),
            ResidualTemporalBlock(latent_dim, kernel_size=3, dilation=1),
            ResidualTemporalBlock(latent_dim, kernel_size=3, dilation=1),
        )

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        x = h.transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        return x.transpose(1, 2)


class RecurrentTemporalAdapter(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 256, num_layers: int = 2, **_):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError("RecurrentTemporalAdapter requires an even latent_dim.")
        self.in_proj = nn.Conv1d(input_dim, latent_dim, kernel_size=1)
        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
        x = self.in_proj(h.transpose(1, 2)).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.out_proj(x)


class TransformerTemporalAdapter(nn.Module):
    def __init__(self, input_dim: int = 512, latent_dim: int = 256, width_steps: int = 24, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, latent_dim, kernel_size=1)
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
        x = self.in_proj(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.positional[:, :x.size(1)]
        x = self.encoder(x)
        return self.norm(x)


ADAPTER_REGISTRY = {
    "convolutional": ConvolutionalTemporalAdapter,
    "relu-norm": TemporalReLUNorm,
    "recurrent": RecurrentTemporalAdapter,
    "transformer": TransformerTemporalAdapter,
}


def build_temporal_adapter(adapter: str = "convolutional", **kwargs):
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter]
    except KeyError as exc:
        options = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown temporal adapter '{adapter}'. Options: {options}") from exc
    return adapter_cls(**kwargs)


TemporalAdapter = ConvolutionalTemporalAdapter
RecurrentAdapter = RecurrentTemporalAdapter
TransformerAdapter = TransformerTemporalAdapter
