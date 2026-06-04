import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .common import ResidualTemporalBlock


class ConvolutionalMelDecoder(nn.Module):
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
        return self.out_proj(x)

class RecurrentMelDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, mel_bins: int = 40, seq_len: int = 49, num_layers: int = 1):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("RecurrentMelDecoder requires an even hidden_size.")
        self.seq_len = seq_len
        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, mel_bins)
        )
        self.refiner = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, kernel_size=5, padding=2),
        )

    def forward(self, z: Tensor):
        if z.dim() == 2:
            z = z.unsqueeze(0)
        x = z.transpose(1, 2)
        x = F.interpolate(
            x,
            size=self.seq_len,
            mode="linear",
            align_corners=False
        ).transpose(1, 2)
        x, _ = self.rnn(x)
        raw_mel = self.out_proj(x).transpose(1, 2)
        refined_mel = raw_mel + self.refiner(raw_mel)
        return refined_mel

class Seq2SeqMelDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, mel_bins: int = 40, seq_len: int = 49, num_layers: int = 2):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("Seq2SeqMelDecoder requires an even hidden_size.")
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.init_proj = nn.Linear(hidden_size, hidden_size)
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.time_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, mel_bins)

    def forward(self, z: Tensor):
        if z.dim() == 2:
            z = z.unsqueeze(0)
        _, hidden = self.encoder(z)
        hidden = hidden.view(self.num_layers, 2, z.size(0), self.hidden_size // 2)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)
        hidden = torch.tanh(self.init_proj(hidden))
        decoder_input = self.start_token.expand(z.size(0), self.seq_len, -1)
        decoder_input = decoder_input + self.time_embedding.expand(z.size(0), -1, -1)
        x, _ = self.decoder(decoder_input, hidden.contiguous())
        x = self.norm(x)
        return self.out_proj(x).transpose(1, 2)


class TransformerMelDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, mel_bins: int = 40, seq_len: int = 49, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.memory_proj = nn.Linear(latent_dim, hidden_size)
        self.time_queries = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, mel_bins)

    def forward(self, z: Tensor):
        if z.dim() == 2:
            z = z.unsqueeze(0)
        memory = self.memory_proj(z)
        queries = self.time_queries.expand(z.size(0), -1, -1)
        x = self.decoder(queries, memory)
        x = self.norm(x)
        return self.out_proj(x).transpose(1, 2)


DECODER_REGISTRY = {
    "convolutional": ConvolutionalMelDecoder,
    "recurrent": RecurrentMelDecoder,
    "seq2seq": Seq2SeqMelDecoder,
    "transformer": TransformerMelDecoder,
}


def build_decoder(decoder: str = "convolutional", **kwargs):
    try:
        decoder_cls = DECODER_REGISTRY[decoder]
    except KeyError as exc:
        options = ", ".join(sorted(DECODER_REGISTRY))
        raise ValueError(f"Unknown decoder '{decoder}'. Options: {options}") from exc
    return decoder_cls(**kwargs)


HorizontalFeaturesToMel = ConvolutionalMelDecoder
