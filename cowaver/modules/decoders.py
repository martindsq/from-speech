import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ConvolutionalMelDecoder(nn.Module):
    """Decode a latent sequence into a fixed-length Mel spectrogram.

    Parameters
    ----------
    latent_dim:
        Number of features at each latent input position.
    hidden_size:
        Number of channels used by the temporal convolution.
    mel_bins:
        Number of frequency bins in each output Mel frame.
    seq_len:
        Number of temporal frames in the output spectrogram.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 256,
        mel_bins: int = 40,
        seq_len: int = 49,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.queries = nn.Parameter(torch.randn(seq_len, latent_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=2,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(latent_dim)
        self.conv = nn.Conv1d(
            latent_dim,
            hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.out_proj = nn.Conv1d(hidden_size, mel_bins, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:
        """Convert a latent sequence into a Mel spectrogram.

        Parameters
        ----------
        z:
            Tensor of shape `[B, 7, latent_dim]`.

        Returns
        -------
        mel: Tensor
            Tensor of shape `[B, mel_bins, seq_len]`.
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)
        B = z.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        x, _ = self.attention(
            query=q,
            key=z,
            value=z,
            need_weights=False
        )
        x = self.attention_norm(q + x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x = self.act(self.norm(x)).transpose(1, 2)
        return self.out_proj(x)

class RecurrentMelDecoder(nn.Module):
    """Decode a latent sequence using learned alignment and recurrence.

    Parameters
    ----------
    latent_dim:
        Number of features at each latent input position and attention query.
    hidden_size:
        Total number of features produced by the bidirectional GRU.
    mel_bins:
        Number of frequency bins in each output Mel frame.
    seq_len:
        Number of temporal frames in the output spectrogram.
    """

    def __init__(self, latent_dim: int = 256, hidden_size: int = 256, mel_bins: int = 40, seq_len: int = 49):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("RecurrentMelDecoder requires an even hidden_size.")
        self.seq_len = seq_len
        self.queries = nn.Parameter(torch.randn(seq_len, latent_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=2,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(latent_dim)
        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.out_proj = nn.Linear(hidden_size, mel_bins)

    def forward(self, z: Tensor):
        """Convert a latent sequence into a Mel spectrogram.

        Parameters
        ----------
        z:
            Tensor of shape `[B, 7, latent_dim]`.

        Returns
        -------
        mel: Tensor
            Tensor of shape `[B, mel_bins, seq_len]`.
        """
        if z.dim() == 2:
            z = z.unsqueeze(0)
        B = z.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        x, _ = self.attention(
            query=q,
            key=z,
            value=z,
            need_weights=False
        )
        x = self.attention_norm(q + x)
        x, _ = self.rnn(x)
        mel = self.out_proj(x).transpose(1, 2)
        return mel

DECODER_REGISTRY = {
    "convolutional": ConvolutionalMelDecoder,
    "recurrent": RecurrentMelDecoder
}


def build_decoder(decoder: str = "convolutional", **kwargs):
    try:
        decoder_cls = DECODER_REGISTRY[decoder]
    except KeyError as exc:
        options = ", ".join(sorted(DECODER_REGISTRY))
        raise ValueError(f"Unknown decoder '{decoder}'. Options: {options}") from exc
    return decoder_cls(**kwargs)
