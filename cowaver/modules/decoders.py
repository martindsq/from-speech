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
        self.query_base = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.query_delta = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.002)
        # self.queries = nn.Parameter(torch.randn(seq_len, latent_dim) * 0.02)
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
        # q = self.queries.unsqueeze(0).expand(B, -1, -1)
        q = self.query_base + torch.cumsum(self.query_delta, dim=1)
        q = q.expand(B, -1, -1)
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
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=mel_bins,
            num_layers=2,
            batch_first=True
        )
        #self.out_proj = nn.Sequential(
        #    nn.Linear(hidden_size, mel_bins)
        #)
        self.refiner = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, kernel_size=3, padding=1),
        )

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
        x = x + self.ffn(x) 
        x, _ = self.rnn(x)
        mel = x.transpose(1, 2)#self.out_proj(x).transpose(1, 2)
        mel = mel + self.refiner(mel)
        return mel

class MultiHeadAttentionMelDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 256,
        mel_bins: int = 100,
        seq_len: int = 49
    ):
        super().__init__()

        self.seq_len = seq_len
        self.mel_bins = mel_bins
        self.hidden_size = hidden_size
        self.letter_pos = nn.Parameter(torch.randn(1, 7, latent_dim) * 0.02)
        self.time_queries = nn.Parameter(torch.randn(1, seq_len, latent_dim) * 0.02)

        # Cabezal de "duracion articulatoria" por cada embedding
        self.duration_head = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.duration_head.weight)
        nn.init.zeros_(self.duration_head.bias)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.to_mel = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, mel_bins)
        )
        self.refiner = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mel_bins, mel_bins, kernel_size=5, padding=2),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def _monotonic_bias(self, z: Tensor):
        """

        Parameters
        ----------
        z:
            Tensor de forma `[B, 7, latent_dim]`.

        Returns
        -------
        bias:
            Tensor de forma `[B, seq_len, 7]` para sumar a la atención.
        """
        B, L, _ = z.shape
        device = z.device
        durations = F.softplus(self.duration_head(z)).squeeze(-1)
        cum = torch.cumsum(durations, dim=1)
        total = cum[:, -1:].clamp_min(1e-6)
        centers = (cum - durations / 2) / total
        t = torch.arange(self.seq_len, device=device).float() / self.seq_len
        dist = (t.view(1, -1, 1) - centers.view(B, 1, L)) ** 2
        align_sigma = 0.12
        bias = -dist / (2 * align_sigma ** 2)
        return bias

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
        
        letters = z + self.letter_pos[:, :z.size(1)]
        queries = self.time_queries.expand(B, -1, -1)

        bias = self._monotonic_bias(z)
        bias = bias.repeat_interleave(self.cross_attn.num_heads, dim=0)

        x, _ = self.cross_attn(
            query=queries,
            key=letters,
            value=letters,
            attn_mask=bias,
            need_weights=False
        )
        x = self.norm1(queries + x)
        x = self.norm2(x + self.ffn(x))
        x, _ = self.rnn(x)          # (B, seq_len, hidden_size)
        mel = self.to_mel(x)
        mel = mel.transpose(1, 2)     # (B, mel_bins, seq_len)
        mel = mel + self.refiner(mel)
        return mel

class RecurrentMLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 256,
        mel_bins: int = 100,
        seq_len: int = 49,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mel_bins = mel_bins
        self.seq_len = seq_len
        self.proj = nn.Sequential(
            nn.Linear(latent_dim * 7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * seq_len),
        )
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=mel_bins,
            num_layers=2,
            batch_first=True
        )
        self.refiner = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mel_bins, mel_bins, kernel_size=3, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(0)
        B = z.size(0)
        x = z.flatten(start_dim=1)
        x = self.proj(x).view(B, self.seq_len, self.hidden_size)
        x, _ = self.rnn(x)
        mel = x.transpose(1, 2)
        mel = mel + self.refiner(mel)
        return mel

class MLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_size: int = 512,
        mel_bins: int = 40,
        seq_len: int = 49,
    ):
        super().__init__()

        self.mel_bins = mel_bins
        self.seq_len = seq_len
        latent_len = 7

        self.output = nn.Sequential(
            nn.Linear(latent_dim * latent_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, mel_bins * seq_len),
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(0)

        batch_size = z.size(0)

        x = z.flatten(start_dim=1)
        mel = self.output(x)
        mel = mel.view(batch_size, self.mel_bins, self.seq_len)

        return mel

DECODER_REGISTRY = {
    "convolutional": ConvolutionalMelDecoder,
    "recurrent": RecurrentMelDecoder,
    "mlp": MLPDecoder,
    "rmlp": RecurrentMLPDecoder,
    "attn": MultiHeadAttentionMelDecoder
}


def build_decoder(decoder: str = "convolutional", **kwargs):
    try:
        decoder_cls = DECODER_REGISTRY[decoder]
    except KeyError as exc:
        options = ", ".join(sorted(DECODER_REGISTRY))
        raise ValueError(f"Unknown decoder '{decoder}'. Options: {options}") from exc
    return decoder_cls(**kwargs)
