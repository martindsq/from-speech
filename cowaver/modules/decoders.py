from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class _LocationLayer(nn.Module):
    """Location layer used by attention to look at previous alignment history."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        conv = nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3, dilation=1, bias=False)
        nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain("linear"))
        self.location_conv = conv

        dense = nn.Linear(32, 128, bias=False)
        nn.init.xavier_uniform_(dense.weight, gain=nn.init.calculate_gain("tanh"))
        self.location_dense = dense

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        """Extract features from the attention weight history.

        Parameters
        ----------
        attention_weights_cat:
            Previous and cumulative attention weights, tensor of shape `[B, 2, seq_len]`.

        Returns
        -------
        processed_attention: Tensor
            Tensor of shape `[B, seq_len, 128]`.
        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class _Attention(nn.Module):
    """Locally sensitive attention over the encoder memory."""

    def __init__(self, latent_dim: int, rnn_dim: int) -> None:
        super().__init__()

        query = nn.Linear(rnn_dim, 128, bias=False)
        nn.init.xavier_uniform_(query.weight, gain=nn.init.calculate_gain("tanh"))
        self.query_layer = query

        memory = nn.Linear(latent_dim, 128, bias=False)
        nn.init.xavier_uniform_(memory.weight, gain=nn.init.calculate_gain("tanh"))
        self.memory_layer = memory

        v = nn.Linear(128, 1, bias=False)
        nn.init.xavier_uniform_(v.weight, gain=nn.init.calculate_gain("linear"))
        self.v = v

        self.location_layer = _LocationLayer(latent_dim)

    def forward(
        self,
        attention_hidden_state: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        attention_weights_cat: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        processed_query = self.query_layer(attention_hidden_state.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        alignment = energies.squeeze(2)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class _Prenet(nn.Module):
    """Two-layer, dropout-regularized projection applied to each decoder input
    frame.

    It restricts and noise up how much of the raw previous frame reaches the
    decoder, which discourages the model from just copying it forward instead
    of relying on attention.
    """

    def __init__(self, mel_bins: int, hidden_size: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for in_size, out_size in [(mel_bins, hidden_size), (hidden_size, hidden_size)]:
            linear = nn.Linear(in_size, out_size, bias=False)
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain("linear"))
            self.layers.append(linear)

    def forward(self, x: Tensor) -> Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Tacotron2Decoder(nn.Module):
    r"""An autoregressive, attention-based decoder that converts an encoder
    memory sequence into a mel spectrogram.

    Parameters
    ----------
    latent_dim:
        Number of dimensions in the encoder memory `z`.
    hidden_size:
        Width of the prenet's hidden bottleneck applied to each decoder input frame.
    mel_bins:
        Number of mel bins in the target spectrogram.
    seq_len:
        Fixed length of the encoder memory `z`.
    """

    def __init__(self,
        latent_dim: int = 512,
        hidden_size: int = 256,
        mel_bins: int = 80,
        seq_len: int = 49
    ) -> None:
        super().__init__()

        self.mel_bins = mel_bins
        self.seq_len = seq_len
        self.attention_rnn_dim = 1024
        self.decoder_rnn_dim = 1024

        self.prenet = _Prenet(mel_bins, hidden_size)

        self.attention_rnn = nn.LSTMCell(hidden_size + latent_dim, self.attention_rnn_dim)
        self.attention_layer = _Attention(latent_dim, self.attention_rnn_dim)
        self.decoder_rnn = nn.LSTMCell(self.attention_rnn_dim + latent_dim, self.decoder_rnn_dim)

        projection = nn.Linear(self.decoder_rnn_dim + latent_dim, mel_bins)
        nn.init.xavier_uniform_(projection.weight, gain=nn.init.calculate_gain("linear"))
        self.linear_projection = projection

    def forward(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        """Pass the input through the AttentionDecoder.

        Parameters
        ----------
        z:
            Tensor of shape `[B, 7, latent_dim]`.
        y:
            Ground-truth mel spectrogram used for teacher forcing, of shape
            `[B, mel_bins, seq_len]`, or `None` to decode autoregressively.

        Returns
        -------
        mel: Tensor
            Predicted mel spectrogram, of shape `[B, mel_bins, seq_len]`.
        """
        B = z.size(0)
        T = y.size(2) if y is not None else self.seq_len
        dtype, device = z.dtype, z.device

        decoder_input = torch.zeros(B, self.mel_bins, dtype=dtype, device=device)

        attention_hidden = torch.zeros(B, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(B, self.attention_rnn_dim, dtype=dtype, device=device)
        decoder_hidden = torch.zeros(B, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(B, self.decoder_rnn_dim, dtype=dtype, device=device)
        attention_weights = torch.zeros(B, 7, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(B, 7, dtype=dtype, device=device)
        attention_context = torch.zeros(B, z.size(2), dtype=dtype, device=device)
        processed_memory = self.attention_layer.memory_layer(z)

        mel_outputs = []
        for t in range(T):
            prenet_input = self.prenet(decoder_input)

            cell_input = torch.cat((prenet_input, attention_context), -1)
            attention_hidden, attention_cell = self.attention_rnn(
                cell_input,
                (attention_hidden, attention_cell)
            )
            attention_hidden = F.dropout(attention_hidden, 0.1, self.training)

            attention_weights_cat = torch.cat(
                (attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1
            )
            attention_context, attention_weights = self.attention_layer(
                attention_hidden, z, processed_memory, attention_weights_cat
            )
            attention_weights_cum += attention_weights

            decoder_rnn_input = torch.cat((attention_hidden, attention_context), -1)
            decoder_hidden, decoder_cell = self.decoder_rnn(decoder_rnn_input, (decoder_hidden, decoder_cell))
            decoder_hidden = F.dropout(decoder_hidden, 0.1, self.training)

            decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
            mel_output = self.linear_projection(decoder_hidden_attention_context)

            mel_outputs += [mel_output]

            decoder_input = y[:, :, t] if y is not None else mel_output

        mel = torch.stack(mel_outputs, dim=2)
        return mel

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

    def forward(self, z: Tensor, y: Tensor | None = None) -> Tensor:
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

    def forward(self, z: Tensor, y: Tensor | None = None):
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

    def forward(self, z: Tensor, y: Tensor | None = None):
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

    def forward(self, z: Tensor, y: Tensor | None = None) -> Tensor:
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

    def forward(self, z: Tensor, y: Tensor | None = None) -> Tensor:
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
    "attn": MultiHeadAttentionMelDecoder,
    "tacotron": Tacotron2Decoder
}


def build_decoder(decoder: str = "convolutional", **kwargs):
    try:
        decoder_cls = DECODER_REGISTRY[decoder]
    except KeyError as exc:
        options = ", ".join(sorted(DECODER_REGISTRY))
        raise ValueError(f"Unknown decoder '{decoder}'. Options: {options}") from exc
    return decoder_cls(**kwargs)
