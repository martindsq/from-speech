import torch
from torch import nn, Tensor
from torch.nn import functional as F

class IdentityAdapter(nn.Module):
    """Use each temporal position independently as latent space.

    Parameters
    ----------
    input_dim:
        Number of features at each input position.
    latent_dim:
        Number of features at each output position.
    """
    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()
        if input_dim != latent_dim:
            raise ValueError("input_dim != latent_dim")

    def forward(self, h: Tensor):
        """Return the same visual sequence without transforming it.

        Parameters
        ----------
        h:
            Tensor of shape `[B, seq_len, input_dim]`.

        Returns
        -------
        z: Tensor
            Tensor of shape `[B, seq_len, latent_dim]`.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        z = h
        return z

class PointwiseTemporalAdapter(nn.Module):
    """Project each temporal position independently into the latent space.

    Parameters
    ----------
    input_dim:
        Number of features at each input position.
    latent_dim:
        Number of features at each output position.
    """

    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim),
        #     nn.GELU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.LayerNorm(latent_dim),
        #     nn.GELU(),
        # )

    def forward(self, h: Tensor) -> Tensor:
        """Project a visual sequence without mixing temporal positions.

        Parameters
        ----------
        h:
            Tensor of shape `[B, seq_len, input_dim]`.

        Returns
        -------
        z: Tensor
            Tensor of shape `[B, seq_len, latent_dim]`.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        z = self.proj(h)
        return z


class ResidualPointwiseTemporalAdapter(nn.Module):
    """Project each temporal position with a direct path plus learned
    correction."""

    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()
        self.skip = (
            nn.Identity()
            if input_dim == latent_dim
            else nn.Linear(input_dim, latent_dim, bias=False)
        )
        self.delta = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, h: Tensor) -> Tensor:
        """Project a visual sequence without mixing temporal positions.

        Parameters
        ----------
        h:
            Tensor of shape `[B, 7, input_dim]`.

        Returns
        -------
        z: Tensor
            Tensor of shape `[B, 7, latent_dim]`.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        z = self.skip(h) + self.delta(h)
        return self.norm(z)


class ConvolutionalTemporalAdapter(nn.Module):
    """Project and mix neighboring temporal positions into the latent space.

    Parameters
    ----------
    input_dim:
        Number of features at each input position.
    latent_dim:
        Number of features at each output position.
    """

    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()

        self.conv = nn.Conv1d(
            input_dim,
            latent_dim,
            kernel_size=3,
            padding=1,
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.act = nn.GELU()

    def forward(self, h: Tensor) -> Tensor:
        """Project a visual sequence while mixing neighboring positions.

        Parameters
        ----------
        h:
            Tensor of shape `[B, 7, input_dim]`.

        Returns
        -------
        z: Tensor
            Tensor of shape `[B, 7, latent_dim]`.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)

        z = self.conv(h.transpose(1, 2)).transpose(1, 2)
        z = self.norm(z)
        return self.act(z)

class SelfAttentionTemporalAdapter(nn.Module):
    """Adapta 7 posiciones visuales manteniendo su correspondencia."""

    def __init__(self, input_dim: int = 512, latent_dim: int = 256, **_):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.letter_pos = nn.Parameter(
            torch.randn(1, 7, latent_dim) * 0.02
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.ffn_norm = nn.LayerNorm(latent_dim)

    def forward(self, h: Tensor) -> Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)

        z = self.input_proj(h)
        z = z + self.letter_pos
        attended, _ = self.self_attn(
            query=z,
            key=z,
            value=z,
            need_weights=False,
        )
        z = self.attn_norm(z + attended)
        z = self.ffn_norm(z + self.ffn(z))
        return z

class RecurrentAdapter(nn.Module):
    """A stack of 1D convolutions followed by a bidirectional LSTM, used to
    turn a sequence of embeddings into a sequence of contextualized hidden
    states.

    Parameters
    ----------
    input_dim:
        Number of input embedding dimensions. Must equal `latent_dim`.
    latent_dim:
        Number of output embedding dimensions.
    """

    def __init__(self, input_dim: int = 512, latent_dim: int = 512, **_) -> None:
        super().__init__()
        assert input_dim == latent_dim
        self.convolutions = nn.ModuleList()
        for _ in range(2):
            conv = nn.Conv1d(
                latent_dim,
                latent_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
            nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain("relu"))
            conv_layer = nn.Sequential(conv, nn.BatchNorm1d(latent_dim))
            self.convolutions.append(conv_layer)
        self.lstm = nn.LSTM(
            latent_dim,
            int(latent_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm.flatten_parameters()

    def forward(self, h: Tensor) -> Tensor:
        """Pass the input through the RecurrentAdapter.

        Parameters
        ----------
        h:
            Tensor of shape `[B, seq_len, input_dim]`.

        Returns
        -------
        z: Tensor
            Tensor of shape `[B, seq_len, latent_dim]`.
        """
        x = h.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        z, _ = self.lstm(x)
        return z

ADAPTER_REGISTRY = {
    "identity": IdentityAdapter,
    "pointwise": PointwiseTemporalAdapter,
    "respointwise": ResidualPointwiseTemporalAdapter,
    "convolutional": ConvolutionalTemporalAdapter,
    "recurrent": RecurrentAdapter,
    "attn": SelfAttentionTemporalAdapter
}


def build_temporal_adapter(adapter: str = "convolutional", **kwargs):
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter]
    except KeyError as exc:
        options = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown temporal adapter '{adapter}'. Options: {options}") from exc
    return adapter_cls(**kwargs)
