import torch.nn as nn
import torch
from torch import Tensor

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

ADAPTER_REGISTRY = {
    "identity": IdentityAdapter,
    "pointwise": PointwiseTemporalAdapter,
    "respointwise": ResidualPointwiseTemporalAdapter,
    "convolutional": ConvolutionalTemporalAdapter,
    "attn": SelfAttentionTemporalAdapter
}


def build_temporal_adapter(adapter: str = "convolutional", **kwargs):
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter]
    except KeyError as exc:
        options = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown temporal adapter '{adapter}'. Options: {options}") from exc
    return adapter_cls(**kwargs)
