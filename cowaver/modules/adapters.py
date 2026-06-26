import torch.nn as nn
from torch import Tensor

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

ADAPTER_REGISTRY = {
    "pointwise": PointwiseTemporalAdapter,
    "convolutional": ConvolutionalTemporalAdapter
}


def build_temporal_adapter(adapter: str = "convolutional", **kwargs):
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter]
    except KeyError as exc:
        options = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown temporal adapter '{adapter}'. Options: {options}") from exc
    return adapter_cls(**kwargs)
