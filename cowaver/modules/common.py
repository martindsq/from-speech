import torch
from torch import nn, Tensor
from torch.nn import functional as F

class Postnet(nn.Module):
    r"""Postnet Module.

    Args:
        n_mels (int): Number of mel bins.
        postnet_embedding_dim (int): Postnet embedding dimension.
        postnet_kernel_size (int): Postnet kernel size.
        postnet_n_convolution (int): Number of postnet convolutions.
    """

    def __init__(
        self,
        n_mels: int = 80,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolution: int = 5,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        for i in range(postnet_n_convolution):
            in_channels = n_mels if i == 0 else postnet_embedding_dim
            out_channels = n_mels if i == (postnet_n_convolution - 1) else postnet_embedding_dim
            padding = int((postnet_kernel_size - 1) / 2)
            init_gain = "linear" if i == (postnet_n_convolution - 1) else "tanh"
            num_features = n_mels if i == (postnet_n_convolution - 1) else postnet_embedding_dim
            conv_layer = nn.Sequential(
                nn.Conv1d(
                   in_channels,
                   out_channels,
                   kernel_size=postnet_kernel_size,
                   stride=1,
                   padding=padding,
                   dilation=1,
                   bias=True,
                ),
                nn.BatchNorm1d(num_features),
            )
            nn.init.xavier_uniform_(
                conv_layer[0].weight,
                gain=nn.init.calculate_gain(init_gain)
            )
            self.convolutions.append(conv_layer)

        self.n_convs = len(self.convolutions)

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through Postnet.

        Args:
            x (Tensor): The input sequence with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).

        Return:
            x (Tensor): Tensor with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
        """

        for i, conv in enumerate(self.convolutions):
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)

        return x

def unpack_batch(batch):
    (x, y), labels = batch
    return x, y.squeeze(1), labels

class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        return self.act(x + y)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return self.act(x + y)
