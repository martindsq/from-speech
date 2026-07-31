import torch
import torch.nn as nn
import torch.nn.functional as F

def cout(x, layer):
    """Unnecessarily complicated but complete way to calculate the output depth,
    height and width size for a Conv2D layer

    Arguments
    ---------
    x: tuple
        Input size (depth, height, width)
    layer: nn.Conv2d
        The Conv2D layer

    Returns
    -------
    out: tuple
        Tuple of out-depth/out-height and out-width
        Output shape as given in [Ref]
        Ref:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    #assert isinstance(layer, nn.Conv2d)
    assert isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d))

    padding = layer.padding
    p = padding if isinstance(padding, tuple) else (padding,)

    kernel_size = layer.kernel_size
    k = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)

    dilation = layer.dilation
    d = dilation if isinstance(dilation, tuple) else (dilation,)

    stride = layer.stride
    s = stride if isinstance(stride, tuple) else (stride,)
    
    in_depth, in_height, in_width = x
    out_depth = layer.out_channels
    out_height = 1 + (in_height + 2 * p[0] - (k[0] - 1) * d[0] - 1) // s[0]
    out_width = 1 + (in_width + 2 * p[-1] - (k[-1] - 1) * d[-1] - 1) // s[-1]
    return (out_depth, out_height, out_width)
    
class BiasLayer(nn.Module):
    """Bias Layer"""

    def __init__(self, shape):
        """Initialise parameters of bias layer

        Arguments
        ---------
        shape: tuple
            Requisite shape of bias layer
        """
        super(BiasLayer, self).__init__()
        init_bias = torch.zeros(shape)
        self.bias = nn.Parameter(init_bias, requires_grad=True)

    def forward(self, x):
        """
        Forward pass

        Arguments
        ---------
        x: torch.tensor
            Input features

        Returns
        -------
        out: torch.tensor
            Output of bias layer
        """
        return x + self.bias

class ConvEncoder(nn.Module):
    """A convolutional Encoder"""
  
    def __init__(self, x_dim, h_dim, n_filters=32, filter_size=5):
        """Initialize parameters of ConvEncoder

        Arguments
        ---------
        x_dim: tuple
            Input dimensions (channels,height, widths)
        h_dim: int
            Hidden dimension, bottleneck dimension, K
        n_filters: int
            Number of filters (number of output channels)
        filter_size: int
            Kernel size
        """
        super().__init__()
        channels, height, widths = x_dim

        # Encoder input bias layer
        self.bias = BiasLayer(x_dim)

        # First encoder conv2d layer
        self.conv_1 = nn.Conv2d(channels, n_filters, filter_size)

        # Output shape of the first encoder conv2d layer given x_dim input
        conv_1_shape = cout(x_dim, self.conv_1)

        # Second encoder conv2d layer
        self.conv_2 = nn.Conv2d(n_filters, n_filters, filter_size)

        # Output shape of the second encoder conv2d layer given conv_1_shape
        # input
        conv_2_shape = cout(conv_1_shape, self.conv_2)

        # The bottleneck is a dense layer, therefore we need a flattenning layer
        self.flatten = nn.Flatten()

        # Conv output shape is (depth, height, width), so the flatten size is:
        flat_after_conv = conv_2_shape[0] * conv_2_shape[1] * conv_2_shape[2]

        # Encoder Linear layer
        self.lin = nn.Linear(flat_after_conv, h_dim)

    def forward(self, x):
        """Encoder

        Arguments
        ---------
        x: torch.tensor
           Input features

        Returns
        -------
        x: torch.tensor
           Encoded output
        """
        s = self.bias(x)
        s = F.relu(self.conv_1(s))
        s = F.relu(self.conv_2(s))
        s = self.flatten(s)
        h = self.lin(s)
        return h

class ConvDecoder(nn.Module):
    """A convolutional Decoder"""
  
    def __init__(self, x_dim, h_dim, n_filters=32, filter_size=5):
        """Initialize parameters of ConvDecoder

        Arguments
        ---------
        x_dim: tuple
            Input dimensions (channels,height, widths)
        h_dim: int
            Hidden dimension, bottleneck dimension, K
        n_filters: int
            Number of filters (number of output channels)
        filter_size: int
            Kernel size
        """
        super().__init__()
        channels, height, widths = x_dim
        
        # Decoder output bias layer
        self.bias = BiasLayer(x_dim)
        
        # Second "deconvolution" layer
        self.deconv_2 = nn.ConvTranspose2d(n_filters, channels, filter_size)

        # First "deconvolution" layer
        self.deconv_1 = nn.ConvTranspose2d(n_filters, n_filters, filter_size)

        # Output shape of the first encoder conv2d layer given x_dim input
        conv_1_shape = cout(x_dim, self.deconv_2)

        # Output shape of the second encoder conv2d layer given conv_1_shape
        # input
        conv_2_shape = cout(conv_1_shape, self.deconv_1)

        # Unflatten data to (depth, height, width) shape
        self.unflatten = nn.Unflatten(dim=-1, unflattened_size=conv_2_shape)

        # Conv output shape is (depth, height, width), so the flatten size is
        flat_after_conv = conv_2_shape[0] * conv_2_shape[1] * conv_2_shape[2]
        
        # Decoder Linear layer
        self.lin = nn.Linear(h_dim, flat_after_conv)

    def forward(self, h):
        """Decoder

        Arguments
        ---------
        h: torch.tensor
           Encoded output

        Returns
        -------
        x_prime: torch.tensor
           Decoded output
        """
        s = F.relu(self.lin(h))
        s = self.unflatten(s)
        s = F.relu(self.deconv_1(s))
        s = self.deconv_2(s)
        x_prime = self.bias(s)
        return x_prime

class ConvAutoEncoder(nn.Module):
    """A Convolutional AutoEncoder"""

    def __init__(self, x_dim, h_dim, n_filters=32, filter_size=5):
        """Initialize parameters of ConvAutoEncoder

        Arguments
        ---------
        x_dim: tuple
            Input dimensions (channels, height, widths)
        h_dim: int
            Hidden dimension, bottleneck dimension, K
        n_filters: int
            Number of filters (number of output channels)
        filter_size: int
            Kernel size
        """
        super().__init__()
        self.encoder = ConvEncoder(x_dim, h_dim, n_filters, filter_size)
        self.decoder = ConvDecoder(x_dim, h_dim, n_filters, filter_size)

    def forward(self, x):
        """Forward pass

        Arguments
        ---------
        x: torch.tensor
            Input features

        Returns
        -------
        out: torch.tensor
            Decoded output
        """
        return self.decoder(self.encoder(x))
