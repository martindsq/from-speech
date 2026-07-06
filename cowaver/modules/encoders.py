import torch.nn as nn
import torch.nn.functional as F

from .cornet import CORnet_Z

class AvgPooledITEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_dim = 512
        self.cornet_z = CORnet_Z()
        self.cornet_z.module.decoder = nn.Identity()
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, 7))

        for param in self.cornet_z.parameters():
            param.requires_grad = False

        for param in self.cornet_z.module.IT.parameters():
            param.requires_grad = False


    def forward(self, x):
        """Average-pool the native CORnet IT map into a horizontal sequence.

        Returns
        -------
        h: Tensor
            Tensor de forma [B, 7, 512].
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.cornet_z(x)
        features = self.vertical_pool(features).squeeze(2)
        return features.transpose(1, 2)
