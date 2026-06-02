import torch.nn as nn
import torch.nn.functional as F

from .cornet import CORnet_Z


class ImageToHorizontalFeatures(nn.Module):
    def __init__(self, feature_dim: int = 256, width_steps: int = 24, height_bands: int = 4):
        super().__init__()

        self.feature_dim = feature_dim
        self.width_steps = width_steps
        self.height_bands = height_bands

        self.cornet_z = CORnet_Z()
        self.cornet_z.module.decoder = nn.Identity()
        self.projector = nn.Conv1d(512 * height_bands, feature_dim, kernel_size=1)

    def forward(self, x):
        """Makes an inference.

        Parameters
        ----------
        x: Tensor
            Un tensor de forma [B, C, H, W] donde B es el tamaño del batch, C
            es el número de canales (tipicamente 3), H y W la altura y el ancho
            de las imágenes respectivamente. Se puede omitir B.

        Returns
        ------
        h: Tensor
            Un tensor de forma [B, width_steps, feature_dim]. B es igual a 1 si se
            omitió en x.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.cornet_z(x)
        # Preserve coarse vertical structure before turning width into time.
        if features.size(-2) != self.height_bands:
            features = F.interpolate(
                features,
                size=(self.height_bands, features.size(-1)),
                mode="bilinear",
                align_corners=False,
            )
        B, C, H, W = features.shape
        features = features.reshape(B, C * H, W)
        if features.size(-1) != self.width_steps:
            features = F.interpolate(
                features,
                size=self.width_steps,
                mode="linear",
                align_corners=False,
            )
        features = self.projector(features)
        return features.transpose(1, 2)


class AvgPooledITEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_dim = 512
        self.cornet_z = CORnet_Z()
        self.cornet_z.module.decoder = nn.Identity()
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, 7))

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
