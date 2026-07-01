"""Effective-resolution estimation network.

ResNet50 backbone (ImageNet pre-trained) + average pooling + a fully connected
layer producing a single scalar with no activation (Table 1).  Input images are
expected in ``[0, 1]``; ImageNet normalisation is applied inside ``forward`` so
that adversarial noise can be generated directly on the pixel values.
"""

import torch
import torch.nn as nn
import torchvision


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class EffResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, final_activation=None):
        super().__init__()
        if backbone != 'resnet50':
            raise ValueError(f'unsupported backbone: {backbone!r}')

        weights = 'IMAGENET1K_V2' if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 1)   # avg-pool is already in resnet
        self.net = net

        # final activation: None / 'linear' -> identity
        if final_activation in (None, 'linear', 'none'):
            self.final_activation = nn.Identity()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError(f'unsupported final activation: {final_activation!r}')

        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x, clip=False):
        """``x``: (N, 3, H, W) in [0, 1].  Returns (N,) predicted ratios.

        ``clip=True`` clamps the output to ``[0, 1]`` (used at inference).
        """
        x = (x - self.mean) / self.std
        out = self.net(x)                 # (N, 1)
        out = self.final_activation(out)
        out = out.squeeze(1)
        if clip:
            out = out.clamp(0.0, 1.0)
        return out
