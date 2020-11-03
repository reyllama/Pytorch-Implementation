import torch
import torch.nn as nn
from custom_layers import PixelwiseNormalization, WSConv2d, UpSamplingBlock, WSLinear
from utils import get_transition_value
from base import ProgressiveBaseModel

def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
    return nn.Sequential(
        WSConv2d(in_dim, out_dim, kernel_size, padding),
        nn.LeakyReLU(0.2),
        PixelwiseNormalization()
    )

class LatentReshape(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

    def forward(self, x):
        x = x.view(x.shape[0], self.latent_size, 4, 4)
        return x

class Generator(ProgressiveBaseModel):
    def __init__(self, start_channel_dim, img_channels, latent_size):
        super().__init__(start_channel_dim, img_channels)
