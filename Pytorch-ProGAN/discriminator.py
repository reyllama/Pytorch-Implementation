import torch.nn as nn
from .custom_layers import WSCONv2d, WSLinear, MinibatchStdLayer
from .utils import get_transition_value
from .base import ProgressiveBaseModel

"""
The discriminator in progressive GAN is the mirror image of the generator.
Mostly follows DCGAN structure, with certain custom layers.
Layers are added progressively, just like generators.
"""

def conv_module_bn(dim_in, dim_out, kernel_size, padding):
    return nn.Sequential(
        WSCONv2d(dim_in, dim_out, kernel_size, padding),
        nn.LeakyReLU(negative_slope=0.2)
    )

class Discriminator(ProgressiveBaseModel):

    def __init__(self, image_channels, start_channel_dim):
        super().__init__(start_channel_dim, image_channels)

        self.from_rgb_new = conv_module_bn(image_channels, start_channel_dim, 1, 0) # 1 by 1 convolution to take in RGB image
        self.from_rgb_old = conv_module_bn(image_channels, start_channel_dim, 1, 0) # 1 by 1 convolution

        self.new_block = nn.Sequential()
        self.core_block = nn.Sequential(
            nn.Sequential(
                MinibatchStdLayer(), # Adds feature statistic as a feature map
                conv_module_bn(start_channel_dim+1, start_channel_dim, 3, 1), # additional channel from MinibatchStdLayer
                conv_module_bn(start_channel_dim, start_channel_dim, 4, 0)
            )
        )
        self.output_layer = WSLinear(start_channel_dim, 1) # Outputs True of False

    def extend(self):
        input_dim = self.transition_channels[self.transition_step]
        output_dim = self.prev_channel_extension # Reduce Spatial Resolution
        if self.transition_step != 0:
            self.core_block = nn.Sequential(
                self.new_block,
                *self.core_block.children()
            )
        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2,2]),
            self.from_rgb_new
        )
        self.from_rgb_new = conv_module_bn(self.image_channels, input_dim, 1, 0) # 1 by 1 convolution
        new_block = nn.Sequential( # Additional blocks to be added as transition steps proceed
            conv_module_bn(input_dim, input_dim, 3, 1),
            conv_module_bn(input_dim, output_dim, 3, 1),
            nn.AgePool2d([2,2]) # Note Pooling comes in for discriminator / Not for generator cause it causes blurry image (checkerboard effect)
        )
        self.new_block = new_block
        super().extend()

    def forward(self, x):
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        x_new = self.new_block(x_new) # Nothing at first, and sequentially added
        x = get_transition_value(x_old, x_new, self.transition_value)
        x = self.core_block(x)
        x = x.view(x.shape[0], -1) # Flatten
        x = self.output_layer(x) # Linear Layer that outputs 0 / 1
        return x
