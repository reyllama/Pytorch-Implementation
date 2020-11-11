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
        super().__init__(start_channel_dim, img_channels) # Execute __init__ method in parent (ProgressiveBaseModel)
        # Transition Blocks
        self.latent_size = latent_size
        self.to_rgb_new = WSConv2d(start_channel_dim, self.image_channels, 1, 0) # 1x1 convolution to switch to 3 channel image (RGB)
        self.to_rgb_old = WSConv2d(start_channel_dim, self.image_channels, 1, 0)

        # Why double Sequential Parentheses?
        self.core_blocks = nn.Sequential(
            nn.Sequential(
                WSLinear(self.latent_size, self.latent_size*4*4),
                LatentReshape(self.latent_size), # If we need reshape operation in the middle, we make that as a class inheriting nn.Module. Lesson learned.
                conv_bn_relu(self.latent_size, self.latent_size, 3, 1) # Inputs nn.Sequential as a whole here.
            )
        )

        self.new_blocks = nn.Sequential()
        self.upsampling = UpSamplingBlock()

    def extend(self):
        output_dim = self.transition_channels[self.transition_step] # Pre-assigned resolution schedule. Keep track of transition_step.
        # Downsampling Module
        if self.transition_step == 0:
            core_blocks = nn.Sequential(
                *self.core_blocks.children(), # WSLinear(expand latent), LatentReshape, conv_bn_relu
                UpsamplingBlock()
            )
            self.core_blocks = core_blocks
        else: # If transition_step > 0:
            self.core_blocks = nn.Sequential(
                *self.core_blocks.children(),
                # Add New Block if step >= 1
                self.new_blocks,
                UpSamplingBlock()
            )

        # Old: previous one. Updates are late
        self.to_rgb_old = self.to_rgb_new
        # New: Current to-rgb state
        self.to_rgb_new = WSConv2d(output_dim, self.image_channels, 1, 0)

        self.new_blocks = nn.Sequential(
            conv_bn_relu(self.prev_channel_extension, output_dim, 3, 1),
            conv_bn_relu(output_dim, output_dim, 3, 1)
        )
        super().extend() # execute "extend" method from parent(ProgressiveBaseModel)

    def new_paramters(self):
        new_parameters = list(self.new_blocks.parameters()) + list(self.to_rgb_new.parameters())
        return new_parameters

    def forward(self, z): # Takes in latent vector as input
        x = self.core_blocks(z) # core_blocks: the main block that is enlarged in each training step
        if self.transition_step == 0: # No to_rgb_old yet since first time
            x = self.to_rgb_new(x)
            return x
        # If training step >= 1
        x_old = self.to_rgb_old(x)
        x_new = self.new_blocks(x)
        x_new = self.to_rgb_new(x_new)

        x = get_transition_value(x_old, x_new, self.transition_value) # Linear Interpolation using torch.lerp / I guess this works as Fade In.
        return x

    def device(self):
        return next(self.parameters()).device

    def generate_latent_variable(self, batch_size):
        return torch.randn(batch_size, self.latent_size, device=self.device())
