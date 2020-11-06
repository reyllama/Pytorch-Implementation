import torch
import numpy as np
from torch import nn

class WSConv2d(nn.Module):
    """
    Weight-scaling conv layer. Initialize with N(0, scale).
    Then, it multiplies the scale every forward pass.
    """
    def __init__(self, inCh, outCh, kernelSize, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=1, padding=padding)

        # New bias to use after weight scaling
        bias = self.conv.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1, 1))
        self.conv.bias = None

        # Calculate weight scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out number of op filters (What's op filters?)
        self.wtScale = gain/np.sqrt(fanIn) # gain = sqrt(2)

        # initialize
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))

    def forward(self, x):
        return self.conv(x*self.wtScale) + self.bias

    def __repr__(self):
        return self.__class__.__name__ + self.name

class WSLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = self.linear.bias
        self.linear.bias = None # Cancel bias once saved
        fanIn = in_dim
        self.wtScale = np.sqrt(2) / np.sqrt(fanIn) # gain = sqrt(2)

        nn.init.normal_(self.linear.weight) # Inplace operation for initialization - Unit Gaussian since no additional argument passed in
        nn.init.constant_(self.bias, val=0) # No bias term on initialization

    def forward(self, x):
        return self.linear(x * self.wtScale) + self.bias

class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super().__inint__()

    def forward(self, x):
        factor = ((x**2).mean(dim=1, keepdim=True) + 1e-8)**0.5 # Note that dimension: B x C x H x W (Here, it's channel-way normalization)
        return x / factor

class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2) # Default: Nearest neighbor / It can be both up or downsampling depending on scale_factor

class MinibatchStdLayer(nn.Module):
    """
    layer for minibatch standard deviation featuremap
    Literature adds replicated single value, the average channel-wise standard deviation as a featuremap and
    feeds to discriminator towards the end, to make the variability of original and generated images as similar as possible
    """
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        size = x.size()
        subGroupSize = min(size[0], self.group_size) # Batchsize
        if size[0] % subGroupSize != 0: # If not multiple
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1: # If it was multiple in the previous if statement
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            y = torch.var(y, 1) # Get channel-wise variance
            y = torch.sqrt(y + 1e-8) # Get standard deviation
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1) # Obtain the single value
            y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1,1,size[2], size[3])) # C = 1, H, W back to original
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        return torch.cat([x,y], dim=1) # Adds this as additional featuremap to original input
