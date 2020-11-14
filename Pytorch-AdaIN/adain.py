import torch
from torch import nn
import torch.nn.functional as F

def compute_mean_std(x, eps=1e-8):
    # eps added to prevent division by zero
    assert len(x.size()) == 4, "Invalid Tensor Shape, {}".format(x.size())
    N, C, H, W = x.size()
    x = x.view(N, C, -1)
    mean = torch.mean(x, dim=2).view(N, C, 1, 1)
    std = torch.std(x, dim=2).view(N, C, 1, 1)
    return mean, std

def AdaIN(x, y):
    # x: content image
    # y: style image
    assert x.size()[:2] == y.size()[:2], "Mismatch in dimensions of content and style image"
    size = x.size()
    x_mean, x_std = compute_mean_std(x)
    y_mean, y_std = compute_mean_std(y)
    scaled_x = (x-x_mean.expand(size)) / x_std.expand(size)
    return y_std.expand(size) * scaled_x + y_mean.expand(size)
