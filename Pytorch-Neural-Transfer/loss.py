import torch
import torch.nn.functional as F
from torch import nn, optim

class ContentLoss(nn.Module):

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target) # Computed loss is saved as a parameter of this module
        return input

def gram_matrix(input):
    N, C, H, W = input.size()
    features = input.view(N*C, H*W)
    G = torch.mm(features, features.t()) # XX^t
    return G.div(N*C*H*W) # Normalize

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super().__init__()
        self.target_feature = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target_feature)
        return input
