import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import time
from torch.util.data import Dataset, DataLoader # Dataset Module Separately Later

class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, h_dim)
        self.layer_mu = nn.Linear(h_dim, z_dim)
        self.layer_sigma = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = F.tanh(self.layer1(x))
        mu, sigma = self.layer_mu(h), self.layer_sigma(h)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(z_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.layer1(z))
        out = F.sigmoid(self.layer2(h))
        return out

class VAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, output_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, h_dim, z_dim)
        self.decoder = Decoder(z_dim, h_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = np.random.normal(0, 1, z_dim) * sigma + mu
        output = self.decoder(z)
        return output, mu, sigma
