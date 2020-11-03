import os
import numpy as np
import torch
from torch import nn, optim
from torch.util.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import time
import argparse
import sys
from model import *

def get_args(sys_args):

    parser = argparse.ArgumentParser(description="Variational Autoencoder")
    parser.add_argument('--input_dim', default=784, type=int, help="Input dimension of data, default=28*28")
    parser.add_argument('--latent_dim', default=100, type=int, help='Latent code dimension. default=100')
    parser.add_argument('--h_dim', default=256, type=int, help='H code dimension. default=256')
    parse.add_argument('--lr', default=1e-3, type=float, help='learning rate. default 1e-3')

    # To be altered
    parser.add_argument('--datatype', default='Continuous', type=str, help='Whether continuous or categorical input variable')

    args = parser.parse_args(sys_args)

    return args

if __name__ == "__main__":
    args_ = get_args(sys.argv[1:])

model = VAE(args_.input_dim, args_.h_dim, args_.latent_dim, args_.input_dim)
optimizer = optim.Adam(model.parameters(), lr=args_.lr)

def loss_function(x_recon, x, mu, sigma):
    recon_loss = nn.BCELoss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1+torch.log(sigma)-mu.pow(2)-sigma)
    return recon_loss + kl_div
