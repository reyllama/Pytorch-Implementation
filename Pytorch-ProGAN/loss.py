import torch
from apex import amp # Nvidia A Pytorch EXtension / Automatic Mixed Precision
from .utils import to_cuda

def check_overflow(grad):
    cpu_sum = float(grad.float().sum())
    if cpu_sum == float("inf") or cpu_sum == -float("inf") or cpu_sum != cpu_sum:
        return True
    return False

class WGANLoss:

    def __init__(self, discriminator, generator, opt_level):
        self.generator = generator
        self.discriminator = discriminator
        if opt_level == "O0": # Only FP32 (Accuracy First)
            self.wgan_gp_scaler = amp.scaler.LossScaler(1)
        else:
            self.wgan_gp_scaler = amp.scaler.LossScaler(2**14)

    def update_optimizer(self, d_optim, g_optim):
        self.d_optim = d_optim
        self.g_optim = g_optim

    def compute_gradient_penalty(self, real_data, fake_data):
        epsilon_shape = [read_data.shape[0]] + [1]*(real_data.dim() - 1)
        epsilon = torch.rand(epsilon_shape)
