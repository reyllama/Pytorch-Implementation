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
        epsilon = torch.rand(epsilon_shape) # Random Initialization
        epsilon = to_cuda(epsilon)
        epsilon = epsilon.to(fake_data.dtype) # Align dtypes
        real_data = real_data.to(fake_data.dtype)
        x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach() # Interpolate between real/fake data
        x_hat.requires_grad = True
        logits = self.discriminator(x_hat)
        logits = logits.sum() * self.wgan_gp_scaler.loss_scale() # AMP
        grad = torch.autograd.grad( # Computes and returns the sum of gradients of outputs w.r.t. the inputs
            outputs = logits, # refers to dy part in leibniz dy/dx notation
            inputs = x_hat, # dx part (partial differentiation)
            grad_outputs = torch.ones(logits.shape).to(fake_data.dtype).to(fake_data.device), # The "vector" in Jacobian-vector product (weight-ish?)
            create_graph = True # This option allows to compute higher order derivative products
        )[0]
        grad = grad.view(x_hat.shape[0], -1) # Unroll
        if check_overflow(grad):
            print("Exploding gradient in gradient penalty calculation")
            self.wgan_gp_scaler._loss_scale /= 2 # Scale down loss (* If check_overflow is True, it means that current grad is Inf. Does dividing by half matters..?)
            print("Scaling down loss to: ", self.wgan_gp_scaler._loss_scale)
            return None
        grad = grad / self.wgan_gp_scaler.loss_scale()
        gradient_pen = ((grad.norm(p=2, dim=1)-1)**2) # Regularize the gradient so that its l2 norm is close to 1
        to_backward = gradient_pen.sum() * 10
        with amp.scale_loss(to_backward, self.d_optimizer, loss_id=1) as scaled_loss:
            scaled_loss.backward(retain_graph=True) # retain_graph: prevent internal buffers from being deleted. Used when we want to calculate grad once again later on.
        return gradient_pen.detach().mean()

    def step(self, real_data):
        # Train Discriminator
        z = self.generator.generate_latent_variable(real_data.shape[0]) # N
        with torch.no_grad():
            fake_data = self.generator(z)
        real_scores = self.discriminator(real_data)
        fake_scores = self.discriminator(fake_data)
        # Wasserstein-1 Distance
        wasserstein_distance = (real_scores - fake_scores).squeeze() # Returns a tensor with all the dimensions of input of size 1 removed. 1 x 1 x H x W => H x W
        # Epsilon Penalty
        epsilon_penalty = (real_scores ** 2).squeeze()

        self.d_optim.zero_grad()
        gradient_pen = self.compute_gradient_penalty(real_data, fake_data) # Deviation of the l2 norm of gradient vector from 1
        if gradient_pen is None:
            return None

        to_backward1 = (-wasserstein_distance).sum()
        with amp.scale_loss(to_backward1, self.d_optim, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)

        to_backward3 = epsilon_penalty.sum() * 0.001
        with amp.scale_loss(to_backward3, self.d_optim, loss_id=2) as scaled_loss:
            scaled_loss.backward()

        self.d_optim.step()
        # Reason why we retained graphs
        z = self.generator.generate_latent_variable(real_data.shape[0])
        fake_data = self.generator(z)
        # Forward G
        for p in self.discriminator.paramters():
            p.requires_grad=False # Bind discriminator
        fake_scores = self.discriminator(fake_data) # Discriminator outputs small number for (what if believes to be) fake, and big number for real
        G_loss = (-fake_scores).sum() # The higher fake scores, the better for generator

        self.g_optim.zero_grad()
        with amp.scale_loss(G_loss, self.g_optim, loss_id=3) as scaled_loss:
            scaled_loss.backward()
        self.g_optim.step()
        for p in self.discriminator.parameters():
            p.requires_grad = True # Back to Training discriminator
        return wasserstein_distance.mean().detach(), gradient_pen.mean().detach(), real_scores.mean().detach(), fake_scores.mean().detach(), epsilon_penalty.mean().detach()

        """
        =====================================================
        LOSS COMPONENTS DICTIONARY
        =====================================================
        wasserstein_distance : real_scores - fake_scores
        gradient_pen : the magnitude the l2 norm of `grad` deviates from 1. The grad is partial derivative of discriminator output w.r.t. interpolated real/fake input
        real_scores : discriminator output w.r.t. real data
        fake_scores : discriminator output w.r.t. fake data
        epsilon_penalty : l2 norm of real_scores. Suppose it shouldn't be too high, 'cause that way it would be similar to exploding gradient? Cause in WGAN, we do not use sigmoid or the like.
        """
