import torch

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1,1,1) # Dummy Dimensions
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, input):
        return (input - self.mean) / self.std
