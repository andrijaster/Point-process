import torch
import math


class HawkesSumGaussianTPP(torch.nn.Module):
    def __init__(self):
        super(HawkesSumGaussianTPP, self).__init__()
        self.mu = torch.randn(1, 1, requires_grad=True)
        self.sigma = torch.randn(1, requires_grad=True)

    def forward(self, x, t):
        variable_part = torch.sum(torch.exp(-x/(2*self.sigma**2))).reshape(1, -1)
        out = torch.abs(self.mu) + (2*math.pi*torch.abs(self.sigma)**2)**(-1) * variable_part
        return out

    def parameters(self):
        return iter((self.mu, self.sigma))
