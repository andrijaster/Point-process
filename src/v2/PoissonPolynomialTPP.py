import torch


class PoissonPolynomialTPP(torch.nn.Module):
    def __init__(self):
        super(PoissonPolynomialTPP, self).__init__()
        self.a = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        self.c = torch.randn(1, requires_grad=True)

    def forward(self, x, t):
        out = torch.abs(self.a) + torch.abs(self.b)*t + torch.abs(self.c)*t**2
        return out

    def parameters(self):
        return iter((self.a, self.b, self.c))
