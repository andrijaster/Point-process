import torch


class PoissonPolynomialFirstOrderTPP(torch.nn.Module):

    def __init__(self):
        super(PoissonPolynomialFirstOrderTPP, self).__init__()
        self.a = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def forward(self, x, t):
        out = torch.abs(self.a) + torch.abs(self.b)*t
        return out

    def parameters(self):
        return iter((self.a, self.b))
