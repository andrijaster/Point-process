import torch


class SelfCorrectingTPP(torch.nn.Module):
    def __init__(self):
        super(SelfCorrectingTPP, self).__init__()
        self.w = torch.randn(1, 1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

    def forward(self, x, t):
        out = torch.clamp(torch.exp(torch.abs(self.w) * t - torch.abs(self.b)*x.size()[0]), min=0.001, max=1)
        return out

    def parameters(self):
        return iter((self.w, self.b))
