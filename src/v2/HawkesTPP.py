import torch


class HawkesTPP(torch.nn.Module):
    def __init__(self):
        super(HawkesTPP, self).__init__()
        self.alpha = torch.randn(1, 1, requires_grad=True)
        self.mu = torch.randn(1, requires_grad=True)

    def forward(self, x, t):
        variable_part = torch.sum(torch.exp(-x)).reshape(1, -1)
        out = torch.abs(self.mu) + torch.abs(self.alpha) * variable_part
        return out

    def parameters(self):
        return iter((self.alpha, self.mu))
