import torch


class IntereventRegressorTPP(torch.nn.Module):
    def __init__(self, input_size=10, output_size=1):
        super(IntereventRegressorTPP, self).__init__()
        self.a = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, 10, requires_grad=True)

    def forward(self, x, t):
        interevents = torch.abs(t-x)
        out = torch.abs(self.a) + torch.sum(torch.abs(self.b)*interevents)
        return out

    def parameters(self):
        return iter((self.a, self.b))
