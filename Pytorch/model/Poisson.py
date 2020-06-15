import numpy as np
import pandas as pd
import torch


class Poisson():
    def __init__(self):
        self.model = Poisson.Model()

    def get_parameters(self):
        return self.model.get_parameters()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Poisson.Model, self).__init__()
            self.a = torch.randn(1, 1, requires_grad=True)
            self.b = torch.randn(1, requires_grad=True)

        def forward(self):
            out = torch.abs(self.b)
            return out

        def get_parameters(self):
            return iter((self.a, self.b))

    def fit(self, time, epochs, lr, in_size, no_steps, h, method, log_epoch=10, log=1):
        opt = torch.optim.Adam(self.get_parameters(), lr=lr)

        for e in range(epochs):
            opt.zero_grad()
            z_, integral_ = Poisson.integral_analytical(self, time)
            loss = self.loss(z_, integral_, time.size()[1])
            if e%log_epoch == 0 and log == 1:
                print(f'Epoch: {e}, loss: {loss}')
            loss.backward()
            opt.step()

    def evaluate(self, time):
        z_, integral_ = Poisson.integral_analytical(self, time)
        loss1 = Poisson.loss(z_, integral_, time.size()[1])
        return loss1

    @staticmethod
    def loss(z, integral, nr_of_events):
        ML = torch.log(z)*nr_of_events - integral
        return torch.neg(ML)

    def integral_analytical(self, time):
        def integral_analytical_solve(t):
            z0 = self.model()
            integral = z0*t
            return z0, integral

        return integral_analytical_solve(time[0, -1])


if __name__ == "__main__":

    time = np.abs(100*np.random.rand(100))
    time.sort()
    time[0] = 0
    times = torch.tensor(time).type('torch.FloatTensor').reshape(1, -1, 1)

    in_size = 5
    out_size = 1
    learning_rate = 0.001
    epochs = 50

    model = Poisson()
    model.fit(times, epochs, learning_rate, in_size, 10, None, 'Trapezoid', log_epoch=10)

    loss_on_train = model.evaluate(times, in_size)
    print(f"Loss: {loss_on_train}")
    evaluation_df = pd.read_csv('../../results/baseline_scores.csv')
    evaluation_df.loc[len(evaluation_df)] = ['PoissonAnalytical', 'synthetic', loss_on_train.data.numpy()[0], None]
    evaluation_df.to_csv('../../results/baseline_scores.csv', index=False)


