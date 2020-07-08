import numpy as np
import pandas as pd
import torch
import math
from scipy.special import p_roots


class HawkesSumGaussians():
    def __init__(self):
        self.model = HawkesSumGaussians.Model()

    def get_parameters(self):
        return self.model.get_parameters()

    class Model(torch.nn.Module):
        def __init__(self):
            super(HawkesSumGaussians.Model, self).__init__()
            self.mu = torch.randn(1, 1, requires_grad=True)
            self.sigma = torch.randn(1, requires_grad=True)

        def forward(self, x, t):
            variable_part = torch.sum(torch.exp(-x/(2*self.sigma**2))).reshape(1, -1)
            out = torch.abs(self.mu) + (2*math.pi*torch.abs(self.sigma)**2)**(-1) * variable_part
            return out

        def get_parameters(self):
            return iter((self.mu, self.sigma))

    def fit(self, time, epochs, lr, in_size, no_steps, h, method, log_epoch=10, log=1):
        opt = torch.optim.Adam(self.get_parameters(), lr=lr)

        for e in range(epochs):
            opt.zero_grad()

            z_, integral_ = HawkesSumGaussians.integral(self, time, in_size, no_steps=no_steps, h=h, method=method)
            loss = self.loss(z_, integral_)
            if e%log_epoch == 0 and log == 1:
                print(f'Epoch: {e}, loss: {loss}')
            loss.backward()
            opt.step()

    def evaluate(self, time, in_size, no_steps = 10, h = None, atribute = None, method = "Euler"):
        z_, integral_ = HawkesSumGaussians.integral(self, time, in_size, no_steps, h = h, method = method)
        loss1 = HawkesSumGaussians.loss(z_, integral_)
        return loss1

    @staticmethod
    def loss(z, integral):
        ML = torch.sum(torch.log(z)) - integral
        return torch.neg(ML)

    def integral(self, time, in_size, no_steps, h = None , atribute = None, method = "Euler"):

        def integral_solve(z0, time_to_t0, t0, t1, atribute, no_steps = 10, h = None, method ="Euler"):
            if no_steps is not None:
                h_max = (t1 - t0)/no_steps
            elif h is not None:
                no_steps = math.ceil((t1 - t0)/h)
                h_max = (t1 - t0)/no_steps

            integral = 0
            t = t0

            def Gaussian_quadrature(n, lower_l, upper_l):
                m = (upper_l-lower_l)/2
                c = (upper_l+lower_l)/2
                [x,w] = p_roots(n+1)
                weights = m*w
                time_train_integral = m*x+c
                return time_train_integral, weights


            if method =="Euler":
                for _ in range(no_steps):
                    integral += z0*h_max
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(time_to_t0, t)

            if method =="Implicit_Euler":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(time_to_t0, t)
                    integral += z0*h_max

            if method == "Trapezoid":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z1 = self.model(time_to_t0, t)
                    integral += (z0+z1)*0.5*h_max
                    z0 = z1

            if method == "Simpsons":
                z = []
                z.append(z0)
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(time_to_t0, t)
                    z.append(z0)
                integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])

            if method == "Gaussian_Q":
                time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
                integral = 0
                for i in range(time_integral.shape[0]):
                    t = time_integral[i]
                    atribute = atribute + h_max
                    z0 = self.model(time_to_t0, t)
                    integral += weights[i]*z0
                atribute = atribute + t1
                z0 = self.model(time_to_t0 + 1, t1)

            return integral, z0, atribute

        integral_ = 0
        time_len = time.size(1)
        if atribute is None:
            atribute = torch.ones(in_size).reshape(1,-1)*0
        z = torch.zeros(time.shape)
        z0 = self.model(time[0, :0], time[0, 0])
        z[:,0] = z0
        for i in range(time_len-1):
            integral_interval, z_, atribute = integral_solve(z0, time[0, :i], time[0, i], time[0, i+1], atribute, no_steps = no_steps, h = h, method = method)
            integral_ += integral_interval
            atribute[:,1:] = atribute[:,:-1].clone()
            atribute[:,0] = 0
            z[:,i+1] = z_
        return z, integral_


if __name__ == "__main__":

    # ny_train_set = pd.read_csv('../../data/nytaxi/taxi_feature_matrix_train.csv')
    # ny_train_times = np.array(ny_train_set.query('target == 1').time)
    # train_times = ny_train_times - ny_train_times[0]
    # times = torch.tensor(train_times).type('torch.FloatTensor').reshape(1, -1, 1)

    time = np.abs(100*np.random.rand(100))
    time.sort()
    time[0] = 0
    times = torch.tensor(time).type('torch.FloatTensor').reshape(1, -1, 1)

    in_size = 5
    out_size = 1
    learning_rate = 0.001
    epochs = 50

    model = HawkesSumGaussians()
    model.fit(times, epochs, learning_rate, in_size, 10, None, 'Trapezoid', log_epoch=10)

    loss_on_train = model.evaluate(times, in_size)
    print(f"Loss: {loss_on_train}")
    evaluation_df = pd.read_csv('../../results/baseline_scores.csv')
    evaluation_df.loc[len(evaluation_df)] = ['HawkesSumGaussians', 'synthetic', loss_on_train.data.numpy()[0][0], None]
    evaluation_df.to_csv('../../results/baseline_scores.csv', index=False)


