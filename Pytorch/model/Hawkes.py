import numpy as np
import pandas as pd
import torch


class Hawkes():
    def __init__(self, in_size, out_size):
        self.model = Hawkes.Model(in_size, out_size)

    class Model(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(Hawkes.Model, self).__init__()
            self.linear = torch.nn.Linear(1, output_size)

        def forward(self, x, t):
            variable_part = torch.sum(torch.exp(-x)).reshape(1, -1)
            out = self.linear(variable_part)
            return out

    @staticmethod
    def loss(z, integral):
        ML = torch.sum(torch.log(z)) - integral
        return torch.neg(ML)

    def integral(self, time, in_size, no_steps, h = None , atribute = None, method = "Euler"):

        def integral_solve(z0, t0, t1, atribute, no_steps = 10, h = None, method = "Euler"):
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
                    z0 = self.model(atribute,t)

            if method =="Implicit_Euler":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(atribute,t)
                    integral += z0*h_max

            if method == "Trapezoid":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z1 = self.model(atribute,t)
                    integral += (z0+z1)*0.5*h_max
                    z0 = z1

            if method == "Simpsons":
                z = []
                z.append(z0)
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(atribute,t)
                    z.append(z0)
                integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])

            if method == "Gaussian_Q":
                time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
                integral = 0
                for i in range(time_integral.shape[0]):
                    t = time_integral[i]
                    atribute = atribute + h_max
                    z0 = self.model(atribute,t)
                    integral += weights[i]*z0
                atribute = atribute_0 + t1
                z0 = self.model(atribute,t1)

            return integral, z0, atribute

        integral_ = 0
        time_len = time.size(1)
        if atribute is None:
            atribute = torch.ones(in_size).reshape(1,-1)*0
        z = torch.zeros(time.shape)
        z0 = self.model(atribute,time[0,0])
        z[:,0] = z0
        for i in range(time_len-1):
            integral_interval, z_, atribute = integral_solve(z0, time[0,i], time[0,i+1], atribute, no_steps = no_steps, h = h, method = method)
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
    # X_input = np.exp(-(train_times - np.concatenate((np.zeros(1), times[:-1]))))

    time = np.abs(100*np.random.rand(100))
    time.sort()
    time[0] = 0
    times = torch.tensor(time).type('torch.FloatTensor').reshape(1,-1,1)

    in_size = 5
    out_size = 1
    learning_rate = 0.001
    epochs = 100

    model = Hawkes(in_size, out_size)
    optimizer_1 = torch.optim.Adam(model.model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer_1.zero_grad()
        z_, integral_ = model.integral(time=times, in_size=in_size,
                                                  no_steps=3, h=None, method="Euler")

        loss = Hawkes.loss(z_, integral_)
        loss.backward()
        optimizer_1.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))




