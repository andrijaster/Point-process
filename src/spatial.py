import torch
from torch import nn
import math

import matplotlib.pyplot as plt
import torch
from scipy.special import p_roots
import numpy as np
import pickle
from pathlib import Path


class FCNSpatialProcess(nn.Module):

    def __init__(self, input_size, output_size, dropout=0, num_layers=2, step=2):
        super(FCNSpatialProcess, self).__init__()
        input_size = input_size*output_size

        lst = nn.ModuleList()
        # Fully connected layer
        out_size = input_size
        for i in range(num_layers):
            inp_size = out_size
            out_size = int(inp_size//step)
            if i == num_layers-1:
                block = nn.Linear(inp_size, output_size)
            else:
                block = nn.Sequential(nn.Linear(inp_size, out_size),
                                      nn.ReLU())
            lst.append(block)

        self.fc = nn.Sequential(*lst)

    def forward(self, x, t):
        t = t.to(torch.float32)
        x = torch.cat((x, torch.cat((x.size(0)*[t])).reshape(-1, 1)), dim=1).reshape(-1, 1)
        print(x.size())
        print(x)
        o = self.fc(x)
        out = torch.exp(o)
        return out


def integral(model, time, in_size, no_steps, h=None, atribute=None, method="Euler"):

    def integral_solve(z0, time_to_t0, t0, t1, atribute, no_steps=10, h=None, method="Euler"):
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

        if method == "Euler":
            for _ in range(no_steps):
                integral += z0*h_max
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)

        if method == "Implicit_Euler":
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)
                integral += z0*h_max

        if method == "Trapezoid":
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z1 = model(atribute, t)
                integral += (z0+z1)*0.5*h_max
                z0 = z1

        if method == "Simpsons":
            z = []
            z.append(z0)
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)
                z.append(z0)
            integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])

        if method == "Gaussian_Q":
            time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
            integral = 0
            for i in range(time_integral.shape[0]):
                t = time_integral[i]
                atribute = atribute + h_max
                z0 = model(atribute, t)
                integral += weights[i]*z0
            atribute = atribute + t1
            z0 = model(atribute, t1)

        return integral, z0, atribute

    integral_ = 0
    time_len = time.size(1)
    out_size = train_time.size(0)
    if atribute is None:
        atribute = torch.zeros(out_size, in_size)
    z = torch.zeros(time.shape)
    z0 = model(atribute, time[0, 0])
    z[:, 0, 0] = z0
    for i in range(time_len-1):
        integral_interval, z_, atribute = integral_solve(z0, time[:, :i, 0], time[:, i, 0], time[:, i+1, 0],
                                                         atribute, no_steps=no_steps, h=h, method=method)
        integral_ += integral_interval
        atribute[:, 1:] = atribute[:, :-1].clone()
        atribute[:, 0] = 0
        z[:, i+1] = z_
    return z, integral_


def loss(z, integral):
    ml = torch.sum(torch.log(z)) - integral
    return torch.neg(ml)


def fit(model, train_time, test_time, in_size, lr, method="Euler", no_steps=10, h=None, no_epoch=100, log=1,
        log_epoch=10, figpath='model.png'):
    train_losses, test_losses = [], []

    init_loss = evaluate(model, train_time, in_size, no_steps=no_steps, h=h, method='Trapezoid').data.numpy().flatten()[0]
    if np.isnan(init_loss) or np.isinf(init_loss):
        print(f"Init loss: {init_loss}. It needs to reinitialize the model.")
        return None

    " Training "
    model.train()

    list_1 = list(model.parameters())
    optimizer_1 = torch.optim.Adam(list_1, lr=lr)

    for e in range(no_epoch):
        optimizer_1.zero_grad()
        if method == "Analytical":
            z_, integral_ = model.integral_analytical(train_time)
        else:
            z_, integral_ = integral(model, train_time, in_size, no_steps=no_steps, h=h, method=method)
        train_loss = loss(z_, integral_)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer_1.step()

        train_losses.append(train_loss.data.numpy().flatten()[0])
        test_loss = evaluate(model, test_time, in_size, no_steps=no_steps, h=h, method='Trapezoid')
        test_losses.append(test_loss.data.numpy().flatten()[0])
        model.train()

        if e % log_epoch == 0 and log == 1:
            print(f"Epoch: {e}, train loss: {train_loss.data.numpy().flatten()[0]}, "
                  f"test loss: {test_loss.data.numpy().flatten()[0]}")
            plt.clf()
            plt.plot(train_losses, color='skyblue', linewidth=2, label='train')
            plt.plot(test_losses, color='darkgreen', linewidth=2, linestyle='dashed', label="test")
            plt.legend(loc="upper right")
            plt.show()

    # always draw final img
    plt.clf()
    plt.plot(train_losses, color='skyblue', linewidth=2, label='train')
    plt.plot(test_losses, color='darkgreen', linewidth=2, linestyle='dashed', label="test")
    plt.legend(loc="upper right")
    plt.savefig(figpath)
    plt.show()

    return model


def evaluate(model, time, in_size, no_steps=10, h=None, method="Euler"):
    model.eval()
    z_, integral_ = integral(model, time, in_size, no_steps, h=h, method=method)
    loss1 = loss(z_, integral_)
    return loss1


if __name__ == "__main__":
    project_dir = str(Path(__file__).parent.parent)

    time = np.abs(100*np.random.rand(3, 100))
    time.sort()
    in_size = 5
    out_size = time.shape[0]
    time[:, 0] = 0

    train_time, test_time = time[:, :80], time[:, 80:]
    train_time = torch.tensor(train_time).type('torch.FloatTensor').reshape(3, -1, 1)
    test_time = torch.tensor(test_time).type('torch.FloatTensor').reshape(3, -1, 1)

    model = FCNSpatialProcess(in_size+1, out_size)
    model_name = "fcn_spatial_test"

    model = fit(model, train_time, test_time, in_size, lr=0.01,
                             no_epoch=100, no_steps=10, method='Trapezoid', log_epoch=10,
                             figpath=f"{project_dir}/img{model_name}.png")
    if model:
        loss_on_train = evaluate(model, train_time, in_size, method='Trapezoid')
        loss_on_test = evaluate(model, test_time, in_size, method='Trapezoid')
        print(f"Model: {model_name}. Loss on train: {str(loss_on_train.data.numpy().flatten()[0])}, "
              f"loss on test: {str(loss_on_test.data.numpy().flatten()[0])}")

        model_filepath = f"{project_dir}/models/{model_name}.torch"
        pickle.dump(model, open(model_filepath, 'wb'))

