# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:45:27 2019

@author: Andri
"""

import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import p_roots
from torch import nn
from pathlib import Path
import pandas as pd

class FCN_point_process_all():

    def __init__(self, in_size, out_size, drop=0, num_layers = 2,
                 step = 2):
        self.model = FCN_point_process_all.Model(in_size, out_size,
                           dropout = drop,
                           num_layers = num_layers, step = step)

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

            elif method =="Implicit_Euler":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(atribute,t)
                    integral += z0*h_max

            elif method == "Trapezoid":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z1 = self.model(atribute,t)
                    integral += (z0+z1)*0.5*h_max
                    z0 = z1

            elif method == "Simpsons":
                z = []
                z.append(z0)
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute + h_max
                    z0 = self.model(atribute,t)
                    z.append(z0)
                integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])

            elif method == "Gaussian_Q":
                time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
                integral = 0
                for i in range(time_integral.shape[0]):
                    t = time_integral[i]
                    atribute = atribute + h_max
                    z0 = self.model(atribute,t)
                    integral += weights[i]*z0
                atribute = atribute + t1
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

    class Model(nn.Module):

        def __init__(self, input_size, output_size,
                     dropout, num_layers, step):
            super(FCN_point_process_all.Model, self).__init__()

            lst = nn.ModuleList()
            # Fully connected layer
            out_size = input_size
            for i in range(num_layers):
                inp_size = out_size
                out_size = int(inp_size//step)
                if i == num_layers-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size),
                                      nn.ReLU())
                lst.append(block)

            self.fc = nn.Sequential(*lst)

        def forward(self, x, t):
            t = t.to(torch.float32)
            x = torch.cat((x,t.reshape(1,-1)),dim = 1)
            out = torch.exp(self.fc(x))
            return out

    def fit(self, train_time, test_time, in_size, atribute_0 = None, no_steps = 10, h = None, no_epoch = 100, log = 1, log_epoch = 1, method ="Euler"):
        epochs, train_losses, test_losses = [], [], []

        " Training "
        self.model.train()

        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)

        for e in range(no_epoch):
            z_, integral_ = FCN_point_process_all.integral(self, train_time, in_size, no_steps = no_steps, h = h, method = method)
            train_loss = FCN_point_process_all.loss(z_, integral_)
            optimizer_1.zero_grad()
            train_loss.backward()
            optimizer_1.step()

            epochs.append(e)
            train_losses.append(train_loss)
            self.model.eval()
            z_test, integral_test = FCN_point_process_all.integral(self, test_time, in_size, no_steps=no_steps, h=h, method='Trapezoid')
            test_loss = FCN_point_process_all.loss(z_test, integral_test)
            test_losses.append(test_loss)
            self.model.train()

            if e%log_epoch==0 and log == 1:
                print(train_loss)
        return epochs, train_losses, test_losses

    def predict(self, time, atribute = None):
        self.model.eval()
        time_len = time.size(1)
        if atribute is None:
            atribute = torch.ones(in_size).reshape(1,-1)*0
        z = torch.zeros(time.shape)
        z0 = self.model(atribute,time[0,0])
        z[:,0] = z0
        for i in range(time_len-1):
            atribute = atribute + time[0,i+1]
            z[:,i+1] = self.model(atribute,time[:,i+1])
            atribute[:,1:] = atribute[:,:-1].clone()
            atribute[:,0] = 0
        return z

    def predict_sim(self, time, atribute):
        self.model.eval()
        time_len = time.size(1)
        z = torch.zeros(time.shape)
        z0 = self.model(atribute,time[0,0])
        z[:,0] = z0
        for i in range(time_len-1):
            atribute = atribute + time[0,i+1]
            z[:,i+1] = self.model(atribute,time[:,i+1])
        return z


    def evaluate(self, time, in_size, no_steps = 10, h = None, atribute = None, method = "Euler"):
        z_, integral_ = FCN_point_process_all.integral(self, time, in_size, no_steps, h = h, method = method)
        loss1 = FCN_point_process_all.loss(z_, integral_)
        return(loss1)


if __name__ == "__main__":

    time = np.abs(100*np.random.rand(10000))
    time.sort()
    in_size = 5

    out_size = 1
    time[0] = 0

    # train_time, test_time = time[:80], time[80:]
    # train_time = torch.tensor(train_time).type('torch.FloatTensor').reshape(1, -1, 1)
    # test_time = torch.tensor(test_time).type('torch.FloatTensor').reshape(1, -1, 1)
    project_dir = str(Path(__file__).parent.parent)

    data_folder = project_dir+'/../../data/geoloc/'
    dataset_path = 'zh_hb_main_station-24-25.082020.csv'  # os.environ["TRAINING_DATASET"]
    data = pd.read_csv(data_folder+dataset_path)

    train_data = data[data.day == 24]
    test_data = data[data.day == 25]
    test_data.loc[:, 'date1_ts'] = test_data.loc[:, 'date1_ts'] - test_data.loc[:, 'date1_ts'].min()
    train_time = torch.tensor(train_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    print(f'Train size: {str(train_time.shape[1])}, test size: {str(test_time.shape[1])} ('
          f'{round((test_time.shape[1] / (train_time.shape[1] + test_time.shape[1])), 2)} %).')

    mod = FCN_point_process_all(in_size+1, out_size, drop = 0.0)
    epochs, train_losses, test_losses = mod.fit(train_time, test_time, in_size, no_epoch=500, no_steps=10, h=None, method="Euler", log=1, log_epoch=10)
    print(mod.predict(train_time))
    loss_on_train = mod.evaluate(train_time, in_size)
    print(loss_on_train)

    train_losses, test_losses = [loss.detach().numpy()[0, 0] for loss in train_losses], \
                                [loss.detach().numpy()[0, 0] for loss in test_losses]
    plt.plot(epochs, train_losses, color='skyblue', linewidth=2, label='train')
    plt.plot(epochs, test_losses, color='darkgreen', linewidth=2, linestyle='dashed', label="test")
    plt.legend()
    # plt.savefig('../../img/fcn_test_data.png')

    # pickle.dump(mod, open('../../models/test.torch', 'wb'))

    # evaluation_df = pd.read_csv('../../results/baseline_scores.csv')
    # evaluation_df.loc[len(evaluation_df)] = ['FCN', 'synthetic', loss_on_train.data.numpy()[0][0], None]
    # evaluation_df.to_csv('../../results/baseline_scores.csv', index=False)
