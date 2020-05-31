# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:45:27 2019

@author: Andri
"""

import torch
import numpy as np
from torch import nn, optim
from scipy.ndimage.interpolation import shift
from scipy.special import p_roots
import math

class FCN_point_process_all():
    
    def __init__(self, in_size, out_size, in_size_stoc, out_size_stoch, drop=0, num_layers = 2,
                 step = 2, drop_stoc=0, num_layers_stoc = 2,
                 step_stoc = 2):
        
        self.model = FCN_point_process_all.Model(in_size, out_size, 
                           dropout = drop,
                           num_layers = num_layers, step = step)

        self.model_stoc = FCN_point_process_all.Model(in_size_stoc, out_size_stoch, 
                           dropout = drop_stoc,
                           num_layers = num_layers_stoc, step = step_stoc)

    @staticmethod
    def loss(z, integral):
        ML = torch.sum(torch.log(z)) - integral
        return torch.neg(ML)

    def integral(self, time, in_size, no_steps, h = None , atribute = None, method = "Euler"):
        
        def integral_solve(z0, t0, t1, atribute_0, no_steps = 10, h = None, method = "Euler"):
            if no_steps is not None:
                h_max = (t1 - t0)/no_steps
            elif h is not None:
                no_steps = math.ceil((t1 - t0)/h)
                h_max = (t1 - t0)/no_steps

            integral = 0
            t = t0

            if method =="Euler": 
                for _ in range(no_steps):
                    integral += z0*h_max
                    t = t + h_max
                    atribute = atribute_0 + t
                    z0 = self.model(atribute,t) + self.model_stoc(atribute,t)                       

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
        
    def fit(self, time, in_size, atribute_0 = None, no_steps = 10, h = None, no_epoch = 100, log = 1, log_epoch = 1, method = "Euler"):

        " Training "
        self.model.train()
        
        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        for e in range(no_epoch):
            z_, integral_ = FCN_point_process_all.integral(self, time, in_size, no_steps = no_steps, h = h, method = method)
            loss1 = FCN_point_process_all.loss(z_, integral_)
            optimizer_1.zero_grad()
            loss1.backward()
            optimizer_1.step()
            if e%log_epoch==0 and log == 1:
                print(loss1)
                
                
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

    def evaluate(self, time, in_size, no_steps = 10, h = None, atribute = None, method = "Euler"):
        z_, integral_ = FCN_point_process_all.integral(self, time, in_size, no_steps, h = h, method = method)
        loss1 = FCN_point_process_all.loss(z_, integral_)
        return(loss1)



if __name__ == "__main__":
    
    time = np.abs(100*np.random.rand(100))
    time.sort()
    in_size = 5
    
    out_size = 1
    time[0] = 0
    
    time = torch.tensor(time).type('torch.FloatTensor').reshape(1,-1,1)
    
    mod = FCN_point_process_all(in_size+1, out_size, drop = 0.0)
    mod.fit(time, in_size, no_epoch=50, no_steps = 10, h = None, method = "Trapezoid", log = 1, log_epoch=10)
    print(mod.predict(time))
    print(mod.evaluate(time, in_size))