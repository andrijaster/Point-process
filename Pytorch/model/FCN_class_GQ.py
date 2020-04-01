# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:45:27 2019

@author: Andri
"""

import torch
import numpy as np
from torch import nn, optim


class FCN_point_process():
    
    def __init__(self, in_size, out_size, drop=0, num_layers = 2,
                 step = 2):
        self.model = FCN_point_process.Model(in_size, out_size, 
                           dropout = drop,
                           num_layers = num_layers, step = step)
    
    class Model(nn.Module):
        
        def __init__(self, input_size, output_size, 
                     dropout, num_layers, step):
            super(FCN_point_process.Model, self).__init__()
            
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
                                      nn.BatchNorm1d(num_features = out_size),
                                      nn.ReLU())    
                lst.append(block)
            
            self.fc = nn.Sequential(*lst)
        
        def forward(self, x):
            out = torch.exp(self.fc(x[0]))                    
            return out
        
    def fit(self, x_train, time, no_epoch = 100, log = 1, log_epoch = 5, 
            mini_batch_size =1):
        
        def Gaussian_quadrature(vec, lower_l, upper_l):
            vec = vec.data.numpy().reshape(-1)
            upper_l = upper_l.data.numpy()
            m = (upper_l-lower_l)/2
            c = (upper_l+lower_l)/2
            vec = (vec-c)/m
            A = np.array([vec**i for i in range(0,vec.shape[0])])
            B = np.array([(1**i-(-1)**i)/i for i in range(1,vec.shape[0]+1)])
            x = m*np.linalg.solve(A, B)
            return x
        
        " Training "
        self.model.train()
        
        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        def loss(lam, weights):
            ML = torch.sum(torch.log(lam)) - torch.sum(weights*lam)
            return torch.neg(ML)
        
        weights = Gaussian_quadrature(time, lower_l = 0, upper_l = time[-1])
        weights = torch.tensor(weights)
        
        for e in range(no_epoch):
            for i in range(0, x_train.shape[0], mini_batch_size):     
                batch_x = x_train[i:i+mini_batch_size]
                lambda_par = self.model(batch_x)
                loss1 = loss(lambda_par, weights)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
        
            if e%log_epoch==0 and log == 1:
                lambda_tot = self.model(x_train)
                print(loss(time, lambda_tot))
                
                
    def predict(self, x_test):
        self.model.eval()
        output =self.model(x_test)
        return output


if __name__ == "__main__":
    
    time = 1000*np.random.rand(1000)
    time.sort()
    in_size = 11
    no_epoch = 5000
    out_size = 1
    num_unroll = len(time)
    x = np.random.randn(1,len(time),in_size-1)
    x_train = np.insert(x, 10, values = time, axis =2)
    
    
    x_train = torch.tensor(x_train).type('torch.FloatTensor')
    time = torch.tensor(time).type('torch.FloatTensor').reshape(-1,1)
    
    mod = FCN_point_process(in_size, out_size, drop = 0.0)
    mod.fit(x_train, time, no_epoch = no_epoch)
    print(mod.predict(x_train))