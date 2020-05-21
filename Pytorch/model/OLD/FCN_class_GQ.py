# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:45:27 2019

@author: Andri
"""

import torch
import numpy as np
from torch import nn, optim
import random


class FCN_point_process():
    
    def __init__(self, in_size, out_size, drop=0, num_layers = 2,
                 step = 2):
        self.model = FCN_point_process.Model(in_size, out_size, 
                           dropout = drop,
                           num_layers = num_layers, step = step)
        
    def Gaussian_quadrature(vec, lower_l, upper_l):
        vec = vec.reshape(-1)
        m = (upper_l-lower_l)/2
        c = (upper_l+lower_l)/2
        vec = (vec-c)/m
        A = np.array([vec**i for i in range(0,vec.shape[0])])
        B = np.array([(1**i-(-1)**i)/i for i in range(1,vec.shape[0]+1)])
        x = m*np.linalg.solve(A, B)
        return x
    
    def loss(lam, weights):
        ML=0
        for i in range(len(lam)):
            ML += torch.sum(torch.log(lam[i])) - torch.sum(weights[i]*lam[i])
        return torch.neg(ML)

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
            out = []
            for i in range(x.shape[0]):
                out.append(torch.exp(self.fc(x[i])))
            return out
     
    def fit(self, x_train, time, no_epoch = 100, log = 1, log_epoch = 5, 
            mini_batch_size = 10):
        
        " Training "
        self.model.train()
        
        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        self.weights = []
        if time.shape[0]>1:
            for i in range(time.shape[0]):
                weights = FCN_point_process.Gaussian_quadrature(time[i], lower_l = 0, upper_l = time[i,-1])
                self.weights.append(torch.tensor(weights))
        else:
            weights = FCN_point_process.Gaussian_quadrature(time, lower_l = 0, upper_l = time[-1])
            self.weights = torch.tensor(weights)
        
        for e in range(no_epoch):
            for i in range(0, x_train.shape[0], mini_batch_size):     
                batch_x = x_train[i:i+mini_batch_size]
                weights = self.weights[i:i+mini_batch_size]
                lambda_par = self.model(batch_x)
                loss1 = FCN_point_process.loss(lambda_par, weights)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
        
            if e%log_epoch==0 and log == 1:
                lambda_tot = self.model(x_train)
                print(FCN_point_process.loss(lambda_tot, self.weights))
                
                
    def predict(self, x_test):
        self.model.eval()
        output =self.model(x_test)
        return output
    
    def evaluate_NLL(self, x_test, time_test):
        self.model.eval()
        output =self.model(x_test)
        weights_eval = []
        if time_test.shape[0]>1:
            for i in range(time_test.shape[0]):
                weights = FCN_point_process.Gaussian_quadrature(time_test[i], lower_l = 0, upper_l = time_test[i,-1])
                weights_eval.append(torch.tensor(weights))
        else:
            weights = FCN_point_process.Gaussian_quadrature(time_test[i], lower_l = 0, upper_l = time_test[-1])
            weights_eval = torch.tensor(weights)
        NLL = FCN_point_process.loss(output, weights_eval)
        return NLL
    


if __name__ == "__main__":
    
    time = 100*np.random.rand(100)
    time.sort()
    in_size = 11
    no_epoch = 10000
    out_size = 1
    # time_train = np.random.choice(time, size= 30, replace = False)
    x = np.random.randn(2,len(time),in_size-1)
    x = np.insert(x, 10, values = time, axis =2)
    idx = random.sample(range(x.shape[1]), 50)
    idx.sort()
    x_train = x[:,idx,:]
    time_train = x[:,idx,-1]
    time_test = x[:,:,-1]
    
    
    x_train = torch.tensor(x_train).type('torch.FloatTensor')
    x = torch.tensor(x).type('torch.FloatTensor')
    # time = torch.tensor(time).type('torch.FloatTensor').reshape(-1,1)
    
    mod = FCN_point_process(in_size, out_size, drop = 0.0)
    mod.fit(x_train, time_train, no_epoch = no_epoch)
    print(mod.evaluate_NLL(x,time_test))
    print(mod.predict(x))