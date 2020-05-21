# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:45:27 2019

@author: Andri
"""

import torch
import numpy as np
from torch import nn, optim
from scipy.special import p_roots
import random


class FCN_point_process():
    
    def __init__(self, in_size, out_size = 1, drop=0, num_layers = 2,
                 step = 2):
        self.model = FCN_point_process.Model(in_size, out_size, 
                           dropout = drop,
                           num_layers = num_layers, step = step)
  
    def Gaussian_quadrature(self, n, lower_l, upper_l):
        m = (upper_l-lower_l)/2
        c = (upper_l+lower_l)/2
        [x,w] = p_roots(n+1)
        self.weights = w*m
        # self.weights = torch.tensor(self.weights).type('torch.FloatTensor') 
        self.time_train_integral = m*x+c
        return [self.time_train_integral, self.weights]
    
    def loss(lam, targets, weights):
        ML=0
        for i in range(len(lam)):
            y1 = torch.log(lam[i]) * targets[:,i].reshape(-1,1)
            y2 = weights*lam[i][torch.where(targets[:,i]==0)]
            ML += torch.sum(y1) - torch.sum(y2)
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
     
    def fit(self, x_train, time, targets, weights, no_epoch = 100, log = 1, log_epoch = 5, 
            mini_batch_size = 10):
        
        " Training "
        self.model.train()
    
        targets = 1-torch. transpose(targets,0,1)
        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        for e in range(no_epoch):
            for i in range(0, x_train.shape[0], mini_batch_size):     
                batch_x = x_train[i:i+mini_batch_size]
                lambda_par = self.model(batch_x)
                loss1 = FCN_point_process.loss(lambda_par, targets, weights)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
        
            if e%log_epoch==0 and log == 1:
                lambda_tot = self.model(x_train)
                print(FCN_point_process.loss(lambda_tot, targets, self.weights))
                
                
    def predict(self, x_test):
        self.model.eval()
        output =self.model(x_test)
        return output
    
    def evaluate_NLL(self, x_test, time_test):
        self.model.eval()
        output =self.model(x_test)
        weights_eval = []
        NLL = FCN_point_process.loss(output, weights_eval)
        return NLL
    


if __name__ == "__main__":
    
    time = 100*np.random.rand(100)
    time.sort()
    in_size = 11
    no_rows = 10
    no_epoch = 30000
    
    x = np.random.randn(no_rows,len(time),in_size-1)
    x = np.insert(x, no_rows, values = time, axis =2)
    idx = random.sample(range(x.shape[1]), 15)
    idx.sort()
    x_train_occurance = x[:,idx,:]
    time_train_occurance = x[:,idx,-1]
    # time_test = x[:,:,-1]
    
    
    # x_train = torch.tensor(x_train).type('torch.FloatTensor')
    # x = torch.tensor(x).type('torch.FloatTensor')
    # time = torch.tensor(time).type('torch.FloatTensor').reshape(-1,1)
    
    mod = FCN_point_process(in_size, drop = 0.0)
    [time_train_integral, weights] = mod.Gaussian_quadrature(n = 50,lower_l = 0, upper_l = 100)
    weights = weights.reshape(-1,1)
    time_train = []
    targets = []
    for i in range(no_rows):
        val = np.random.choice(time, size= 30, replace = False)
        val = np.insert(val, 0, time_train_integral, axis =0)
        val.sort()
        targets.append(np.isin(val,time_train_integral))
        time_train.append(val)
    
    time_train = np.array(time_train)
    targets = np.array(targets)*1
    
    # weight = np.zeros(targets.shape)
    # for i in range(targets.shape[0]):
    #     index = np.where(targets[i] == 1)
    #     weight[i,index[0]] = weights
            
    x_train = np.random.randn(no_rows,time_train.shape[1],in_size-1)
    x_train = np.insert(x_train, 10, values = time_train, axis =2)
    
    x_train = torch.tensor(x_train).type('torch.FloatTensor')
    time_train = torch.tensor(time_train).type('torch.FloatTensor') 
    targets = torch.tensor(targets).type('torch.FloatTensor')
    weights = torch.tensor(weights).type('torch.FloatTensor')

    
    mod.fit(x_train, time_train, targets, weights, no_epoch = no_epoch)
    # print(mod.evaluate_NLL(x,time_test))
    print(mod.predict(x_train))