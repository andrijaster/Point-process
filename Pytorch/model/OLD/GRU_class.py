# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 18:37:18 2019

@author: Andri
"""

import torch
import numpy as np
from torch import nn, optim


class GRU_point_process():
    
    def __init__(self, in_size, out_size, h_dim, n_layer, drop=0, num_layers = 2,
                 step = 2):
        self.model = GRU_point_process.Model(in_size, out_size, 
                           hidden_dim=h_dim, n_layers=n_layer, dropout = drop,
                           num_layers = num_layers, step = step)
    
    class Model(nn.Module):
        
        def __init__(self, input_size, output_size, hidden_dim, 
                     n_layers, dropout, num_layers, step):
            super(GRU_point_process.Model, self).__init__()
    
            # Defining some parameters
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
    
            #Defining the layers
            # RNN Layer
            self.rnn = nn.GRU(input_size, hidden_dim, n_layers, 
                              dropout = dropout, batch_first=True)   
            lst = nn.ModuleList()
            # Fully connected layer
            out_size = hidden_dim
            for i in range(num_layers):
                inp_size = out_size
                out_size = int(inp_size//step)
                if i == num_layers-1:
                    block = nn.Linear(inp_size, 1)
                else:
                    block = nn.Sequential(nn.Linear(inp_size, out_size), 
                                      nn.BatchNorm1d(num_features = out_size),
                                      nn.Sigmoid())    
                lst.append(block)
            
            self.fc = nn.Sequential(*lst)
        
        
        def forward(self, x):
            
            batch_size = x.size(0)
    
            # Initializing hidden state for first input using method defined below
            hidden = self.init_hidden(batch_size)
    
            # Passing in the input and hidden state into the model and obtaining outputs
            out, hidden = self.rnn(x, hidden)
            
            # Reshaping the outputs such that it can be fit into the fully connected layer
            if out.shape[0] == 1:
                out = out.contiguous().view(-1, self.hidden_dim)
                out = torch.exp(self.fc(out))
            else: out = torch.transpose(torch.exp(self.fc(out)).squeeze(),0,1)
                    
            return out, hidden
        
        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
            return hidden
    
    def fit(self, x_train, time, targets, no_epoch = 100, log = 1, log_epoch = 5, 
            mini_batch_size =1):
        " Training "
        self.model.train()
        
        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        def trapezoidal_integral_approx(t, y):
            return torch.sum(torch.mul(t[1:] - t[:-1],
                              (y[:-1] + y[1:]) / 2))
        
        def loss(time, targets, lam):
            y = lam*targets
            column = torch.nonzero(y)
            y = y[column[:,0]]
            ML = torch.sum(torch.log(y)) - trapezoidal_integral_approx(time, lam)
            return torch.neg(ML)
        
        for e in range(no_epoch):
            for i in range(0, x_train.shape[0], mini_batch_size):     
                batch_x, batch_targets = (x_train[i:i+mini_batch_size], targets[i:i+mini_batch_size])
                batch_targets = torch.transpose(batch_targets,0,1)
                lambda_par, _ = self.model(batch_x)
                loss1 = loss(time, batch_targets, lambda_par)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
        
            if e%log_epoch==0 and log == 1:
                lambda_tot, _ = self.model(x_train)
                tar = torch.transpose(targets,0,1)
                print(loss(time, tar, lambda_tot))
                
                
    def predict(self, x_test):
        self.model.eval()
        output,_ =self.model(x_test)
        return output


if __name__ == "__main__":
    
    time = np.linspace(1,100,100)
    choice = np.random.choice(time, size= 30, replace = False)
    targets = np.isin(time,choice).astype(int)
    in_size = 11
    no_epoch = 1000
    out_size = 1
    num_unroll = len(time)
    x = np. random.randn(1,len(time),in_size-1)
    x_train = np.insert(x, 10, values = time, axis =2)
    mini_batch_size = 60
    
    
    x_train = torch.tensor(x_train).type('torch.FloatTensor')
    time = torch.tensor(time).type('torch.FloatTensor').reshape(-1,1)
    targets = torch.tensor(targets).type('torch.FloatTensor').reshape(1,-1)
    
    mod = GRU_point_process(in_size, out_size, h_dim=32, n_layer=3, drop = 0.1)
    mod.fit(x_train,time,targets)
    print(mod.predict(x_train))