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

class RNN_point_process_all():
    
    def __init__(self, in_size, out_size, drop=0, num_layers = 2, n_layers = 2,
                 step = 2, hidden_dim = 2):
        
        self.model = RNN_point_process_all.Model(in_size, out_size, 
                           dropout = drop, n_layers = n_layers,
                           num_layers = num_layers, hidden_dim = hidden_dim, step = step)

    @staticmethod
    def loss(z, integral):
        ML = torch.sum(torch.log(z)) - integral
        return torch.neg(ML)

    def integral(self, time, in_size, no_steps, h = None , atribute = None, method = "Euler"):

        def integral_solve(z0, t0, t1, atribute_0, no_steps = 10, 
                            h = None, hidden = None, method = "Euler"):
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
                    atribute = atribute_0 + t
                    z0, hidden = self.model(atribute, t, hidden = hidden)            

            if method =="Implicit_Euler": 
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute_0 + t
                    z0, hidden = self.model(atribute, t, hidden = hidden)
                    integral += z0*h_max

            if method == "Trapezoid":
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute_0 + t
                    z1, hidden = self.model(atribute, t, hidden = hidden )
                    integral += (z0+z1)*0.5*h_max 
                    z0 = z1  
            
            if method == "Simpsons":
                z = []
                z.append(z0)
                for _ in range(no_steps):
                    t = t + h_max
                    atribute = atribute_0 + t
                    z0, hidden = self.model(atribute, t, hidden = hidden)
                    z.append(z0)       
                integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])      

            if method == "Gaussian_Q":
                time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
                integral = 0
                for i in range(time_integral.shape[0]):
                    t = time_integral[i]
                    atribute = atribute_0 + t
                    z0 , hidden= self.model(atribute, t, hidden = hidden)
                    integral += weights[i]*z0  
                atribute = atribute_0 + t1
                z0, hidden = self.model(atribute,t1)                  

            return integral, z0, atribute, hidden

        integral_ = 0
        time_len = time.size(1)
        if atribute is None:
            atribute = torch.ones(in_size).reshape(1,-1)*0
        z = torch.zeros(time.shape)
        z0, hidden = self.model(atribute,time[0,0])
        z[:,0] = z0
        for i in range(time_len-1):
            integral_interval, z_, atribute, hidden = integral_solve(z0, time[0,i], time[0,i+1], atribute, hidden = hidden, no_steps = no_steps, h = h, method = method)
            integral_ += integral_interval
            atribute[:,1:] = atribute[:,:-1].clone()
            atribute[:,0] = 0
            z[:,i+1] = z_
        return z, integral_
    
    class Model(nn.Module):
        
        def __init__(self, input_size, output_size, hidden_dim, 
                     n_layers, dropout, num_layers, step):
            
            super(RNN_point_process_all.Model, self).__init__()
    
            # Defining some parameters
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
    
            #Defining the layers
            # RNN Layer
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, 
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
                                      nn.Sigmoid())    
                lst.append(block)
            
            self.fc = nn.Sequential(*lst)

                    
        def forward(self, x, t, hidden = None):
            t = t.to(torch.float32)
            x = torch.cat((x,t.reshape(1,-1)),dim = 1)
            x = x.reshape(1,1,-1)
            
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
            # This method generates the first hidden state of zeros which we'll use in the forward pass
            # We'll send the tensor holding the hidden state to the device we specified earlier as well
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            return hidden

    def fit(self, time, in_size, atribute_0 = None, no_steps = 10, h = None, no_epoch = 100, log = 1, log_epoch = 1, method = "Euler"):

        " Training "
        self.model.train()
        
        list_1 = list(self.model.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.001)
        
        for e in range(no_epoch):
            z_, integral_ = RNN_point_process_all.integral(self, time, in_size, no_steps = no_steps, h = h, method = method)
            loss1 = RNN_point_process_all.loss(z_, integral_)
            optimizer_1.zero_grad()
            loss1.backward()
            optimizer_1.step()
            if e%log_epoch==0 and log == 1:
                print(loss1)
                
                
    def predict(self, time, hidden = None, atribute = None):
        self.model.eval()
        time_len = time.size(1)
        if atribute is None:
            atribute = torch.ones(in_size).reshape(1,-1)*0
        z = torch.zeros(time.shape)
        if hidden is None:
            z0, hidden = self.model(atribute,time[0,0])
        elif hidden is not None:
            z0, hidden = self.model(atribute,time[0,0], hidden)
        z[:,0] = z0
        for i in range(time_len-1):
            atribute = atribute + time[0,i+1]
            z[:,i+1], hidden = self.model(atribute,time[:,i+1], hidden)
            atribute[:,1:] = atribute[:,:-1].clone()
            atribute[:,0] = 0
        return z

    def evaluate(self, time, in_size, no_steps = 10, h = None, atribute = None, method = "Euler"):
        z_, integral_ = RNN_point_process_all.integral(self, time, in_size, no_steps, h = h, method = method)
        loss1 = RNN_point_process_all.loss(z_, integral_)
        return(loss1)



if __name__ == "__main__":
    
    time = np.abs(100*np.random.rand(100))
    time.sort()
    in_size = 5
    out_size = 1
    time[0] = 0
    
    time = torch.tensor(time).type('torch.FloatTensor').reshape(1,-1,1)
    
    mod = RNN_point_process_all(in_size+1, out_size, drop = 0.0)
    
    mod.fit(time, in_size, no_epoch=50, no_steps = 10, h = None, method = "Trapezoid", log = 1, log_epoch=1)
    print(mod.predict(time))
    print(mod.evaluate(time, in_size))