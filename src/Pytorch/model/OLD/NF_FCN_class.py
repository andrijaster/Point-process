# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:30:54 2019

@author: Andri
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:02:42 2019

@author: Andri
"""


import torch
import numpy as np
from torch import nn, optim


class NF_FCN_point_process():
    
    def __init__(self, in_size, out_size, drop=0, num_layers = 2,
                 step = 2, step_z = 1.1):
        
        self.affine = NF_FCN_point_process.Affine_transform(in_size, step_z)
        
        inp_2 = int(in_size//step_z**2)
        self.flow = NF_FCN_point_process.Flow_transform(inp_2, out_size, 
                          dropout = drop, num_layers = num_layers, step = step)
    
    class Affine_transform(nn.Module):    
       
        def __init__(self, input_size, step_z):
            super(NF_FCN_point_process.Affine_transform, self).__init__()
            output = int(input_size//step_z)
            self.fc1 = nn.Sequential(nn.Linear(input_size, output),
                    nn.BatchNorm1d(num_features=output),
                    nn.ReLU())
            self.fc21 = nn.Linear(output, int(input_size//step_z**2))
            self.fc22 = nn.Linear(output, int(input_size//step_z**2))
    
        def P_zx(self, x):
            h1 = self.fc1(x)
            return self.fc21(h1), self.fc22(h1)  
        
        def reparameterize(self, mu, logvar, no_samples):
            std = torch.exp(0.5*logvar)
            number = list(std.size())
            number.insert(0,no_samples)
            eps = torch.randn(number)
            return mu + eps*std
        
        def forward(self, x, no_samples = 1):
            x = x.squeeze()
            mu, logvar = self.P_zx(x)
            z = self.reparameterize(mu, logvar, no_samples)
            return z    
    
    class Flow_transform(nn.Module):
        
        def __init__(self, input_size, output_size, 
                     dropout, num_layers, step):
            super(NF_FCN_point_process.Flow_transform, self).__init__()
            
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
        
        def forward(self, x):
            out = torch.exp(self.fc(x)).squeeze()  
            if len(out.shape) > 1:
                out = torch.transpose(out.squeeze(),0,1)  
            else:
                out = out.reshape(-1,1)
            return out
    
    
   
    def fit(self, x_train, time, targets, no_epoch = 100, log = 1, log_epoch = 5, 
            mini_batch_size =1):
        " Training "
        self.affine.train()
        self.flow.train()
        
        list_1 = list(self.affine.parameters()) + list(self.flow.parameters())
        optimizer_1 = torch.optim.Adam(list_1, lr = 0.01)
        
        def trapezoidal_integral_approx(t, y):
            return torch.sum(torch.mul(t[1:] - t[:-1],
                              (y[:-1] + y[1:]) / 2))
        
        def loss(time, targets, lam):
            y = lam*targets
            column = torch.nonzero(y[:,0])
            ML = 0
            ind = 0
            for i in column[:,0]:
                ML += torch.log(torch.mean(y[i]*torch.exp(-trapezoidal_integral_approx(time[ind:i+1], lam[ind:i+1]))))
#                ML += torch.log(y[i]) - trapezoidal_integral_approx(time[ind:i+1], lam[ind:i+1])
                ind = i
            return torch.neg(ML)
        
        for e in range(no_epoch):
            for i in range(0, x_train.shape[0], mini_batch_size):     
                batch_x, batch_targets = (x_train[i:i+mini_batch_size], targets[i:i+mini_batch_size])
                batch_targets = torch.transpose(batch_targets,0,1)
                z = self.affine(batch_x)
                lambda_par = self.flow(z)
                
#                lambda_par = lambda_par.reshape(-1,1)
                loss1 = loss(time, batch_targets, lambda_par)
                optimizer_1.zero_grad()
                loss1.backward()
                optimizer_1.step()
        
            if e%log_epoch==0 and log == 1:
                z = self.affine(x_train)
                lambda_tot = self.flow(z)
                tar = torch.transpose(targets,0,1)
                print(loss(time, tar, lambda_tot))
                
                
    def predict(self, x_test, no_samples = 5):
        self.affine.eval()
        self.flow.eval()
        z = self.affine(x_test, no_samples)
        output = self.flow(z)
        return output


if __name__ == "__main__":
    
    time = np.linspace(0,100,100)
    choice = np.random.choice(time, size= 30, replace = False)
    targets = np.isin(time,choice).astype(int)
    in_size = 11
    no_epoch = 2500
    out_size = 1
    num_unroll = len(time)
    x = np. random.randn(1,len(time),in_size-1)
    x_train = np.insert(x, 10, values = time, axis =2)
    mini_batch_size = 60
    
    
    x_train = torch.tensor(x_train).type('torch.FloatTensor')
    time = torch.tensor(time).type('torch.FloatTensor').reshape(-1,1)
    targets = torch.tensor(targets).type('torch.FloatTensor').reshape(1,-1)
    
    mod = NF_FCN_point_process(in_size, out_size, drop = 0.1)
    mod.fit(x_train,time,targets, no_epoch=no_epoch)
    print(mod.predict(x_train))