# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:54:46 2019

@author: Andri
"""

import model as ml
import torch
import numpy as np
from scipy.interpolate import interp1d

class Simulation:
    
    def __init__(self, fun, x, time):
        self.time = time.squeeze()
        lamb = fun.predict(x).data.numpy().squeeze()[:,0]
        lamb = np.mean(lamb, axis = 1)
        self.lamb_y = interp1d(self.time, lamb)
        self.lambmax = lamb.max()
    
    def step_simulation(self):
        sm = 0
        n = 0
        m = 0
        tn = []
        while sm<self.time[-1]:
            u = np.random.rand()
            w = - np.log(u)/self.lambmax
            sm += w
            D = np.random.rand()
            if sm>time[-1]: break
            if D < self.lamb_y(sm):
                tn.append(sm)
                n += 1
            m += 1
        return tn
    
    def simulate(self, no_simulation):
        simulation = [Simulation.step_simulation(self) for _ in range(no_simulation)]
        return simulation
        


        
if __name__ == "__main__":
    
    time = np.linspace(0,100,100)
    choice = np.random.choice(time, size= 25, replace = False)
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
    
    mod = ml.NF_FCN_point_process(in_size, out_size, num_layers=3 ,drop = 0.1)
    mod.fit(x_train,time,targets, no_epoch = no_epoch)
    sim_model = Simulation(mod, x_train, time)     
    simulation = sim_model.simulate(no_simulation = 100) 
        