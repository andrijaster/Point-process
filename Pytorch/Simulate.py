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
    
    def __init__(self, fun, time_upper, time_lower):
        self.time_upper = time_upper
        self.time_lower = time_lower
        self.mod = fun
    
    def step_simulation(self, atribute = None, no_steps_max = 1000):
        n = 0
        m = 0
        tn = []
        sm = self.time_lower
        if atribute is None:
            atribute = torch.ones(in_size).reshape(1,-1)*0
        while sm < self.time_upper:
            time = np.linspace(sm, self.time_upper, no_steps_max)
            time = torch.tensor(time).type('torch.FloatTensor').reshape(1,-1,1)
            lamb_max = self.mod.predict_sim(time, atribute).max().data.numpy()
            if lamb_max.data == np.inf:
                lamb_max = 1e9
            u = np.random.rand()
            w = - np.log(u)/lamb_max
            sm = sm + w
            print(w,sm)
            D = np.random.rand()
            atribute = atribute + sm
            lamb = self.mod.predict_sim(torch.tensor(sm).reshape(1,-1), atribute).data.numpy()
            if sm>self.time_upper: break
            if D*lamb_max < lamb.data:
                tn.append(sm)
                atribute[:,1:] = atribute[:,:-1].clone()
                atribute[:,0] = 0
                n += 1
            m += 1
        return tn
    
    def simulate(self, no_simulation = 1):
        simulation = [Simulation.step_simulation(self) for _ in range(no_simulation)]
        return simulation
        


        
if __name__ == "__main__":
    
    time = np.abs(100*np.random.rand(100))
    time.sort()
    in_size = 5

    out_size = 1
    time[0] = 0

    time = torch.tensor(time).type('torch.FloatTensor').reshape(1,-1,1)

    mod = ml.FCN_point_process_all(in_size+1, out_size, drop = 0.0)
    mod.fit(time, in_size, no_epoch=150, no_steps = 10, h = None, method = "Trapezoid", log = 1, log_epoch=1)

    sim_model = Simulation(mod, time_upper = 100, time_lower = 0)     
    simulation = sim_model.simulate(no_simulation = 1) 
    print(simulation)
        