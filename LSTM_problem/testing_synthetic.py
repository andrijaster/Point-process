# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:02:14 2019

@author: Andri
"""

import numpy as np
from LSTM_class import LSTM
import warnings
import os
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    name1 = 'graph_1'
    time = np.linspace(0,3600,7201)
    choice = np.random.choice(time, size= 800, replace = False)
    targets = np.isin(time,choice).astype(int)
    targets = np.expand_dims(np.expand_dims(targets, axis = 0),axis=2)
    
    
    in_size = 11
    no_epoch = 100
    out_size = 1
    num_unroll = len(time)
    x = np. random.randn(1,len(time),in_size-1)
    x = np.insert(x, 10, values = time, axis =2)
    
    directory = os.getcwd()
    model1 = LSTM(input_size=in_size, output_size=out_size, lstm_size=32, num_layers=2,
                 num_steps=num_unroll, keep_prob=1, batch_size=1, init_learning_rate=0.01,
                 learning_rate_decay=0.992, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name1)
    
    model1.build_lstm_graph_with_config()
    model1.train_lstm_graph(x, targets, time)