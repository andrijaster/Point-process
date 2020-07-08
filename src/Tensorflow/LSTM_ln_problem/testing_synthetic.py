# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:02:14 2019

@author: Andri
"""

import numpy as np
from LSTM_ln_class import LSTM_ln
import warnings
import os
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    name1 = 'graph_1'
    time = np.linspace(1,100,100)
    choice = np.random.choice(time, size= 30, replace = False)
    targets = np.isin(time,choice).astype(int)
    
    
    in_size = 11
    no_epoch = 100
    out_size = 1
    num_unroll = len(time)
    x = np. random.randn(1,len(time),in_size-1)
    x = np.insert(x, 10, values = time, axis =2)
    
    directory = os.getcwd()
    model1 = LSTM_ln(value = targets, input_size=in_size, output_size=out_size, lstm_size=32, num_layers=3,
                 num_steps=num_unroll, keep_prob=1, batch_size=1, init_learning_rate=0.1,
                 learning_rate_decay=0.999, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name1)
    
    
    time_prediction = np.linspace(1,50,50)
    choice_prediction = np.random.choice(time_prediction, size= 10, replace = False)
    targets_prediction = np.isin(time_prediction, choice_prediction).astype(int)
    targets_prediction = np.expand_dims(np.expand_dims(targets_prediction, axis = 0),axis=2)
    x_prediction = np. random.randn(1,len(time_prediction),in_size-1)
    x_prediction = np.insert(x_prediction, 10, values = time_prediction, axis =2)
    
    
    model1.build_lstm_graph_with_config()
    model1.train_lstm_graph(x, targets, time)
    seq = model1.prediction_by_trained_graph(no_epoch,x_prediction)
    seq1 = model1.prediction_by_trained_graph(no_epoch,x)