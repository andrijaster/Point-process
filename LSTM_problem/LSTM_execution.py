# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:51:57 2019

@author: Andri
"""

import numpy as np
import pandas as pd
import os
from LSTM_class import LSTM
from Train_test_datasets import Train_test
import warnings
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

atributes = pd.read_csv('Atribute.csv',index_col=0)
output = pd.read_csv('Output.csv',index_col=0)



if __name__ == "__main__":
    
    atributes = pd.read_csv('Atribute.csv',index_col=0)
    output = pd.read_csv('Output.csv',index_col=0) 
    num_unroll = 320
    test_size1 = 0.2
    test_size2 = 0.35
    train1 = Train_test(num_unroll,test_size1,test_size2)
    train_X, test_X, train_y, test_y, train_X_mean, train_X_var, train_y_mean, train_y_var = train1.train_test_fun(atributes, output)
    
    steps_size = np.array([2**i for i in range(4,8)])
    steps_layers = np.arange(1,4)
    R2_score_1 = np.zeros([len(steps_layers), len(steps_size)])
    R2_score_2 = np.zeros([len(steps_layers), len(steps_size)])
    MAE_1 = np.zeros([len(steps_layers), len(steps_size)])
    MAE_2 = np.zeros([len(steps_layers), len(steps_size)])
    MSE_1 = np.zeros([len(steps_layers), len(steps_size)])
    MSE_2 = np.zeros([len(steps_layers), len(steps_size)])
    i=0

    
    for layers in steps_layers:
        j=0        
        for size in steps_size:
        
            name1 = 'graph_1'
            name2 = 'graph_2'
           
            _, in_size = atributes.shape
            _, out_size = output.shape
            no_epoch = 200
            directory = os.getcwd()
    
            model1 = LSTM(input_size=in_size, output_size=out_size, lstm_size=size, num_layers=layers,
                             num_steps=num_unroll, keep_prob=0.8, batch_size=256, init_learning_rate=0.10,
                             learning_rate_decay=0.992, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name1)
            model2 = LSTM(input_size=in_size, output_size=out_size, lstm_size=size, num_layers=layers,
                             num_steps=num_unroll, keep_prob=1, batch_size=256, init_learning_rate=0.10,
                             learning_rate_decay=0.992, init_epoch=7, max_epoch=no_epoch, MODEL_DIR = directory, name = name2)
            
            model1.build_lstm_graph_with_config()
            model1.train_lstm_graph(train_X_mean, train_y_mean, train_X_var, train_y_var)
            prediction,loss = model1.prediction_by_trained_graph(no_epoch,test_X, test_y)
            
            model2.build_lstm_graph_with_config()
            model2.train_lstm_graph(train_X_mean, train_y_mean, train_X_var, train_y_var)
            prediction2,loss2 = model2.prediction_by_trained_graph(no_epoch,test_X, test_y)

                    
            R2_score_1[i,j] = r2_score(test_y,prediction)            
            R2_score_2[i,j] = r2_score(test_y,prediction2)
            MAE_1[i,j] = mean_absolute_error(test_y,prediction)
            MAE_2[i,j] = mean_absolute_error(test_y,prediction2)
            MSE_1[i,j] = mean_squared_error(test_y,prediction)
            MSE_2[i,j] = mean_squared_error(test_y,prediction2)
            
            j+=1
        i+=1
        
    np.savetxt('R2_score_1.csv', R2_score_1)
    np.savetxt('R2_score_2.csv', R2_score_2)
    np.savetxt('MAE_1.csv', MAE_1)
    np.savetxt('MAE_2.csv', MAE_2)
    np.savetxt('MSE_1.csv', MSE_1)
    np.savetxt('MSE_2.csv', MSE_2)    