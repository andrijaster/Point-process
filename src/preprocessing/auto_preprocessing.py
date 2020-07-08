#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:22:43 2020

@author: dimitrijemilenkovic
"""

import math
import pandas as pd
import numpy as np
from src.preprocessing import create_train_space_and_targets, create_dataframe, create_feature_matrix


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


    df1 = pd.read_csv('012017_bgd-nis_1.csv', sep=';', header = None)
    df2 = pd.read_csv('022017_bgd-nis_2.csv', sep=';', header = None)

    frames = [df1, df2]
    result = pd.concat(frames)
    result.drop(labels = [0,3,5],axis=1, inplace = True)
    kolone = {1:'UlazStan',2:'UlazTraka',4:'date1',6:'IzlazStan',7:'IzlazTraka',8:'date2',9:'Kategorija',10:'tag'}
    result.rename(columns=kolone, inplace = True)

    # choosing location subset
    # result.UlazStan.value_counts()
    # result.query('UlazStan == 1').UlazTraka.value_counts()
    DATETIME_COLUMN = 'date1'
    LOCATION_CONDITION = 'UlazStan == 1 and UlazTraka == 3'
    location_result = result.query(LOCATION_CONDITION)

    # datetime parsing
    location_result.date1 = pd.to_datetime(location_result.date1)
    print(f"Rows before datetime rounding: {location_result.shape[0]}")
    location_result.date1 = location_result.date1.dt.round('15s')
    location_result.drop_duplicates(subset='date1', keep='first', inplace=True)
    print(f"Rows after datetime was rounded and duplicates are dropped: {location_result.shape[0]}")

    location_result = location_result[['UlazStan', 'UlazTraka', 'date1','Kategorija']]
    location_result['epoch_rounded'] = location_result[[DATETIME_COLUMN]].astype(np.int64) // math.pow(10,9)
    location_result.sort_values(by='epoch_rounded', inplace=True)

    location_result['time_points'] = location_result['epoch_rounded'] - location_result['epoch_rounded'].min()
    # save
    location_result.to_csv('data/autoput/012017_bg-nis-stan3-traka1-prepared.csv', index=False)
    location_result.head(20)

    # feature dataframe
    target_times = location_result[['time_points']].values.ravel()
    train_space, train_targets = create_train_space_and_targets(target_times, 0, 15)
    df = create_dataframe(train_space, train_targets)
    df = df[['num', 'target']]
    df.to_csv('data/autoput/012017_bg-nis-stan3-traka1-time_df.csv', index=False)
    feature_matrix = create_feature_matrix(df, 2)







