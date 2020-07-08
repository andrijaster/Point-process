# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from src.Pytorch import FCN_point_process
from src.Pytorch import LSTM_point_process
import torch

#df.columns
#counts = df[['VendorID', 'PULocationID']].groupby(['PULocationID']).agg(['count'])
#counts.head()
#counts.columns = counts.columns.to_flat_index()
#counts.columns = ['location', 'count']
#counts.sort_values(by='count', ascending=False)
#one_location_df[['VendorID', 'PULocationID']].groupby(['PULocationID']).agg(['count'])


def get_arr_of_times_ms_based_on_query(input_df, condition, time_col):

    one_location_series = input_df.query(condition)[[time_col]]
    one_location_series = one_location_series.astype('str')
    one_location_series = pd.to_datetime( \
            one_location_series.stack(),  \
            infer_datetime_format=True).unstack()
    one_location_series.sort_values(by=time_col, ascending=True, inplace=True)

    one_location_series = one_location_series.iloc[1:,:] # CUSTOM: remove outlier from 2008
    one_location_series.loc[:,time_col] = one_location_series.loc[:,time_col].astype(np.int64) // math.pow(10,9)

    arr_of_times_from_zero = one_location_series[[time_col]].values.ravel()

    return arr_of_times_from_zero


def create_train_space_and_targets(event_times, delta_seconds, time_step):
    min_ts, max_ts = event_times.min(), event_times.max()
    train_space = np.arange(min_ts-delta_seconds, max_ts+delta_seconds, time_step)
    print('Training: The event happens every apx. ' + str(train_space.shape[0] / event_times.shape[0]) + ' time step.')

    train_targets = np.isin(train_space,event_times).astype(int)

    return train_space, train_targets

def create_dataframe(X, y):
    df = pd.DataFrame({'time': X, 'target': y}, columns=['time', 'target'])
    df.reset_index(inplace=True)
    df.columns = ['num', 'time', 'target']
    return df

def generate_times_since_and_between_last_n(full_df, row, n=5):
    prior = full_df.loc[:(row.num-1), :]
    happen = prior.query('target == 1').reset_index(drop=True).iloc[-(n+1):,:].reset_index(drop=True)
    if happen.shape[0] < n:
        for i in range(1,n+1):
            row['time_since_'+str(i)] = np.nan
            row['between_'+str(i)] = np.nan
    else:
       for i in range(1,n+1):
           row['time_since_'+str(i)] = row.num - happen.iloc[-i,1]
           if(i==1):
               row['between_'+str(i)] = row.num - happen.iloc[-i,1]
           else:
               row['between_'+str(i)] = happen.iloc[-i+1,1] - happen.iloc[-i,1]
    return row

def create_feature_matrix(base_df, n_last=5):
    tmp_df = base_df.apply(lambda row: generate_times_since_and_between_last_n(base_df, row, n_last), axis=1)
    fm = tmp_df.loc[~tmp_df['time_since_1'].isna(),:].reset_index(drop=True)
    return fm


def get_X_y_and_time(dataset, X_cols, y_col='target', time_col='time'):
    X_np, y_np = dataset[X_cols].values.T, dataset[y_col].values
    time_np = dataset[time_col].values

    X = torch.tensor(np.dstack(X_np)).type('torch.FloatTensor')
    time = torch.tensor(time_np).type('torch.FloatTensor').reshape(-1,1)
    y = torch.tensor(y_np).type('torch.FloatTensor').reshape(1,-1)

    return X, y, time

if __name__ == "__main__":

    DATETIME_COLUMN = 'tpep_pickup_datetime'
    LOCATION_CONDITION = 'PULocationID == 237.0'

    EVENT_TIMES_FILE_PATH = 'data/ny_taxi_237_prepared'
    OUTPUT_FILE_TRAIN_SPACE = 'data/taxi_train_space'
    OUTPUT_FILE_TRAIN_TARGETS = 'data/taxi_train_targets'

    DELTA_SECONDS = 3600


    taxi_df = pd.read_csv("2018_Yellow_Taxi_Trip_Data.csv")
    arr_of_event_times = get_arr_of_times_ms_based_on_query(taxi_df, LOCATION_CONDITION, DATETIME_COLUMN)
    np.save(EVENT_TIMES_FILE_PATH, arr_of_event_times)
    print(arr_of_event_times)
    print('Array has been saved in file: ' + EVENT_TIMES_FILE_PATH)
    # arr_of_event_times = np.load(FILE_PATH+'.npy')
    train_space, train_targets = create_train_space_and_targets(arr_of_event_times, DELTA_SECONDS)
    np.save(OUTPUT_FILE_TRAIN_SPACE, train_space)
    np.save(OUTPUT_FILE_TRAIN_TARGETS, train_targets)
    print('Files have been saved in  ' + OUTPUT_FILE_TRAIN_SPACE + ' and ' + OUTPUT_FILE_TRAIN_TARGETS)
    # train_space, train_targets = np.load(OUTPUT_FILE_TRAIN_SPACE), np.load(OUTPUT_FILE_TRAIN_TARGETS)
    df = create_dataframe(train_space, train_targets)
    small_df = df.iloc[:50000, :]
    feature_matrix = create_feature_matrix(small_df)

    feature_matrix.time = feature_matrix.time - feature_matrix.time.min()

    index_divider = round(feature_matrix.shape[0]*0.8)
    train_set, test_set = feature_matrix.iloc[:index_divider, :], feature_matrix.iloc[index_divider:, :]
    train_set.to_csv('data/taxi_feature_matrix_train.csv', index=False)
    test_set.to_csv('data/taxi_feature_matrix_test.csv', index=False)

    train_set = pd.read_csv('data/taxi_feature_matrix_train.csv')
    test_set  = pd.read_csv('data/taxi_feature_matrix_test.csv')
    in_size = 11
    no_epoch = 5000
    out_size = 1

    y_column = 'target'
    X_columns = ['time', 'time_since_1', 'between_1', 'time_since_2',
       'between_2', 'time_since_3', 'between_3', 'time_since_4', 'between_4',
       'time_since_5', 'between_5']
    time_column = 'time'

    X, y, time = get_X_y_and_time(train_set, X_columns, y_column, time_column)
    mod = FCN_point_process(in_size, out_size, drop = 0.1)
    mod.fit(X, time, y, no_epoch = no_epoch)


    X_pred_on_training = mod.predict(X)
    np_X_pred_on_training = np.rint(X_pred_on_training.detach().numpy()).flatten()
    sum_of_errors = (y_train - np_X_pred_on_training).sum()
    print(str(sum_of_errors/np_X_pred_on_training.shape[0]) + ' on training')

    X_test.shape
    X_test, y_test, time_test = get_X_y_and_time(test_set, X_columns, y_column, time_column)
    X_pred_on_test = mod.predict(X_test)
    np_X_pred_on_test = np.rint(X_pred_on_test.detach().numpy()).flatten()
    sum_of_errors_test = (y_test - np_X_pred_on_test).sum()
    print(str(sum_of_errors/np_X_pred_on_test.shape[0]) + ' on test')

    modLSTM = LSTM_point_process(in_size, out_size, h_dim=32, n_layer=3, drop = 0.1)
    modLSTM.fit(X, time, y)

    X_pred_on_training_lstm = modLSTM.predict(X)
    np_X_pred_on_training_lstm = X_pred_on_training_lstm.detach().numpy()
    np.sum(np_X_pred_on_training_lstm<0.5)
    sum_of_errors = (y_train - np_X_pred_on_training_lstm).sum()
    print(str(sum_of_errors/np_X_pred_on_training_lstm.shape[0]) + ' on training')




