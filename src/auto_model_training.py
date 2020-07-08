
import pickle

import pandas as pd
import torch
import os

from Pytorch.model import FCN_point_process_all, GRU_point_process_all, LSTM_point_process_all, RNN_point_process_all

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    data_folder = os.environ['DATA_FOLDER']
    data = pd.read_csv(data_folder+os.environ["TRAINING_DATASET"])
    data.date1 = pd.to_datetime(data.date1)
    train_data = data[data.date1.dt.day <= 24]
    test_data = data[data.date1.dt.day > 24]
    train_time = torch.tensor(train_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    in_size = 5
    out_size = 1

    learning_param_map = [
        {'rule': 'Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Implicit Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Trapezoid', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Simpsons', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Gaussian_Q', 'no_step': 10, 'learning_rate': 0.001}
    ]
    analytical_definition = [{'rule': 'Analytical', 'no_step': 10, 'learning_rate': 0.001}]
    models_to_evaluate = [
        {'model': FCN_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map},
        {'model': GRU_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map},
        {'model': LSTM_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map},
        {'model': RNN_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map}
    ]

    print(f'Train size: {str(train_time.shape[1])}, test size: {str(test_time.shape[1])}.')

    in_size = 5
    out_size = 1
    epochs = 50
    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'loss_on_train', 'loss_on_test'])

    for model_definition in models_to_evaluate:
        for params in model_definition['learning_param_map']:
            model = model_definition['model']
            model.fit(train_time, test_time, in_size, no_epoch=epochs,
                      no_steps=params['no_step'], method=params['rule'], log_epoch=10)

            loss_on_train = model.evaluate(train_time, in_size, method='Trapezoid')
            loss_on_test = model.evaluate(test_time, in_size, method='Trapezoid')
            print(f"Model: {type(model).__name__}. Loss on train: {str(loss_on_train.data.numpy())},  "
                  f"loss on test: {str(loss_on_test.data.numpy())}")
            evaluation_df.loc[len(evaluation_df)] = [type(model).__name__,
                                                     params['rule'],
                                                     params['no_step'],
                                                     params['learning_rate'],
                                                     loss_on_train.data.numpy()[0],
                                                     loss_on_test.data.numpy()[0]]
            model_filepath = f"../models/autoput-012017/autoput-012017-{type(model).__name__}-{params['rule']}.torch"
            pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)
    evaluation_df.to_csv('../results/jan_autoput_scores_0.1.csv', index=False)




