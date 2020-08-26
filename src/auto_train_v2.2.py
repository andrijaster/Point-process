import os
import pickle
import time
from pathlib import Path

import pandas as pd
import torch
from v2 import BaseTraining
from v2.GRUPointProcess import GRUPointProcess
from v2.LSTMPointProcess import LSTMPointProcess
from v2.RNNPointProcess import RNNPointProcess

if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    project_dir = str(Path(__file__).parent.parent)

    data_folder = os.environ['DATA_FOLDER']
    dataset_path = os.environ["TRAINING_DATASET"]
    data = pd.read_csv(data_folder+dataset_path)
    data.date1 = pd.to_datetime(data.date1)
    train_data = data[data.date1.dt.day <= 5]
    test_data = data[data.date1.dt.day > 5]
    test_data.loc[:, 'date1_ts'] = test_data.loc[:, 'date1_ts'] - test_data.loc[:, 'date1_ts'].min()
    train_time = torch.tensor(train_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    in_size = 5
    out_size = 1

    learning_param_map = [
        {'rule': 'Euler', 'no_step': 10, 'learning_rate': 0.001},
        # {'rule': 'Implicit Euler', 'no_step': 10, 'learning_rate': 0.01},
        # {'rule': 'Trapezoid', 'no_step': 10, 'learning_rate': 0.01},
        # {'rule': 'Simpsons', 'no_step': 10, 'learning_rate': 0.01},
        {'rule': 'Gaussian_Q', 'no_step': 10, 'learning_rate': 0.001}
    ]
    models_to_evaluate = [
        # {'model': FCNPointProcess(in_size+1, out_size, dropout=0.1), 'learning_param_map': learning_param_map},
        {'model': GRUPointProcess(in_size+1, out_size, dropout=0.0), 'learning_param_map': learning_param_map},
        {'model': LSTMPointProcess(in_size+1, out_size, dropout=0.0), 'learning_param_map': learning_param_map},
        {'model': RNNPointProcess(in_size+1, out_size, dropout=0.0), 'learning_param_map': learning_param_map}
    ]
    print(f'Train size: {str(train_time.shape[1])}, test size: {str(test_time.shape[1])}.')

    in_size = 5
    out_size = 1
    no_epochs = 500
    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'training_time', 'loss_on_train', 'loss_on_test'])

    for model_definition in models_to_evaluate:
        for params in model_definition['learning_param_map']:
            model = model_definition['model']
            model_name = f"autoput-012017-{type(model).__name__}-{params['learning_rate']}-{params['rule']}"

            print(f"Starting to train a model: {model_name}")
            t0 = time.time()

            model = BaseTraining.fit(model, train_time, test_time, in_size, lr=params['learning_rate'],
                                     no_epoch=no_epochs, no_steps=params['no_step'], method=params['rule'], log_epoch=10,
                                     figpath=f"{project_dir}/img/{model_name}.png")

            loss_on_train = BaseTraining.evaluate(model, train_time, in_size, method=params['rule'])
            loss_on_test = BaseTraining.evaluate(model, test_time, in_size, method='Trapezoid')
            print(f"Model: {model_name}. Loss on train: {str(loss_on_train.data.numpy().flatten()[0])}, "
                  f"loss on test: {str(loss_on_test.data.numpy().flatten()[0])}")
            evaluation_df.loc[len(evaluation_df)] = [model_name,
                                                     params['rule'],
                                                     params['no_step'],
                                                     params['learning_rate'],
                                                     str(round(time.time() - t0)),
                                                     loss_on_train.data.numpy().flatten()[0],
                                                     loss_on_test.data.numpy().flatten()[0]]
            model_filepath = f"models/autoput-012017/{model_name}.torch"
            pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)
    evaluation_df.to_csv(f"results/jan_autoput_scores_{str(learning_param_map[0]['learning_rate'])}.csv",
                         index=False)
