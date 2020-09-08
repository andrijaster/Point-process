import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    data = np.unique(np.sort(np.round(np.random.exponential(size=(1000,))*100)))
    sns.distplot(data.flatten())
    plt.show()

    train_data = data[:200]
    test_data = data[200:]
    train_time = torch.tensor(train_data).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_data).type('torch.FloatTensor').reshape(1, -1, 1)
    in_size = 50
    out_size = 1
    no_epochs=200
    learning_param_map = [
        {'rule': 'Euler', 'no_step': 10, 'learning_rate': 0.01},
        # {'rule': 'Implicit Euler', 'no_step': 10, 'learning_rate': 0.01},
        # {'rule': 'Trapezoid', 'no_step': 10, 'learning_rate': 0.01},
        # {'rule': 'Simpsons', 'no_step': 10, 'learning_rate': 0.01},
        # {'rule': 'Gaussian_Q', 'no_step': 10, 'learning_rate': 0.01}
    ]
    models_to_evaluate = [
        {'model': GRUPointProcess, 'learning_param_map': learning_param_map},
        {'model': LSTMPointProcess, 'learning_param_map': learning_param_map},
        {'model': RNNPointProcess, 'learning_param_map': learning_param_map}
    ]
    print(f'Train size: {str(train_time.shape[1])}, test size: {str(test_time.shape[1])} ('
          f'{round((test_time.shape[1] / (train_time.shape[1] + test_time.shape[1])), 2)} %).')

    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'training_time',
                                          'nr_of_reinit', 'loss_on_train', 'loss_on_test'])

    for model_definition in models_to_evaluate:
        for params in model_definition['learning_param_map']:
            model = None
            counter = 0

            while not model:
                counter += 1
                model = model_definition['model'](in_size+1, out_size, dropout=0.0)
                model_name = f"dummy_data-{type(model).__name__}-{params['learning_rate']}-{params['rule']}"

                print(f"{counter}. Starting to train a model: {model_name}")
                t0 = time.time()
                model = BaseTraining.fit(model, train_time, test_time, in_size, lr=params['learning_rate'],
                                         no_epoch=no_epochs, no_steps=params['no_step'], method=params['rule'], log_epoch=10,
                                         figpath=f"{project_dir}/img/dummy/{model_name}.png")
                if model:
                    loss_on_train = BaseTraining.evaluate(model, train_time, in_size, method=params['rule'])
                    loss_on_test = BaseTraining.evaluate(model, test_time, in_size, method='Trapezoid')
                    print(f"Model: {model_name}. Loss on train: {str(loss_on_train.data.numpy().flatten()[0])}, "
                          f"loss on test: {str(loss_on_test.data.numpy().flatten()[0])}")
                    evaluation_df.loc[len(evaluation_df)] = [type(model).__name__,
                                                             params['rule'],
                                                             params['no_step'],
                                                             params['learning_rate'],
                                                             str(round(time.time() - t0)),
                                                             str(counter),
                                                             loss_on_train.data.numpy().flatten()[0],
                                                             loss_on_test.data.numpy().flatten()[0]]
                    model_filepath = f"{project_dir}/models/dummy/{model_name}.torch"
                    pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)
    evaluation_df.to_csv(f"{project_dir}/results/dummy_data_{str(learning_param_map[0]['learning_rate'])}_0.1.csv",
                         index=False)
