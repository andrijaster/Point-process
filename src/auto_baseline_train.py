import pickle
from pathlib import Path

import pickle
import time
from pathlib import Path

import pandas as pd
import torch
from v2 import BaseBaselineTraining as bb_train
from v2.FCNPointProcess import FCNPointProcess
from v2.PoissonTPP import PoissonTPP
from v2.PoissonPolynomialTPP import PoissonPolynomialTPP
from v2.PoissonPolynomialFirstOrderTPP import PoissonPolynomialFirstOrderTPP
from v2.SelfCorrectingTPP import SelfCorrectingTPP
from v2.HawkesSumGaussianTPP import HawkesSumGaussianTPP
from v2.HawkesTPP import HawkesTPP


if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    project_dir = str(Path(__file__).parent.parent)

    data_folder = f"{project_dir}/data/autoput/prepared/"  # os.environ['DATA_FOLDER']
    dataset_path = "stan1_traka1_7-17_04072017.csv"  # os.environ["TRAINING_DATASET"]
    data = pd.read_csv(data_folder+dataset_path)
    data.date1 = pd.to_datetime(data.date1)
    train_data = data[data.date1.dt.hour <= 13]
    test_data = data[data.date1.dt.hour > 13]

    test_data.loc[:, 'date1_ts'] = test_data.loc[:, 'date1_ts'] - test_data.loc[:, 'date1_ts'].min()
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
    models_to_evaluate = [
        {'model': HawkesTPP, 'learning_param_map': learning_param_map},
        {'model': HawkesSumGaussianTPP, 'learning_param_map': learning_param_map}
        # {'model': FCNPointProcess, 'learning_param_map': learning_param_map},
        # {'model': PoissonTPP, 'learning_param_map': learning_param_map},
        # {'model': PoissonPolynomialTPP, 'learning_param_map': learning_param_map},
        # {'model': PoissonPolynomialFirstOrderTPP, 'learning_param_map': learning_param_map},
        # TODO: {'model': SelfCorrectingTPP, 'learning_param_map': learning_param_map},
        # {'model': PoissonPolynomialTPP, 'learning_param_map': learning_param_map}
    ]
    print(f'Train size: {str(train_time.shape[1])}, test size: {str(test_time.shape[1])} ('
          f'{round((test_time.shape[1] / (train_time.shape[1] + test_time.shape[1])), 2)} %).')

    in_size = 5
    out_size = 1
    no_epochs = 2000
    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'training_time',
                                          'nr_of_reinit', 'loss_on_train', 'loss_on_test'])

    for model_definition in models_to_evaluate:
        for params in model_definition['learning_param_map']:
            model = None
            counter = 0

            while not model:
                counter += 1
                if model_definition['model'].__name__ == 'FCNPointProcess':
                    model = model_definition['model'](in_size+1, out_size, dropout=0.1)
                else:
                    model = model_definition['model']()
                model_name = f"autoput-040717-{type(model).__name__}-{params['learning_rate']}-{params['rule']}"

                print(f"{counter}. Starting to train a model: {model_name}")
                t0 = time.time()
                model = bb_train.fit(model, train_time, test_time, in_size, lr=params['learning_rate'],
                                         no_epoch=no_epochs, no_steps=params['no_step'], method=params['rule'], log_epoch=10,
                                         figpath=f"{project_dir}/img/autoput/{model_name}.png")
                if model:
                    loss_on_train = bb_train.evaluate(model, train_time, in_size, method=params['rule'])
                    loss_on_test = bb_train.evaluate(model, test_time, in_size, method='Trapezoid')
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
                    model_filepath = f"{project_dir}/models/autoput/{model_name}.torch"
                    pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)
    evaluation_df.to_csv(f"{project_dir}/results/autoput_baselines_scores_040717_{str(learning_param_map[0]['learning_rate'])}.csv",
                         index=False)
