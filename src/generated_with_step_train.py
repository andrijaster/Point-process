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
from v2 import BaseBaselineTraining as bb_train
from v2.FCNPointProcess import FCNPointProcess
from v2.PoissonTPP import PoissonTPP
from v2.PoissonPolynomialTPP import PoissonPolynomialTPP
from v2.PoissonPolynomialFirstOrderTPP import PoissonPolynomialFirstOrderTPP
from v2.SelfCorrectingTPP import SelfCorrectingTPP
from v2.HawkesSumGaussianTPP import HawkesSumGaussianTPP
from v2.IntereventRegressorTPP import IntereventRegressorTPP
from v2.HawkesTPP import HawkesTPP
from generated_with_const_train import generate_process_with_exponential_interevent, get_interevents, \
    expectation_of_lambda_per_second, expectation_of_lambda_as_mean_of_interevents, plot_results


def generate_mixed_process_based_on_lambda(first_event_time, lambdas, n_events):
    events = [first_event_time]
    for i, l in enumerate(lambdas):
        events_part = generate_process_with_exponential_interevent(first_event_time=events[-1]+5,
                                                                   lambda_gen=l,
                                                                   n_events=n_events[i])
        events = events + events_part
    return events


if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    project_dir = str(Path(__file__).parent.parent)

    l_generator = (50, 5, 50)
    train_events = generate_mixed_process_based_on_lambda(0, l_generator, (500, 500, 500))
    train_interevents = get_interevents(train_events)
    print(f"[Train] Lambda used for generator: {l_generator}. Empirical lambda from sequence: "
          f"1.) {expectation_of_lambda_per_second(train_events)}"
          f"\t2.) {expectation_of_lambda_as_mean_of_interevents(train_interevents)}")

    test_events = generate_mixed_process_based_on_lambda(0, l_generator, (500, 500, 500))
    test_interevents = get_interevents(test_events)
    print(f"[Test] Lambda used for generator: {l_generator}. Empirical lambda from sequence: "
          f"1.) {expectation_of_lambda_per_second(test_events)}"
          f"\t2.) {expectation_of_lambda_as_mean_of_interevents(test_interevents)}")

    train_time = torch.tensor(train_events).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_events).type('torch.FloatTensor').reshape(1, -1, 1)
    in_size = 50
    out_size = 1
    no_epochs = 1000

    learning_param_map = [
        {'rule': 'Euler', 'no_step': 3, 'learning_rate': 0.1}
    ]
    models_to_evaluate = [
        {'model': PoissonTPP, 'type': 'baseline', 'learning_param_map': learning_param_map},
        {'model': PoissonPolynomialTPP, 'type': 'baseline', 'learning_param_map': learning_param_map},
        {'model': IntereventRegressorTPP, 'type': 'baseline', 'learning_param_map': learning_param_map},
        {'model': HawkesTPP, 'type': 'baseline', 'learning_param_map': learning_param_map},
        {'model': HawkesSumGaussianTPP, 'type': 'baseline', 'learning_param_map': learning_param_map}
        # {'model': FCNPointProcess, 'type': 'nn', 'learning_param_map': learning_param_map}
        # {'model': RNNPointProcess, 'type': 'nn', 'learning_param_map': learning_param_map},
        # {'model': GRUPointProcess, 'type': 'nn', 'learning_param_map': learning_param_map},
        # {'model': LSTMPointProcess, 'type': 'nn', 'learning_param_map': learning_param_map},
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
                if model_definition['type'] == 'nn':
                    model = model_definition['model'](in_size+1, out_size)
                else:
                    model = model_definition['model']()
                model_name = f"generated_exp_step-{type(model).__name__}-{params['learning_rate']}-{l_generator}"

                print(f"{counter}. Starting to train a model: {model_name}")
                t0 = time.time()
                model = bb_train.fit(model, train_time, test_time, in_size, lr=params['learning_rate'],
                                     no_epoch=no_epochs, no_steps=params['no_step'], method=params['rule'], log_epoch=10,
                                     figpath=f"{project_dir}/img/dummy/{model_name}_train.png")

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

                    predicted_lambdas = bb_train.predict(model, test_time, in_size)

                    plot_results(model, test_time, train_interevents, test_interevents, predicted_lambdas, l=l_generator,
                                 figpath=f"{project_dir}/img/dummy/{model_name}.png")
                    model_filepath = f"{project_dir}/models/dummy/{model_name}.torch"
                    pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)


    evaluation_df.to_csv(f"{project_dir}/results/generated_exp_step_{str(learning_param_map[0]['learning_rate'])}_0.1.csv",
                         index=False)

