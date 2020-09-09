import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
# from v2 import BaseTraining as b_train
from v2 import BaseBaselineTraining as b_train
from pathlib import Path
from generated_with_const_train import get_interevents, expectation_of_lambda_per_second, expectation_of_lambda_as_mean_of_interevents

from generated_with_step_train import generate_mixed_process_based_on_lambda


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
    in_size = 100
    out_size = 1

    model_filename = project_dir + "/models/dummy/generated_exp_step-HawkesTPP-0.01-(50, 5, 50).torch"
    loaded_model = pickle.load(open(model_filename, 'rb'))
    loss_on_test = b_train.evaluate(loaded_model, test_time, in_size, method='Trapezoid')

    predicted_lambdas = b_train.predict(loaded_model, test_time, in_size)
    print(f"Predicted lambda mean: {1/predicted_lambdas.mean()}")

    print(f"ls: {1/expectation_of_lambda_per_second(train_events)}  , {predicted_lambdas.mean()}")

    plt.plot(test_time.data.numpy().flatten(), predicted_lambdas.data.numpy().flatten(), 'g.')
    plt.show()

    for i, p in enumerate(loaded_model.parameters()):
        if i == 0:
            alpha = p.detach().numpy().flatten()[0]
        else:
            mu = p.detach().numpy().flatten()[0]
    print(f"mu={mu}, alpha={alpha}")




