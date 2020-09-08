import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
from v2 import BaseTraining as b_train
# from v2 import BaseBaselineTraining as b_train
from pathlib import Path
from generated_with_const_train import get_interevents, expectation_of_lambda_per_second, expectation_of_lambda_as_mean_of_interevents


if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    project_dir = str(Path(__file__).parent.parent)

    data_folder = project_dir+'/data/skijasi/'
    dataset_path = 'ski_kg_2005-2020.csv'
    # dataset_path = 'zh_hb_main_station-24-25.082020.csv'
    # model_filename = project_dir + '/models/geoloc/zh_main_station-240820-GRUPointProcess-0.01-Euler.torch'
    # model_filename = project_dir + '/models/geoloc/zh_main_station-240820-FCNPointProcess-0.01-Euler.torch'
    # model_filename = project_dir + '/models/geoloc/zh_main_station-240820-PoissonPolynomialFirstOrderTPP-0.001-Euler.torch'
    model_filename = project_dir + '/models/ski/ski-kg-LSTMPointProcess-0.01-Euler.torch'

    data = pd.read_csv(data_folder+dataset_path)
    in_size = 5

    # train_data = data[data.day == 24]
    # test_data = data[data.day == 25]
    train_data = data[data.godina == 2018]
    test_data = data[data.godina == 2019]
    test_data.loc[:, 'date1_ts'] = test_data.loc[:, 'date1_ts'] - test_data.loc[:, 'date1_ts'].min()
    train_time = torch.tensor(train_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)

    train_list = list(train_data.date1_ts.values)
    test_list = list(test_data.date1_ts.values)

    train_interevents = get_interevents(train_list)
    print(f"[Train] Empirical lambda from sequence: "
          f"1.) {expectation_of_lambda_per_second(train_list)}"
          f"\t2.) {expectation_of_lambda_as_mean_of_interevents(train_interevents)}")

    test_interevents = get_interevents(test_list)
    print(f"[Test] Empirical lambda from sequence: "
          f"1.) {expectation_of_lambda_per_second(test_list)}"
          f"\t2.) {expectation_of_lambda_as_mean_of_interevents(test_interevents)}")

    loaded_model = pickle.load(open(model_filename, 'rb'))
    loss_on_test = b_train.evaluate(loaded_model, test_time, in_size, method='Trapezoid')

    predicted_lambdas = b_train.predict(loaded_model, test_time, in_size)
    print(f"Predicted lambda mean: {1/predicted_lambdas.mean()}")

    plt.plot(test_time.data.numpy().flatten(), predicted_lambdas.data.numpy().flatten(), 'g.')
    plt.show()




