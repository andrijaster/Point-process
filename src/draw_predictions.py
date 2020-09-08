import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
from v2 import BaseTraining
from v2 import BaseBaselineTraining
from pathlib import Path


if __name__ == "__main__":

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    project_dir = str(Path(__file__).parent.parent)

    data_folder = project_dir+'/data/geoloc/'
    dataset_path = 'zh_hb_main_station-24-25.082020.csv'  # os.environ["TRAINING_DATASET"]
    model_filename = project_dir + '/models/geoloc/ski_kg_2018_-HawkesTPP-0.001-Euler.torch'
    data = pd.read_csv(data_folder+dataset_path)
    in_size = 5

    train_data = data[data.day == 24]
    test_data = data[data.day == 25]
    test_data.loc[:, 'date1_ts'] = test_data.loc[:, 'date1_ts'] - test_data.loc[:, 'date1_ts'].min()
    train_time = torch.tensor(train_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)

    loaded_model = pickle.load(open(model_filename, 'rb'))
    loss_on_test = BaseBaselineTraining.evaluate(loaded_model, test_time, in_size, method='Trapezoid')

    predicted_lambdas = BaseBaselineTraining.predict(loaded_model, test_time, in_size)

    plt.plot(test_time.data.numpy().flatten(), predicted_lambdas.data.numpy().flatten(), 'g.')
    plt.show()




