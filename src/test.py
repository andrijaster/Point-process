import matplotlib.pyplot as plt
import numpy as np
import torch

from v2 import BaseTraining
from v2.FCNPointProcess import FCNPointProcess
from v2.GRUPointProcess import GRUPointProcess
from v2.LSTMPointProcess import LSTMPointProcess
from v2.RNNPointProcess import RNNPointProcess
from pathlib import Path


if __name__ == "__main__":
    time = np.abs(100*np.random.rand(100))
    time.sort()
    in_size = 5
    out_size = 1
    time[0] = 0
    project_dir = str(Path(__file__).parent.parent)
    lr = 0.1

    train_time, test_time = time[:80], time[80:]
    train_time = torch.tensor(train_time).type('torch.FloatTensor').reshape(1, -1, 1)
    test_time = torch.tensor(test_time).type('torch.FloatTensor').reshape(1, -1, 1)

    mod = FCNPointProcess(in_size+1, out_size, dropout=0.0)
    model_name = 'lstm_' + str(lr)

    mod = BaseTraining.fit(mod, train_time, test_time, in_size, lr=0.01, no_epoch=50, method="Euler",
                           figpath=f"{project_dir}/img/{model_name}.png")
    loss_on_train = BaseTraining.evaluate(mod, train_time, in_size, method='Trapezoid')
    print(loss_on_train)

    # pickle.dump(mod, open('../../models/lstm_test.torch', 'wb'))
