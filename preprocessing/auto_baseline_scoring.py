
import pandas as pd
from Pytorch.model.Poisson import Poisson
from Pytorch.model.PoissonPolynomial import PoissonPolynomial
from Pytorch.model.PoissonPolynomialFirstOrder import PoissonPolynomialFirstOrder
from Pytorch.model.Hawkes import Hawkes
from Pytorch.model.HawkesSumGaussians import HawkesSumGaussians
from Pytorch.model.SelfCorrectingProcess import SelfCorrectionProcess
import torch


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.read_csv("../data/autoput/012017_bg-nis-stan3-traka1-time_df.csv")

    learning_param_map = [
        {'rule': 'Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Implicit Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Trapezoid', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Simpsons', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Gaussian_Q', 'no_step': 10, 'learning_rate': 0.001}
    ]
    analytical_definition = [{'rule': 'Analytical', 'no_step': 10, 'learning_rate': 0.001}]
    models_to_evaluate = [
        {'model': PoissonPolynomial(), 'learning_param_map': learning_param_map+analytical_definition},
        {'model': PoissonPolynomialFirstOrder(), 'learning_param_map': learning_param_map+analytical_definition},
        {'model': Hawkes(), 'learning_param_map': learning_param_map+analytical_definition},
        {'model': HawkesSumGaussians(), 'learning_param_map': learning_param_map},
        {'model': SelfCorrectionProcess(), 'learning_param_map': learning_param_map}
    ]

    events_df = df.query('target == 1').reset_index()
    time = torch.tensor(events_df['num'].values[:3000]).type('torch.FloatTensor').reshape(1, -1, 1)
    print(f'Shape of full df: {str(df.shape[0])}, shape of events: {str(time.shape[1])}. '
          f'Percentage: {str(time.shape[1]/df.shape[0])}')

    in_size = 5
    out_size = 1
    learning_rate = 0.001
    epochs = 50
    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'loss_on_train'])
    for model_definition in models_to_evaluate:
        for params in model_definition['learning_param_map']:
            model = model_definition['model']
            model.fit(time, epochs, params['learning_rate'], in_size,
                      params['no_step'], None, params['rule'], log_epoch=10)

            loss_on_train = model.evaluate(time, in_size)
            print(f"Model: {type(model).__name__}. Loss on train: {str(loss_on_train.data.numpy())}")
            evaluation_df.loc[len(evaluation_df)] = [type(model).__name__,
                                                     params['rule'],
                                                     params['no_step'],
                                                     params['learning_rate'],
                                                     loss_on_train.data.numpy()[0]]

    print(evaluation_df)
    evaluation_df.to_csv('../results/jan_autoput_baseline_scores.csv', index=False)




