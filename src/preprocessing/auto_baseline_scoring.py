
import pandas as pd
from src.Pytorch.model.SelfCorrectingProcess import SelfCorrectingProcess
import torch
import pickle

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df1 = pd.read_csv('../data/autoput/012017_bgd-nis_1.csv', sep=';', header=None)
    df1.drop(labels=[0, 3, 5], axis=1, inplace=True)
    kolone = {1: 'UlazStan', 2: 'UlazTraka', 4: 'date1', 6: 'IzlazStan',
              7: 'IzlazTraka', 8: 'date2', 9: 'Kategorija', 10: 'tag'}
    df1.rename(columns=kolone, inplace = True)
    DATETIME_COLUMN = 'date1'
    LOCATION_CONDITION = 'UlazStan == 1 and UlazTraka == 3'
    OBSERVED_DAY = '2017-01-01'
    location_result = df1.query(LOCATION_CONDITION)

    # datetime parsing
    location_result.date1 = pd.to_datetime(location_result.date1)
    one_day_location = location_result[location_result.date1.dt.strftime("%Y-%m-%d") == OBSERVED_DAY]
    one_day_location['date1_ts'] = one_day_location.date1.astype('int') / 10**9
    one_day_location['date1_ts'] = one_day_location['date1_ts'] - one_day_location['date1_ts'].min()
    one_day_location = one_day_location.sort_values(by='date1_ts')
    time = torch.tensor(one_day_location.date1_ts.values[:1000]).type('torch.FloatTensor').reshape(1, -1, 1)

    learning_param_map = [
        {'rule': 'Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Implicit Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Trapezoid', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Simpsons', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Gaussian_Q', 'no_step': 10, 'learning_rate': 0.001}
    ]
    analytical_definition = [{'rule': 'Analytical', 'no_step': 10, 'learning_rate': 0.001}]
    models_to_evaluate = [
        # {'model': Poisson(), 'learning_param_map': learning_param_map+analytical_definition}
        # {'model': PoissonPolynomial(), 'learning_param_map': learning_param_map+analytical_definition},
        # {'model': PoissonPolynomialFirstOrder(), 'learning_param_map': learning_param_map+analytical_definition},
        # {'model': Hawkes(), 'learning_param_map': learning_param_map+analytical_definition},
        # {'model': HawkesSumGaussians(), 'learning_param_map': learning_param_map}  # ,
        {'model': SelfCorrectingProcess(), 'learning_param_map': learning_param_map}
    ]

    print(f'Observed location: {LOCATION_CONDITION}, observed day: {OBSERVED_DAY}. '
          f'Number of events: {str(time.shape[1])}')

    in_size = 5
    out_size = 1
    learning_rate = 0.001
    epochs = 50
    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'loss_on_train'])
    evaluation_df = pd.read_csv('../results/jan_autoput_baseline_scores.csv')

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
            model_filepath = f"../models/auto-{OBSERVED_DAY}-{type(model).__name__}-{params['rule']}.torch"
            pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)
    # evaluation_df.to_csv('../results/jan_autoput_baseline_scores.csv', index=False)




