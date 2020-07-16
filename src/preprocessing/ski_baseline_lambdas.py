
import pandas as pd
from src.Pytorch.model.Poisson import Poisson
import torch
import math
import numpy as np

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df1 = pd.read_csv('../data/skijasi/karaman_greben.csv', header=None)
    OBSERVED_DAY = '2007-02-20'

    # datetime parsing
    df1['date'] = pd.to_datetime(df1.iloc[:, 4])
    one_day_df = df1[df1.date.dt.strftime("%Y-%m-%d") == OBSERVED_DAY]

    one_day_df['date1'] = one_day_df['date'] + pd.to_timedelta(np.tile(np.arange(0, 15),
                                                                        math.ceil(one_day_df.shape[0]/15))[:one_day_df.shape[0]], unit='s')
    one_day_df = one_day_df.drop_duplicates('date1', keep='first')
    one_day_df['date1_ts'] = one_day_df.date1.astype('int') / 10**9
    one_day_df['date1_ts'] = one_day_df['date1_ts'] - one_day_df['date1_ts'].min()
    one_day_df = one_day_df.sort_values(by='date1_ts')

    learning_param_map = {'rule': 'Analytical', 'no_step': 10, 'learning_rate': 0.001}
    models_to_evaluate = {'model': Poisson(), 'learning_param_map': learning_param_map}

    print(f'Observed day: {OBSERVED_DAY}. Number of events: {str(one_day_df.shape[0])}')

    in_size = 5
    out_size = 1
    learning_rate = 0.01
    epochs = 500
    evaluation_df = pd.DataFrame(columns=['model_name', 'hour', 'lambda', 'loss_on_train'])
    # evaluation_df = pd.read_csv(f'../results/skijasi_baseline_scores-{OBSERVED_DAY}.csv')
    active_hours = [f'0{str(h)}' if len(str(h))==1 else str(h) for h in np.arange(8,18)]

    # for h in active_hours:
    #     hour_df = one_day_df[one_day_df.date1.dt.strftime('%H') == h]
    #     hour_df['date1_ts'] = hour_df.date1_ts - hour_df.date1_ts.min()
    #     print(f'Observed hour: {h}. Number of events: {str(hour_df.shape[0])}')
    #     time = torch.tensor(hour_df.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    #
    #     if time.shape[1] > 0:
    #         model = models_to_evaluate['model']
    #         model.fit(time, epochs, learning_param_map['learning_rate'], in_size,
    #                   learning_param_map['no_step'], None, learning_param_map['rule'], log_epoch=10)
    #         loss_on_train = model.evaluate(time, in_size)
    #         print(f"Model: {h} hour. Loss on train: {str(loss_on_train.data.numpy())}")
    #         evaluation_df.loc[len(evaluation_df)] = ['Poisson', h, model.model(time, []), loss_on_train.data.numpy()[0]]
    #
    # print(evaluation_df)
    # evaluation_df.to_csv(f'../results/skijasi_lambdas-{OBSERVED_DAY}.csv', index=False)




