import pandas as pd


def read_and_prepare_auto_data(filepath, columns_to_save=['UlazStan', 'UlazTraka', 'date1']):
    df1 = pd.read_csv(filepath, sep=';', header=None)
    df1.drop(labels=[0, 3, 5], axis=1, inplace=True)
    column_names = {1: 'UlazStan', 2: 'UlazTraka', 4: 'date1', 6: 'IzlazStan',
              7: 'IzlazTraka', 8: 'date2', 9: 'Kategorija', 10: 'tag'}
    df1.rename(columns=column_names, inplace=True)
    return df1[columns_to_save]


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    data_folder = '../../data/autoput/prepared/'
    df = pd.read_csv(data_folder+'stan1_traka1_012017.csv')

    DATETIME_COLUMN = 'date1'

    df.date1 = pd.to_datetime(df.date1)
    df['date1_ts'] = df.date1.astype('int') / 10**9
    df['date1_ts'] = df['date1_ts'] - df['date1_ts'].min()
    df = df.sort_values(by='date1_ts')

    week_df = df[(df.date1.dt.day < 8) & (df.date1.dt.month == 1)]
    week_df.to_csv(data_folder+'stan1_traka1_01-07.012017.csv', index=False)
