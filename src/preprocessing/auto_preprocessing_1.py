
from os import walk

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

    files = []
    data_folder = '../../data/autoput/prepared/'
    for (dirpath, dirnames, filenames) in walk(data_folder):
        files.extend(filenames)
        break
    files = [file for file in files if '_bgd-nis_' in file]

    df = pd.DataFrame()
    for file in files:
        df = pd.concat([df, read_and_prepare_auto_data(data_folder+file)])

    DATETIME_COLUMN = 'date1'
    LOCATION_CONDITION = 'UlazStan == 1 and UlazTraka == 3'
    location_result = df.query(LOCATION_CONDITION)

    location_result.date1 = pd.to_datetime(location_result.date1)
    location_result['date1_ts'] = location_result.date1.astype('int') / 10**9
    location_result['date1_ts'] = location_result['date1_ts'] - location_result['date1_ts'].min()
    location_result = location_result.sort_values(by='date1_ts')

    location_result.to_csv(data_folder+'stan1_traka1_2017_full.csv', index=False)
    month_location_result = location_result[(location_result.date1.dt.year == 2017) & (location_result.date1.dt.month == 1)]
    month_location_result.to_csv(data_folder+'stan1_traka1_012017.csv', index=False)
