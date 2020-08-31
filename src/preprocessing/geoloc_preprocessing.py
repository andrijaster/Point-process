import pandas as pd
from pathlib import Path

from numpy.lib import format

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    project_dir = str(Path(__file__).parent.parent)
    data_folder = project_dir+'/../data/geoloc/'

    df = pd.read_csv(data_folder+'zh_hb_main_station-24-25.082020.csv')
    df['date'] = pd.to_datetime(df['ts'], unit='s')
    df['day'] = df.date.dt.day

    df['date1_ts'] = df['ts'] - df['ts'].min()
    df = df.sort_values(by='date1_ts')
    df = df.drop_duplicates('date1_ts', keep='first')

    df.to_csv(data_folder+'zh_hb_main_station-24-25.082020.csv', index=False)
