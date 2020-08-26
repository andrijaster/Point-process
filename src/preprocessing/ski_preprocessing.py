import pandas as pd
from pathlib import Path

from numpy.lib import format

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    project_dir = str(Path(__file__).parent.parent)
    data_folder = project_dir+'/../data/skijasi/'

    df = pd.read_excel(data_folder+'GSS2020.xlsx', sheet_name='Sheet1')

    # subset by location
    df = df.query("value==\'Karaman greben\'")

    DATETIME_COLUMN = 'vreme_nesrece'
    df['date1'] = pd.to_datetime(df[DATETIME_COLUMN], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df['date1_ts'] = df.date1.astype('int') / 10**9
    df['date1_ts'] = df['date1_ts'] - df['date1_ts'].min()
    df = df.sort_values(by='date1_ts')

    df = df[['ski_pass', 'godina', 'vreme_nesrece', 'value', 'date1', 'date1_ts']]
    df.to_csv(data_folder+'ski_kg_2005-2020.csv', index=False)
