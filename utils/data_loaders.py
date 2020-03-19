import os
import numpy as np
import pandas as pd


def load_clc_db_records(path, feds_data=None):
    """Load all CLC futures data into dict. One entry per asset. Path specifies location of .csv files"""

    if feds_data is not None:
        short_rate = pd.read_csv(feds_data)
        short_rate['DATE'] = pd.to_datetime(short_rate['DATE'], format='%Y-%m-%d')
        feds_label = short_rate.columns[-1]

    data = {}
    files = [file for file in os.listdir(path) if '.csv' in file.lower()]
    for file in files:
        asset = file[:-4].split('_')[0]
        data[asset] = pd.read_csv(os.path.join(path, file), names=['Date', 'Open', 'High', 'Low', 'Settle', 'Volume', 'Open_Interest'])
        data[asset]['Date'] = pd.to_datetime(data[asset]['Date'], format='%m/%d/%Y')
        if feds_data is not None:
            data[asset] = data[asset].merge(short_rate, how='left', left_on='Date', right_on='DATE')
            data[asset][feds_label].ffill(inplace=True)
            data[asset][feds_label] /= 100
            data[asset]['Short_Rate_Daily'] = (1 + data[asset][feds_label]) ** (1 / 252) - 1
            data[asset]['Short_Rate_Annual'] = data[asset][feds_label]
            del data[asset][feds_label]
            del data[asset]['DATE']

    return data


if __name__ == '__main__':

    clc_data = load_clc_db_records('../data/clc/rev/', '../data/DFF.csv')
    print(list(clc_data.values())[-1])
