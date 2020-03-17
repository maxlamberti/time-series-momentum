import os
import pandas as pd


def load_clc_db(path):
    """Load all CLC futures data. Path specifies location of .csv files"""

    data = {}
    files = [file for file in os.listdir(path) if '.csv' in file.lower()]
    for file in files:
        asset = file[:-4].split('_')[0]
        data[asset] = pd.read_csv(os.path.join(path, file), names=['Date', 'Open', 'High', 'Low', 'Settle', 'Volume', 'Open_Interest'])
        data[asset]['Date'] = pd.to_datetime(data[asset]['Date'], format='%m/%d/%Y')

    return data


if __name__ == '__main__':

    clc_data = load_clc_db('../data/clc/rev/')
    print(list(clc_data.values())[-1])
