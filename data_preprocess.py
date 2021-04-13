import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

def aggregate_data():
    """Loads data and creates single file with data"""
    data_path = '/data/cloud_data/AzurePublicDataset2019/processed_data/univariate_data/depl_ANu_all/'
    data_frame = pd.DataFrame()
    filenames = list()
    max_list = list()
    col_names = ['timestamp', 'avgcpu', 'mincpu', 'maxcpu']

    for (_, _, files) in os.walk(data_path):
        filenames.extend(files)
        break

    for ind, f in enumerate(filenames):
        df = pd.read_csv(data_path + f,  delimiter=',', usecols=col_names)
        if ind < 1:
            data_frame = time_encoder(df)
        data_frame[col_names[1] + str(ind)] = df[col_names[1]]
        data_frame[col_names[2] + str(ind)] = df[col_names[2]]
        data_frame[col_names[3] + str(ind)] = df[col_names[3]]
        max_list.append(col_names[3] + str(ind))

    data_frame['maxmaxcpu'] = data_frame[max_list].max(axis=1)

    min_max_scaler = MinMaxScaler()
    data_frame[data_frame.columns] = min_max_scaler.fit_transform(data_frame[data_frame.columns])

    dump(min_max_scaler, data_path + 'min_max_scaler.joblib')

    # Total of 8627
    row_lines = len(data_frame)
    train_lines = int(0.7*row_lines)
    val_lines = int(0.2*row_lines)
    test_start = train_lines + val_lines

    train_df = data_frame.iloc[:train_lines,:]
    val_df = data_frame.iloc[train_lines+1:test_start, :]
    test_df = data_frame.iloc[test_start +1:, :]

    train_df.to_csv(data_path + 'train_data.csv', index=False)
    val_df.to_csv(data_path + 'val_data.csv', index=False)
    test_df.to_csv(data_path + 'test_data.csv', index=False)

    return True


def time_encoder(data_frame):
    """Converts the timestamp column from the Azure DF into an hourly and daily encoded timestamp"""
    df = pd.DataFrame()
    sec_in_h = 3600
    sec_in_d = 3600 * 24
    df['hourly_s'] = np.sin(2 * np.pi * data_frame['timestamp'] / sec_in_h)
    df['daily_s'] = np.sin(2 * np.pi * data_frame['timestamp'] / sec_in_d)
    df['hourly_c'] = np.cos(2 * np.pi * data_frame['timestamp'] / sec_in_h)
    df['daily_c'] = np.cos(2 * np.pi * data_frame['timestamp'] / sec_in_d)

    return df

if __name__ == '__main__':
    done = aggregate_data()