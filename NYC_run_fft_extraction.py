import numpy as np
# from numpy import fft
import pandas as pd
# from scipy import signal as sig
# from cmath import phase
# import math
from fft_utils import FFTFeatureExtractor
import logging
# import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from argparse import ArgumentParser

# mpl.rcParams['figure.figsize'] = (8, 6)
# mpl.rcParams['axes.grid'] = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--num_days', help='Number of days to visualize into the future', type=int)
    args = parser.parse_args()
    print(args.num_days)
    day = 24*60

    # NYC taxi dataset
    train_df = pd.read_csv('../sample-data/train/train.csv')
    # test_df = pd.read_csv('../sample-data/test/test.csv')

    train_df.pickup_datetime = pd.to_datetime(train_df.pickup_datetime)
    train_df.dropoff_datetime = pd.to_datetime(train_df.dropoff_datetime)
    train_df.dropoff_datetime = pd.to_datetime(train_df.dropoff_datetime)
    train_df.store_and_fwd_flag = train_df.store_and_fwd_flag.apply(lambda x: 1 if x=='Y' else 0)

    ts_train = train_df.groupby(pd.Grouper(key='pickup_datetime', freq='T'))[['passenger_count','trip_duration']].sum().reset_index()
    ts_train['time_min'] = (ts_train['pickup_datetime'] - min(ts_train['pickup_datetime'])).dt.total_seconds()/60
    ts_train['time_sec'] = (ts_train['pickup_datetime'] - min(ts_train['pickup_datetime'])).dt.total_seconds()
    date_time = ts_train.pop('pickup_datetime')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    ts_train['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    ts_train['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    ts_train['pass_count_standardized'] = (ts_train['passenger_count'] - np.mean(ts_train['passenger_count'])) / np.std(ts_train['passenger_count'])
    
    fft = FFTFeatureExtractor(ts_train['pass_count_standardized'], time_series=date_time)
    plt.show()
    fft.fft_transform(freqlim_max=.005, timelim_max=48*60)
    # print(ts_train.head())
    x = fft.frequency_table_viewer()
    # print(x)
    filtered_residuals = fft.ifft_transform()
    w=ts_train.join(pd.DataFrame(filtered_residuals.real)).rename(columns={0:'filtered_residuals'})
    print(w)

    X = w[['time_min','filtered_residuals']]
    y = w['pass_count_standardized']

    modelNew = LinearRegression()
    modelNew.fit(X,y)

    y_pred = modelNew.predict(X)

    N = 24 * 60 * args.num_days

    plt.figure(figsize=(10,4))
    plt.plot(X['time_min'][:N], y[:N], linewidth=1, label='Original Signal')
    plt.plot(X['time_min'][:N], y_pred[:N], linewidth=1, label='Predicted Signal')
    plt.legend(loc='upper right')
    plt.suptitle('First {} Days'.format(int(N/24/60)))
    plt.grid()
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    print()

    fft.fourier_terms_df_creator()
    decomposedResult = fft.decompose_df_into_pure_freq(signal=ts_train['pass_count_standardized'], time_min= ts_train['time_min']  )
    decomposedResult['FT_All_Std'] = (decomposedResult['FT_All'] - np.mean(decomposedResult['FT_All'])) / np.std(decomposedResult['FT_All'])
    decomposedResult['pass_count_std-FT_All_Std'] = (decomposedResult['pass_count_standardized'] - decomposedResult['FT_All_Std'])

    print("Mean and standard deviation of standardized passenger count")
    print(np.mean(decomposedResult['pass_count_standardized']), np.std(decomposedResult['pass_count_standardized']))

    print("Mean and standard deviation of standardized passenger count subtract standardized total frequency")
    print(np.mean(decomposedResult['pass_count_std-FT_All_Std']), np.std(decomposedResult['pass_count_std-FT_All_Std']))
    print("Mean and standard deviation of Inverse FTT-transformed peaks of FFT-transformed standardized passenger counts")
    print(np.mean(filtered_residuals), np.std(filtered_residuals))

    plt.figure(figsize=(10,4))
    # plt.plot(X['time_min'][:N], y[:N], linewidth=1, label='Original Signal')
    plt.plot(X['time_min'][:N], decomposedResult['pass_count_std-FT_All_Std'][:N], linewidth=1, label='Predicted Signal')
    plt.legend(loc='upper right')
    plt.suptitle('First {} Days'.format(int(N/24/60)))
    plt.grid()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    print()

    print(decomposedResult.head())

if __name__ == "__main__":
    main()