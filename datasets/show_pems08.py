import os

import numpy as np
from matplotlib import pyplot as plt
import torch

data_name = 'PEMS03'
data = np.load(os.path.join(fr'{data_name}/', "data.npz"))["data"].astype(np.float32)
x = data[..., 0]


def showTimeseries(timeSeries, time):
    """
    timeSeries: {node: timeSeries}
    """
    plt.figure()

    for key in timeSeries.keys():
        print(timeSeries[key])
        plt.plot(timeSeries[key], label=key, linewidth=2)
    tick_positions = np.arange(len(time))
    plt.xticks(tick_positions[::sample], time[::sample], rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.title(data_name)
    # plt.ylim(-2, 75)
    plt.subplots_adjust(top=0.937, bottom=0.229)
    plt.legend()
    plt.grid()
    plt.show()

def showTimeseries_fft(timeSeries, time):
    """
    timeSeries: {node: timeSeries}
    """
    plt.figure()

    for key in timeSeries.keys():
        time_series = timeSeries[key]
        ffted = torch.fft.fft(torch.tensor(time_series), dim=-1)
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        print(key)
        print('ffted_real:', ffted_real)
        print('ffted_imag:', ffted_imag)
        plt.plot(ffted_real, label=key, linewidth=2)
    tick_positions = np.arange(len(time))
    plt.xticks(tick_positions[::sample], time[::sample], rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.title(data_name)
    # plt.ylim(-2, 75)
    plt.subplots_adjust(top=0.937, bottom=0.229)
    plt.legend()
    plt.grid()
    plt.show()


interval = 12
sample = 1
for i in range(0, x.shape[0], interval):
    for j in range(0, x.shape[1], 3):
        time_series = {}
        for k in range(3):
            time_series[j+k] = x[i:i+interval, j+k]
        showTimeseries(time_series, range(interval))
        showTimeseries_fft(time_series, range(interval))

