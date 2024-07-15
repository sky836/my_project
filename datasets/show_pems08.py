import os

import numpy as np
from matplotlib import pyplot as plt

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


interval = 12*24*2
sample = 12*24
for i in range(0, x.shape[0], interval):
    for j in range(0, x.shape[1], 3):
        time_series = {}
        for k in range(3):
            time_series[j+k] = x[i:i+interval, j+k]
        showTimeseries(time_series, range(interval))

