import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import ARIMA
from utils.metrics import metric


data = pd.read_csv(r'../datasets/T-Drive/T-Drive.grid')

data_col = ['time', 'row_id', 'column_id', 'inflow', 'outflow']
data = data[data_col]
geo = pd.read_csv(r'../datasets/T-Drive/T-Drive.geo')
timesolts = list(data['time'][:int(data.shape[0] / geo.shape[0])])
# idx_of_timesolts = dict()
timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
timesolts = np.array(timesolts, dtype='datetime64[ns]')
# for idx, _ts in enumerate(timesolts):
#     idx_of_timesolts[_ts] = idx
feature_dim = len(data.columns) - 3
df = data[data.columns[-feature_dim:]]
len_time = len(timesolts)
data = []
for i in range(0, df.shape[0], len_time):
    data.append(df[i:i + len_time].values)
data = np.array(data, dtype=np.float64)
data = data.swapaxes(0, 1)
# data = data.reshape(data.shape[0], -1)
# data = pd.DataFrame(data)
num_train = round(len(data) * 0.7)
num_test = round(len(data) * 0.2)
train_data = data[:num_train]  # l, n, c
test_data = data[-num_test:]
maes, rmses, mapes = [], [], []
l = len(test_data)
num_nodes = geo.shape[0]
num_feats = data.shape[-1]

for i in range(num_nodes):
    for j in range(num_feats):
        x = train_data[:, i, j]
        y = test_data[:, i, j]
        ARIMA_model = ARIMA(x, order=(6, 2, 6))
        ARIMA_model = ARIMA_model.fit()
        predicts = ARIMA_model.forecast(steps=l)
        predicts, y = torch.tensor(predicts).clone().detach(), torch.tensor(y).clone().detach()
        mae, rmse, mape = metric(predicts, y, 0.0, 10)
        print(mae)
        maes.append(mae.item())
        rmses.append(rmse.item())
        mapes.append(mape.item())

print('============================================================================')
print(f'mae:{np.average(maes)}, rmse:{np.average(rmses)}, mape:{np.average(mapes)}')

