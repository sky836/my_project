import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR as VARModel
from statsmodels.tsa.api import ARIMA
import torch.nn as nn

from data_provider.data_factory import data_provider
from utils.metrics import metric


class VAR(nn.Module):
    """
    VAR class.

    This class encapsulates a process of using VAR models for time series prediction.
    """

    def __init__(self, lags=13):
        super(VAR, self).__init__()
        self.scaler = StandardScaler()
        self.lags = lags
        self.results = None

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "VAR"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        return {}

    def forecast_fit(
        self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
    ):
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        self.scaler.fit(train_data.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data.values),
            columns=train_data.columns,
            index=train_data.index,
        )
        model = VARModel(train_data_value)
        self.results = model.fit(self.lags)

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :return: An array of predicted results.
        """
        train = pd.DataFrame(
            self.scaler.transform(series.values),
            columns=series.columns,
            index=series.index,
        )
        z = self.results.forecast(train.values, steps=horizon)

        predict = self.scaler.inverse_transform(z)

        return predict


# TODO: VAR_model is kept for backward compatibility, remove all references to VAR_model in the future
VAR_model = VAR(lags=6)
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
train_data = data[:num_train]
# VAR_model.forecast_fit(train_data)

test_data = data[-num_test:]
maes, rmses, mapes = [], [], []
l = len(test_data)
num_nodes = geo.shape[0]
num_feats = data.shape[-1]
for i in range(l-6):
    s_begin = i
    s_end = s_begin + 6
    r_begin = s_end
    r_end = r_begin + 1
    x = test_data[s_begin:s_end].reshape(-1)
    y = test_data[r_begin:r_end].reshape(-1)
    lx = x.shape[0]
    ly = y.shape[0]
    ARIMA_model = ARIMA(x, order=(6, 2, 6))
    ARIMA_model = ARIMA_model.fit()
    predicts = ARIMA_model.forecast(steps=ly)
    predicts, y = torch.tensor(predicts).clone().detach(), torch.tensor(y).clone().detach()
    mae, rmse, mape = metric(predicts, y, 0.0, 10)
    print(mae)
    if mae > 100:
        print(i)
    maes.append(mae.item())
    rmses.append(rmse.item())
    mapes.append(mape.item())
    # predicts = VAR_model.forecast(horizon=1, series=x)

print('============================================================================')
print(f'mae:{np.average(maes)}, rmse:{np.average(rmses)}, mape:{np.average(mapes)}')

