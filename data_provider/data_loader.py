import datetime
import os

import numpy as np
import pandas as pd
import torch

# from sklearn.preprocessing import StandardScaler
from utils.tools import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_h5(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, scale=True, time_to_feature=0):
        super(Dataset_h5, self).__init__()
        if size == None:
            self.seq_len = 12
            self.label_len = 1
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.time_to_feature = time_to_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data_file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_hdf(data_file_path)[:]  # [time_len, node_num]
        dates = df_raw.index.values.astype('datetime64')
        # print(dates)
        num_train = round(len(df_raw) * 0.7)
        num_test = round(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_y = df_raw.values

        if self.scale:
            train_data = df_raw[border1s[0]:border2s[0]].values
            self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
            print('mean:', train_data.mean())
            print('std:', train_data.std())
            # 对数据进行标准化
            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        if self.time_to_feature == 0:
            num_samples, num_nodes = df_raw.shape
            time_ind = (df_raw.index.values - df_raw.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list = [np.expand_dims(data, axis=-1), time_in_day]
            dow = df_raw.index.dayofweek
            dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)
            processed_data = np.concatenate(feature_list, axis=-1)
            processed_data_y = [np.expand_dims(data_y, axis=-1), time_in_day, dow_tiled]
            processed_data_y = np.concatenate(processed_data_y, axis=-1)
        elif self.time_to_feature == 1:
            feature_list = [np.expand_dims(data, axis=-1)]
            l, n = data.shape
            # 对时间进行编码，返回是一个编码后的矩阵，每一行对应一个时间，列为编码后的特征
            stamp = time_features(pd.to_datetime(df_raw.index.values), freq='T')
            # 进行转置，每一行对应一个特征，列为对应的时间
            stamp = stamp.transpose(1, 0)
            # print(stamp[:10])
            # print(stamp.shape)
            stamp_tiled = np.tile(stamp, [n, 1, 1]).transpose((1, 0, 2))
            feature_list.append(stamp_tiled)
            processed_data = np.concatenate(feature_list, axis=-1)
            processed_data_y = [np.expand_dims(data_y, axis=-1), stamp_tiled]
            processed_data_y = np.concatenate(processed_data_y, axis=-1)
        else:
            processed_data = data

        time_stamp = {'date': dates}
        df_stamp = pd.DataFrame(time_stamp)
        df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(columns=['date']).values

        self.data_x = processed_data[border1:border2]
        self.data_y = processed_data_y[border1:border2]
        self.data_stamp = data_stamp
        print(self.data_x[:10, 0])
        print(self.data_x.shape)
        print(type(self.data_x))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS04(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, scale=True, time_to_feature=1, type='flow'):
        super(Dataset_PEMS04, self).__init__()
        if size == None:
            self.seq_len = 12
            self.label_len = 1
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type
        self.set_type = type_map[flag]

        self.scale = scale
        self.time_to_feature = time_to_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data_file_path = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file_path)['data']
        if self.type == 'speed':
            data = data[..., 2]
        else:
            data = data[..., 0]

        num_train = round(len(data) * 0.7)
        num_test = round(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
            print('mean:', train_data.mean())
            print('std:', train_data.std())
            # 对数据进行标准化
            data = self.scaler.transform(data)
        else:
            data = data

        num_samples, num_nodes = data.shape
        if self.time_to_feature == 0:
            feature_list = [np.expand_dims(data, axis=-1)]
            steps_per_day = 288
            # numerical time_of_day
            tod = [i % steps_per_day / steps_per_day for i in range(data.shape[0])]
            tod = np.array(tod) - 0.5
            tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(tod_tiled)
            # numerical day_of_week
            dow = [(i // steps_per_day) % 7 for i in range(data.shape[0])]
            dow = np.array(dow) - 0.5
            dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)
            processed_data = np.concatenate(feature_list, axis=-1)
        elif self.time_to_feature == 1:
            steps_per_hour = 12
            steps_per_day = 24
            steps_per_month = 30
            steps_per_year = 12
            feature_list = [np.expand_dims(data, axis=-1)]
            moh = [i % steps_per_hour / (steps_per_hour - 1) for i in range(data.shape[0])]
            moh_tiled = np.tile(moh, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            hod = [(i // steps_per_hour) % steps_per_day / (steps_per_day - 1) for i in range(data.shape[0])]
            hod_tiled = np.tile(hod, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            dom = [(i // (steps_per_hour*steps_per_day)) % steps_per_month / (steps_per_month - 1) for i in range(data.shape[0])]
            dom_tiled = np.tile(dom, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            moy = [(i // (steps_per_hour*steps_per_day*steps_per_month)) % steps_per_year / (steps_per_year - 1) for i in range(data.shape[0])]
            moy_tiled = np.tile(moy, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5

            feature_list.append(moh_tiled)
            feature_list.append(hod_tiled)
            feature_list.append(dom_tiled)
            feature_list.append(moy_tiled)

            processed_data = np.concatenate(feature_list, axis=-1)
        else:
            processed_data = data

        self.data_x = processed_data[border1:border2]
        self.data_y = processed_data[border1:border2]
        self.data_stamp = processed_data[..., 1:]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len - self.label_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS08(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, scale=True, time_to_feature=1, type='flow'):
        if size == None:
            self.seq_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type
        self.set_type = type_map[flag]

        self.scale = scale
        self.time_to_feature = time_to_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data_file_path = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file_path)['data'].astype(np.float32)

        time = data[..., 1:]
        data = data[..., 0]

        num_train = round(len(data) * 0.6)
        num_test = round(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_y = data

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
            print('mean:', train_data.mean())
            print('std:', train_data.std())
            # 对数据进行标准化
            data = self.scaler.transform(data)
        else:
            data = data

        num_samples, num_nodes = data.shape
        if self.time_to_feature == 0:
            processed_data = np.concatenate([np.expand_dims(data, axis=-1), time], axis=-1)
            processed_datay = np.concatenate([np.expand_dims(data_y, axis=-1), time], axis=-1)
        elif self.time_to_feature == 1:
            steps_per_hour = 12
            steps_per_day = 24
            steps_per_month = 30
            steps_per_year = 12
            feature_list = [np.expand_dims(data[..., 0], axis=-1)]
            moh = [i % steps_per_hour / (steps_per_hour - 1) for i in range(data.shape[0])]
            moh_tiled = np.tile(moh, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            hod = [(i // steps_per_hour) % steps_per_day / (steps_per_day - 1) for i in range(data.shape[0])]
            hod_tiled = np.tile(hod, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            dom = [(i // (steps_per_hour*steps_per_day)) % steps_per_month / (steps_per_month - 1) for i in range(data.shape[0])]
            dom_tiled = np.tile(dom, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            moy = [(i // (steps_per_hour*steps_per_day*steps_per_month)) % steps_per_year / (steps_per_year - 1) for i in range(data.shape[0])]
            moy_tiled = np.tile(moy, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5

            feature_list.append(moh_tiled)
            feature_list.append(hod_tiled)
            feature_list.append(dom_tiled)
            feature_list.append(moy_tiled)

            processed_data = np.concatenate(feature_list, axis=-1)
            processed_datay = np.concatenate([np.expand_dims(data_y, axis=-1), moy_tiled, hod_tiled, dom_tiled, moy_tiled], axis=-1)
        else:
            processed_data = data

        self.data_x = processed_data[border1:border2]
        self.data_y = processed_datay[border1:border2]
        self.data_stamp = processed_data[..., 1:]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pretrain_Forecast(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, scale=True, time_to_feature=1, type='flow'):
        if size == None:
            self.seq_len = 12
            self.label_len = 12*24*7
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type
        self.set_type = type_map[flag]

        self.scale = scale
        self.time_to_feature = time_to_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data_file_path = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file_path)['data'].astype(np.float32)

        time = data[..., 1:]
        data = data[..., 0]

        num_train = round(len(data) * 0.6)
        num_test = round(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_y = data

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
            print('mean:', train_data.mean())
            print('std:', train_data.std())
            # 对数据进行标准化
            data = self.scaler.transform(data)
        else:
            data = data

        num_samples, num_nodes = data.shape
        if self.time_to_feature == 0:
            processed_data = np.concatenate([np.expand_dims(data, axis=-1), time], axis=-1)
            processed_datay = np.concatenate([np.expand_dims(data_y, axis=-1), time], axis=-1)
        elif self.time_to_feature == 1:
            steps_per_hour = 12
            steps_per_day = 24
            steps_per_month = 30
            steps_per_year = 12
            feature_list = [np.expand_dims(data[..., 0], axis=-1)]
            moh = [i % steps_per_hour / (steps_per_hour - 1) for i in range(data.shape[0])]
            moh_tiled = np.tile(moh, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            hod = [(i // steps_per_hour) % steps_per_day / (steps_per_day - 1) for i in range(data.shape[0])]
            hod_tiled = np.tile(hod, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            dom = [(i // (steps_per_hour*steps_per_day)) % steps_per_month / (steps_per_month - 1) for i in range(data.shape[0])]
            dom_tiled = np.tile(dom, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5
            moy = [(i // (steps_per_hour*steps_per_day*steps_per_month)) % steps_per_year / (steps_per_year - 1) for i in range(data.shape[0])]
            moy_tiled = np.tile(moy, [1, num_nodes, 1]).transpose((2, 1, 0)) - 0.5

            feature_list.append(moh_tiled)
            feature_list.append(hod_tiled)
            feature_list.append(dom_tiled)
            feature_list.append(moy_tiled)

            processed_data = np.concatenate(feature_list, axis=-1)
            processed_datay = np.concatenate([np.expand_dims(data_y, axis=-1), moy_tiled, hod_tiled, dom_tiled, moy_tiled], axis=-1)
        else:
            processed_data = data

        self.data_x = processed_data[border1:border2]
        self.data_y = processed_datay[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        l_begin = s_end - self.label_len
        if l_begin < 0:
            seq_long = np.zeros((self.label_len, self.data_x.shape[1], self.data_x.shape[2]))
            seq_long[-self.seq_len:] = seq_x
        else:
            seq_long = self.data_x[l_begin:s_end]

        return seq_x, seq_y, seq_long

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_NYCTaxi(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, scale=True, time_to_feature=0, type='flow'):
        if size == None:
            self.seq_len = 6
            self.label_len = 12*24*7
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type
        self.set_type = type_map[flag]

        self.scale = scale
        self.time_to_feature = time_to_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data_file_path = self.root_path + self.data_path + '.grid'
        geo_file_path = self.root_path + self.data_path + '.geo'
        data = pd.read_csv(data_file_path)
        geo = pd.read_csv(geo_file_path)

        data_col = ['time', 'row_id', 'column_id', 'inflow', 'outflow']
        data = data[data_col]
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
        data = np.array(data, dtype=np.float)
        data = data.swapaxes(0, 1)

        num_train = round(len(data) * 0.7)
        num_test = round(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        data_y = data

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
            print('mean:', train_data.mean())
            print('std:', train_data.std())
            # 对数据进行标准化
            data = self.scaler.transform(data)
        else:
            data = data

        num_samples, num_nodes, _ = data.shape
        if self.time_to_feature == 0:
            data_list = []
            time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
            dayofweek = pd.Index(timesolts).dayofweek
            day_in_week = np.tile(dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(day_in_week)
            processed_data = np.concatenate([data] + data_list, axis=-1)
            processed_datay = np.concatenate([data_y] + data_list, axis=-1)
        elif self.time_to_feature == 1:
            l, n, _ = data.shape
            # 对时间进行编码，返回是一个编码后的矩阵，每一行对应一个时间，列为编码后的特征
            stamp = time_features(pd.to_datetime(timesolts), freq='T')
            # 进行转置，每一行对应一个特征，列为对应的时间
            stamp = stamp.transpose(1, 0)
            stamp_tiled = np.tile(stamp, [n, 1, 1]).transpose((1, 0, 2))
            processed_data = np.concatenate([data, stamp_tiled], axis=-1)
            processed_datay = np.concatenate([data, stamp_tiled], axis=-1)
        else:
            processed_data = data

        time_stamp = {'date': timesolts}
        df_stamp = pd.DataFrame(time_stamp)
        df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(columns=['date']).values

        self.data_x = processed_data[border1:border2]
        self.data_y = processed_datay[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    root_path = '../datasets/'
    data_path = 'NYCTaxi/NYCTaxi'
    # data_path = 'PEMS-BAY/PEMS-BAY.h5'
    data = Dataset_NYCTaxi(root_path=root_path, data_path=data_path, time_to_feature=0)
    # data = Dataset_h5(root_path=root_path, data_path=data_path, time_to_feature=True)
    seq_x, seq_y, seq_x_mark, seq_y_mark = data.__getitem__(0)
    print(seq_x.shape)
    print(seq_y.shape)
    print(seq_x)
    print(seq_y)
    print(seq_x_mark)
    print(seq_y_mark)
    print(type(seq_x[0, 0, 0]))
    print(type(seq_x[0, 0, :]))
    print(type(seq_x[0, :, :]))
    print(type(seq_x))
