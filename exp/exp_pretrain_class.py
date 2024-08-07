import datetime
import pickle

import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, print_log
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Pretrain_Class(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        # 读取邻接矩阵
        # with open(self.args.adj_path, 'rb') as f:
        #     pickle_data = pickle.load(f, encoding="latin1")
        # adj_mx = pickle_data[2]
        # adj = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
        # # num_nodes = len(pickle_data[0])
        # supports = [torch.tensor(i).to(self.device) for i in adj]
        # .float(): 将模型的参数和张量转换为浮点数类型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # msg = model.load_state_dict(torch.load(self.args.best_model_path), strict=True)
        # print(msg)
        return model

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        valid_loss = []
        with torch.no_grad():
            for i, (data, label) in enumerate(vali_loader):
                data = data.float().to(self.device)
                label = label.float().to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, label)

                valid_loss.append(loss.item())

        total_loss = np.average(valid_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # log_path = self.args.log_path + setting + '/'
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        log_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log = os.path.join(log_path, f"{self.args.model}-{self.args.data}-{now}.log")
        log = open(log, "a")
        log.seek(0)
        log.truncate()

        if self.device == 0:
            print_log(
                log,
                summary(
                    self.model,
                    [
                        (self.args.batch_size, self.args.label_len, self.args.num_nodes, 3)
                    ],
                    verbose=0,  # avoid print twice
                )
            )
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print_log(log, 'number of params (M): %.2f' % (n_parameters / 1.e6))

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # path = '/mnt/workspace/'  # 使用阿里天池跑代码的路径
        path = '/kaggle/working/' # 使用kaggle跑实验时的路径

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = nn.CrossEntropyLoss(reduction='mean')
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim,
                                                                  mode='min', factor=0.1, patience=5,
                                                                  verbose=False, threshold=0.001, threshold_mode='rel',
                                                                  cooldown=0, min_lr=1e-7, eps=1e-08)

        # tensorboard_path = os.path.join('./runs/{}/'.format(setting))
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)
        # tensorboard_path = '/mnt/workspace/'  # 使用阿里天池跑实验时的路径
        if self.device == 0:
            tensorboard_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
            writer = SummaryWriter(log_dir=tensorboard_path)

        step = 0
        time_now = time.time()
        best_loss = np.Inf

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (data, label) in enumerate(train_loader):
                print(f"Batch {i} - Data shape: {data}, Label shape: {label}")
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                data = data.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, label)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    if self.device == 0:
                        print_log(log,
                                  "\tepoch: {1} | iters: {0} | loss: {2:.7f} | speed: {3:.4f}s/iter | left time: {4:.4f}s".
                                  format(i + 1, epoch + 1, loss.item(), speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if self.device == 0:
                print_log(log, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                valid_loss = self.vali(vali_data, vali_loader, criterion)
                print_log(log, "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Valid Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
                lr_scheduler.step(valid_loss)
                current_lr = model_optim.param_groups[0]['lr']
                print_log(log, "Epoch: {} current lr: {}".format(epoch + 1, current_lr))
                writer.add_scalar(scalar_value=train_loss, global_step=epoch+1, tag='Loss/train')
                writer.add_scalar(scalar_value=valid_loss, global_step=epoch+1, tag='Loss/valid')

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print_log(log, f'Saving state ...')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': model_optim.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),  # 如果使用了学习率调度器
                        'loss': train_loss,
                    }, path + '/' + 'checkpoint.pth')

        return self.model
