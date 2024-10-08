import datetime
import pickle

import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, masked_mae
from utils.tools import EarlyStopping, save_trainlog, print_log
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_stTrans_mae(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        # # 读取邻接矩阵
        # with open(self.args.adj_path, 'rb') as f:
        #     pickle_data = pickle.load(f, encoding="latin1")
        # adj_mx = pickle_data[2]
        # adj = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
        # # num_nodes = len(pickle_data[0])
        # supports = [torch.tensor(i).to(self.device) for i in adj]
        # .float(): 将模型的参数和张量转换为浮点数类型
        supports = None
        model = self.model_dict[self.args.model].Model(self.args, supports).float()

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = masked_mae
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, maes, mses, rmses, mapes, mspes = [], [], [], [], [], []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_long) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_long = batch_long.float().to(self.device)

                # encoder - decoder
                outputs, _ = self.model(batch_x, batch_long)
                outputs = outputs.squeeze(-1)
                y = batch_y[..., 0]
                if vali_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = vali_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                pred_len, n_nodes)

                loss = criterion(outputs, y)
                total_loss.append(loss.item())

                mae, mse, rmse, mape, mspe = metric(outputs, y)
                maes.append(mae.item())
                mses.append(mse.item())
                rmses.append(rmse.item())
                mapes.append(mape.item())
                mspes.append(mspe.item())
                preds.append(outputs)
                trues.append(y)

        total_loss, maes, mses, rmses, mapes, mspes = np.average(total_loss), np.average(maes), \
                                                      np.average(mses), np.average(rmses), \
                                                      np.average(mapes), np.average(mspes)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        self.model.train()
        return total_loss, maes, mses, rmses, mapes, mspes, preds, trues

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # log_path = self.args.log_path + setting + '/'
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        log_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log = os.path.join(log_path, f"{self.args.model}-{self.args.data}-{now}.log")
        log = open(log, "a")
        log.seek(0)
        log.truncate()

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # path = '/mnt/workspace/'  # 使用阿里天池跑代码的路径
        path = '/kaggle/working/'  # 使用kaggle跑实验时的路径

        time_now = time.time()

        train_steps = len(train_loader)
        if self.device == 0:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model_optim,
            milestones=[20, 30, 105],
            gamma=0.1
        )
        criterion = self._select_criterion()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.device == 0:
            print_log(
                log,
                summary(
                    self.model,
                    [
                        (self.args.batch_size, self.args.seq_len, self.args.num_nodes, 3),
                        (self.args.batch_size, self.args.label_len, self.args.num_nodes, 3)
                    ],
                    verbose=0,  # avoid print twice
                )
            )
            print_log(log, 'number of params (M): %.2f' % (n_parameters / 1.e6))

        # tensorboard_path = os.path.join('./runs/{}/'.format(setting))
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)
        # tensorboard_path = '/mnt/workspace/'  # 使用阿里天池跑实验时的路径
        tensorboard_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        if self.device == 0:
            writer = SummaryWriter(log_dir=tensorboard_path)

        step, best_epoch = 0, -1

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_long) in enumerate(train_loader):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_long = batch_long.float().to(self.device)

                outputs, time_pred = self.model(batch_x, batch_long)
                outputs = outputs.squeeze(-1)
                y = batch_y[..., 0]

                batch_size, pred_len, n_nodes = outputs.shape
                if train_data.scale and self.args.inverse:
                    outputs = train_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                 pred_len, n_nodes)

                loss = criterion(outputs, y)
                # loss = criterion(outputs, y) + criterion(time_pred, batch_y[:, :, 0, 1:])
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
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                model_optim.step()

            if self.device == 0:
                current_lr = model_optim.param_groups[0]['lr']
                print_log(log, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                print_log(log, "Epoch: {} current lr: {}".format(epoch + 1, current_lr))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe, _, _ = self.vali(vali_data, vali_loader,
                                                                                             criterion)
            scheduler.step()  # 学习率调整
            if self.device == 0:
                print_log(
                    log,
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".
                    format(epoch + 1, train_steps, train_loss, vali_loss)
                )
            test_loss, test_mae, test_mse, test_rmse, test_mape, \
            test_mspe, test_preds, test_trues = self.vali(test_data, test_loader, criterion)
            if self.device == 0:
                print_log(
                    log,
                    "Epoch: {0}, Steps: {1} | Test Loss: {2:.7f} Test mae:{3:.7f} Test rmse:{4:.7f} Test mape: {5:.7f}".
                    format(epoch + 1, train_steps, test_loss, test_mae, test_rmse, test_mape)
                )
                _, pred_len, _ = test_preds.shape
                for i in range(pred_len):
                    mae, mse, rmse, mape, mspe = metric(test_preds[:, i, :], test_trues[:, i, :])
                    print_log(
                        log,
                        f'Evaluate model on test data for horizon {i}, Test MAE: {mae}, Test RMSE: {rmse}, Test MAPE: {mape}'
                    )
            if self.device == 0:
                early_stopping(vali_loss, self.model, path, epoch, self.device)
            if self.device == 0:
                if early_stopping.early_stop:
                    print_log(log, "Early stopping")
                    print_log(log, "best epoch: {0}".format(early_stopping.best_epoch))
                    break

            if self.device == 0:
                writer.add_scalar(scalar_value=train_loss, global_step=step, tag='Loss/train')
                writer.add_scalar(scalar_value=vali_loss, global_step=step, tag='Loss/valid')

        best_model_path = path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')), strict=False)
            print("Model = %s" % str(self.model))
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('number of params (M): %.2f' % (n_parameters / 1.e6))
        preds = []
        trues = []
        x_trues = []
        x_marks = []
        y_marks = []

        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_long) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_long = batch_long.float().to(self.device)

                outputs, _ = self.model(batch_x, batch_long)
                outputs = outputs.squeeze(-1)
                y = batch_y[..., 0]

                if test_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                pred_len, n_nodes)

                mae, mse, rmse, mape, mspe = metric(outputs, y)
                print("\tmae: {0:.7f} | mse: {1:.7f} | rmse: {2:.7f} | mape: {3:.7f} | mspe: {4:.7f}".format(mae, mse,
                                                                                                             rmse, mape,
                                                                                                             mspe))
                maes.append(mae.item())
                mses.append(mse.item())
                rmses.append(rmse.item())
                mapes.append(mape.item())
                mspes.append(mspe.item())

                inputs = batch_x[:, :, :, 0].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    batch_size, seq_len, n_nodes = inputs.shape
                    inputs = test_data.inverse_transform(inputs.reshape(-1, n_nodes)).reshape(batch_size, seq_len,
                                                                                              n_nodes)
                x_trues.append(inputs.astype(np.float32))
                preds.append(outputs)
                trues.append(y)

        mae, mse, rmse, mape, mspe = np.average(maes), \
                                          np.average(mses), np.average(rmses), \
                                          np.average(mapes), np.average(mspes)
        print('rmse:{:.7f}, mae:{:.7f}, mape:{:.7f}'.format(rmse, mae, mape))

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # folder_path = '/kaggle/working/'  # 使用kaggle跑代码时的输出路径

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        x_trues = np.array(x_trues).astype(np.float32)
        maes = np.array(maes)
        mses = np.array(mses)
        rmses = np.array(rmses)
        mspes = np.array(mspes)
        mapes = np.array(mapes)
        print('test shape:', preds.shape, trues.shape, x_trues.shape, maes.shape,
              mses.shape, rmses.shape, mspes.shape, mapes.shape)
        x_trues = x_trues.reshape((-1, x_trues.shape[-2], x_trues.shape[-1]))
        print('test shape:', preds.shape, trues.shape, x_trues.shape, maes.shape,
              mses.shape, rmses.shape, mspes.shape, mapes.shape)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds.detach().cpu().numpy())
        np.save(folder_path + 'true.npy', trues.detach().cpu().numpy())
        np.save(folder_path + 'x_trues.npy', x_trues)
        np.save(folder_path + 'maes.npy', maes)
        np.save(folder_path + 'mses.npy', mses)
        np.save(folder_path + 'rmses.npy', rmses)
        np.save(folder_path + 'mapes.npy', mapes)
        np.save(folder_path + 'mspes.npy', mspes)
