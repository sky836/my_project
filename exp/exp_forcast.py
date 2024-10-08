import pickle

import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, masked_mae
from utils.tools import EarlyStopping, save_trainlog
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        # 读取邻接矩阵
        with open(self.args.adj_path, 'rb') as f:
            pickle_data = pickle.load(f, encoding="latin1")
        adj_mx = pickle_data[2]
        adj = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
        # num_nodes = len(pickle_data[0])
        supports = [torch.tensor(i).to(self.device) for i in adj]
        # .float(): 将模型的参数和张量转换为浮点数类型
        model = self.model_dict[self.args.model].Model(self.args, supports=supports, device=self.device).float()

        if self.args.is_finetune:
            msg = model.load_state_dict(torch.load(self.args.best_model_path), strict=False)
            print(msg)

        if self.args.use_multi_gpu and self.args.use_gpu:
            # nn.DataParallel: 这是 PyTorch 中的一个模块，用于在多个 GPU 上并行地运行模型。
            # 它将输入模型封装在一个新的 DataParallel 模型中。
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        if self.args.data in ("METR-LA", "PEMS-BAY"):
            criterion = masked_mae
        else:
            criterion = nn.HuberLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, maes, mses, rmses, mapes, mspes = [], [], [], [], [], []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, cross_time_attn_scores, cross_target_attn_scores, cross_merge_attn_scores, outputs = \
                                self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                        else:
                            _, _, _, _, _,outputs = self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                else:
                    if self.args.output_attention:
                        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, cross_time_attn_scores, cross_target_attn_scores, cross_merge_attn_scores, outputs = \
                            self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                    else:
                        _, _, _, _, _, outputs = self.model(batch_x, batch_y[:, self.args.label_len:, :, :])

                y = batch_y[:, self.args.label_len:, :, 0]
                if vali_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = vali_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                 pred_len, n_nodes)
                    y = vali_data.inverse_transform(y.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                     pred_len, n_nodes)
                if self.args.data in ("METR-LA", "PEMS-BAY"):
                    loss = criterion(outputs, y, 0.0)
                else:
                    loss = criterion(outputs, y)
                total_loss.append(loss.item())

                outputs = outputs.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                mae, mse, rmse, mape, mspe = metric(outputs, y)
                maes.append(mae)
                mses.append(mse)
                rmses.append(rmse)
                mapes.append(mape)
                mspes.append(mspe)
                preds.append(outputs)
                trues.append(y)

        total_loss, maes, mses, rmses, mapes, mspes = np.average(total_loss), np.average(maes), \
                                                      np.average(mses), np.average(rmses), \
                                                      np.average(mapes), np.average(mspes)
        preds = np.array(preds)
        trues = np.array(trues)
        _, _, pred_len, n_nodes = preds.shape
        preds, trues = preds.reshape((-1, pred_len, n_nodes)), trues.reshape((-1, pred_len, n_nodes))
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

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # path = '/mnt/workspace/'  # 使用阿里天池跑代码的路径
        path = '/kaggle/working/' # 使用kaggle跑实验时的路径

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model_optim,
            milestones=[20, 30],
            gamma=0.1
        )
        criterion = self._select_criterion()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        save_trainlog(log_path, "Model = %s" % str(self.model))
        save_trainlog(log_path, 'number of params (M): %.2f' % (n_parameters / 1.e6))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # tensorboard_path = os.path.join('./runs/{}/'.format(setting))
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)
        # tensorboard_path = '/mnt/workspace/'  # 使用阿里天池跑实验时的路径
        tensorboard_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        writer = SummaryWriter(log_dir=tensorboard_path)

        step = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            train_pbar = tqdm(train_loader, position=0, leave=True)  # 可视化训练的过程

            preds = []
            trues = []
            x_trues = []
            x_marks = []
            y_marks = []
            maes, mses, rmses, mapes, mspes = [], [], [], [], []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_pbar):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, decoder_self_attn_score, cross_attn, outputs = \
                                self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                        else:
                            _, _, _, _, _, outputs = self.model(batch_x, batch_y[:, self.args.label_len:, :, :])

                else:
                    if self.args.output_attention:
                        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, cross_time_attn_scores, cross_target_attn_scores, cross_merge_attn_scores, outputs = \
                            self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                    else:
                        _, _, _, _, _, outputs = self.model(batch_x, batch_y[:, self.args.label_len:, :, :])

                y = batch_y[:, self.args.label_len:, :, 0]
                if train_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = train_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                 pred_len, n_nodes)
                    y = train_data.inverse_transform(y.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                     pred_len, n_nodes)
                # if (outputs < 0).any():
                #     # 找到小于0的元素的索引
                #     indices = (outputs < 0).nonzero(as_tuple=True)
                #     # 遍历索引并打印对应的元素值
                #     for idx in zip(*indices):
                #         print(f'pred:{outputs[idx]}, true:{y[idx]}')
                if self.args.data in ("METR-LA", "PEMS-BAY"):
                    loss = criterion(outputs, y, 0.0)
                else:
                    loss = criterion(outputs, y)
                train_loss.append(loss.item())

                train_pbar.set_description(f'Epoch [{epoch + 1}/{self.args.train_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})

                outputs = outputs.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                if epoch == self.args.train_epochs - 1:
                    input = batch_x[..., 0].detach().cpu().numpy()
                    if train_data.scale and self.args.inverse:
                        batch_size, pred_len, n_nodes = input.shape
                        input = train_data.inverse_transform(input.reshape(-1, n_nodes)).reshape(batch_size, pred_len,
                                                                                                 n_nodes)
                    x_trues.append(input)
                    batch_x_mark = batch_x_mark.detach().cpu().numpy()
                    batch_y_mark = batch_y_mark.detach().cpu().numpy()
                    x_marks.append(batch_x_mark)
                    y_marks.append(batch_y_mark)
                    preds.append(outputs)
                    trues.append(y)

                if (i + 1) % 100 == 0:
                    mae, mse, rmse, mape, mspe = metric(outputs, y)
                    # print('pred:', outputs)
                    # print(outputs.shape)
                    # print('true:', y)
                    # print(y.shape)
                    save_trainlog(log_path, "\titers: {0}, epoch: {1} | loss: {2:.7f} | mae: {3:.7f} | mse: {4:.7f} | rmse: {5:.7f} | mape: {6:.7f} | mspe: {7:.7f}".
                          format(i + 1, epoch + 1, loss.item(), mae, mse, rmse, mape, mspe))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    save_trainlog(log_path, '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    model_optim.step()

            save_trainlog(log_path, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            lr = scheduler.get_last_lr()
            save_trainlog(log_path, "Epoch: {} lr: {}".format(epoch + 1, lr))
            scheduler.step()  # 学习率调整
            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe, _, _ = self.vali(vali_data, vali_loader, criterion)

            save_trainlog(log_path, "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali mae: {4:.7f} Vali mse: {5:.7f} "
                  "Vali rmse: {6:.7f} Vali mape: {7:.7f} Vali mspe: {8:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe))

            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe, test_preds, test_trues = self.vali(
                test_data, test_loader, criterion)
            save_trainlog(log_path, "Epoch: {0}, Steps: {1} | Test Loss: {2:.7f} Test mae: {3:.7f} Test mse: {4:.7f} "
                  "Test rmse: {5:.7f} Test mape: {6:.7f} Test mspe: {7:.7f}".format(
                epoch + 1, train_steps, test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe))
            _, pred_len, _ = test_preds.shape
            for i in range(pred_len):
                mae, mse, rmse, mape, mspe = metric(test_preds[:, i, :], test_trues[:, i, :])
                save_trainlog(log_path,
                    f'Evaluate model on test data for horizon {i}, Test MAE: {mae}, Test RMSE: {rmse}, Test MAPE: {mape}')

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                save_trainlog(log_path, "Early stopping")
                break

            writer.add_scalar(scalar_value=train_loss, global_step=step, tag='Loss/train')
            writer.add_scalar(scalar_value=vali_loss, global_step=step, tag='Loss/valid')
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            '''
            # ==================保存训练过程中间结果===================================================
            # folder_path = './train_results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            folder_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
            preds = np.array(preds)
            trues = np.array(trues)
            x_trues = np.array(x_trues)
            x_marks = np.array(x_marks)
            y_marks = np.array(y_marks)
            maes = np.array(maes)
            mses = np.array(mses)
            rmses = np.array(rmses)
            mspes = np.array(mspes)
            mapes = np.array(mapes)
            print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape,
                  mses.shape, rmses.shape, mspes.shape, mapes.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            x_trues = x_trues.reshape(-1, x_trues.shape[-2], x_trues.shape[-1])
            x_marks = x_marks.reshape(-1, x_marks.shape[-2], x_marks.shape[-1])
            y_marks = y_marks.reshape(-1, y_marks.shape[-2], y_marks.shape[-1])
            print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape,
                  mses.shape, rmses.shape, mspes.shape, mapes.shape)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            np.save(folder_path + '{0}metrics.npy'.format(epoch), np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + '{0}pred.npy'.format(epoch), preds)
            np.save(folder_path + '{0}true.npy'.format(epoch), trues)
            np.save(folder_path + '{0}x_trues.npy'.format(epoch), x_trues)
            np.save(folder_path + '{0}x_marks.npy'.format(epoch), x_marks)
            np.save(folder_path + '{0}y_marks.npy'.format(epoch), y_marks)
            np.save(folder_path + '{0}maes.npy'.format(epoch), maes)
            np.save(folder_path + '{0}mses.npy'.format(epoch), mses)
            np.save(folder_path + '{0}rmses.npy'.format(epoch), rmses)
            np.save(folder_path + '{0}mapes.npy'.format(epoch), mapes)
            np.save(folder_path + '{0}mspes.npy'.format(epoch), mspes)
            # ==================保存训练过程中间结果===================================================
            '''

        best_model_path = path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            print("Model = %s" % str(self.model))
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('number of params (M): %.2f' % (n_parameters / 1.e6))
        preds = []
        trues = []
        x_trues = []
        x_marks = []
        y_marks = []
        encoder_time_attns_scores, encoder_target_attns_scores, encoder_merge_attns_scores, decoder_self_attns_score, cross_attns = [], [], [], [], []
        maes, mses, rmses, mapes, mspes = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, cross_time_attn_scores, cross_target_attn_scores, cross_merge_attn_scores, outputs = \
                                self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                        else:
                            _, _, _, _, _, outputs = self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                else:
                    if self.args.output_attention:
                        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, decoder_self_attn_score, cross_attn, outputs = \
                            self.model(batch_x, batch_y[:, self.args.label_len:, :, :])
                    else:
                        _, _, _, _, _, outputs = self.model(batch_x, batch_y[:, self.args.label_len:, :, :])

                y = batch_y[:, self.args.label_len:, :, 0]
                if test_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                pred_len, n_nodes)
                    y = test_data.inverse_transform(y.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                    pred_len, n_nodes)
                outputs = outputs.detach().cpu().numpy()[:, :, :]
                y = y.detach().cpu().numpy()[:, :, :]
                mae, mse, rmse, mape, mspe = metric(outputs, y)
                print("\tmae: {0:.7f} | mse: {1:.7f} | rmse: {2:.7f} | mape: {3:.7f} | mspe: {4:.7f}".format(mae, mse,
                                                                                                             rmse, mape,
                                                                                                             mspe))
                maes.append(mae)
                mses.append(mse)
                rmses.append(rmse)
                mapes.append(mape)
                mspes.append(mspe)

                inputs = batch_x[:, :, :, 0].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    batch_size, seq_len, n_nodes = inputs.shape
                    inputs = test_data.inverse_transform(inputs.reshape(-1, n_nodes)).reshape(batch_size, seq_len,
                                                                                              n_nodes)
                x_trues.append(inputs.astype(np.float32))
                batch_x_mark = batch_x_mark.detach().cpu().numpy()
                batch_y_mark = batch_y_mark.detach().cpu().numpy()
                x_marks.append(batch_x_mark)
                y_marks.append(batch_y_mark)
                preds.append(outputs.astype(np.float32))
                trues.append(y.astype(np.float32))
                if self.args.output_attention:
                    encoder_time_attns_scores.append(encoder_time_attn_scores)
                    encoder_target_attns_scores.append(encoder_target_attn_scores)
                    encoder_merge_attns_scores.append(encoder_merge_attn_scores)
                    cross_attns.append(cross_attn)

        if self.args.output_attention:
            # [189, 1, 5, 207, 4, 1, 24, 24]
            encoder_time_attns_scores = np.array(encoder_time_attns_scores).transpose([0, 2, 1, 3, 4, 5, 6, 7])
            # [189, 1, 5, 207, 4, 1, 24, 24]
            encoder_target_attns_scores = np.array(encoder_target_attns_scores).transpose([0, 2, 1, 3, 4, 5, 6, 7])
            # [189, 1, 5, 207, 4, 24, 24]
            encoder_merge_attns_scores = np.array(encoder_merge_attns_scores).transpose([0, 2, 1, 3, 4, 5, 6])
            # [189, 1, 5, 207, 4, 1, 24]
            cross_attns = np.array(cross_attns).transpose([0, 2, 1, 3, 4, 5, 6])

            shape = encoder_time_attns_scores.shape
            encoder_time_attns_scores = encoder_time_attns_scores.reshape(-1, shape[2], shape[3], shape[4], shape[5], shape[6], shape[7])
            encoder_target_attns_scores = encoder_target_attns_scores.reshape(-1, shape[2], shape[3], shape[4], shape[5], shape[6], shape[7])
            encoder_merge_attns_scores = encoder_merge_attns_scores.reshape(-1, shape[2], shape[3], shape[4], shape[6], shape[7])
            cross_attns = cross_attns.reshape(-1, shape[2], shape[3], shape[4], shape[5], shape[6])

        preds = np.array(preds).astype(np.float32)
        trues = np.array(trues).astype(np.float32)
        x_trues = np.array(x_trues).astype(np.float32)
        x_marks = np.array(x_marks)
        y_marks = np.array(y_marks)
        maes = np.array(maes).astype(np.float32)
        mses = np.array(mses).astype(np.float32)
        rmses = np.array(rmses).astype(np.float32)
        mspes = np.array(mspes).astype(np.float32)
        mapes = np.array(mapes).astype(np.float32)
        print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape,
              mses.shape, rmses.shape, mspes.shape, mapes.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        x_trues = x_trues.reshape(-1, x_trues.shape[-2], x_trues.shape[-1])
        x_marks = x_marks.reshape(-1, x_marks.shape[-2], x_marks.shape[-1])
        y_marks = y_marks.reshape(-1, y_marks.shape[-2], y_marks.shape[-1])
        print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape,
              mses.shape, rmses.shape, mspes.shape, mapes.shape)

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # folder_path = '/kaggle/working/'  # 使用kaggle跑代码时的输出路径

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x_trues.npy', x_trues)
        np.save(folder_path + 'x_marks.npy', x_marks)
        np.save(folder_path + 'y_marks.npy', y_marks)
        np.save(folder_path + 'maes.npy', maes)
        np.save(folder_path + 'mses.npy', mses)
        np.save(folder_path + 'rmses.npy', rmses)
        np.save(folder_path + 'mapes.npy', mapes)
        np.save(folder_path + 'mspes.npy', mspes)

        if self.args.output_attention:
            np.save(folder_path + 'encoder_time_attns_scores.npy', encoder_time_attns_scores)
            np.save(folder_path + 'encoder_target_attns_scores.npy', encoder_target_attns_scores)
            np.save(folder_path + 'encoder_merge_attns_scores.npy', encoder_merge_attns_scores)
            np.save(folder_path + 'cross_attns.npy', cross_attns)


    def Autoregressive_test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        x_trues = []
        x_marks = []
        y_marks = []
        maes, mses, rmses, mapes, mspes = [], [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_size, len, n_nodes, _ = batch_y.shape
                # TODO:decoder_input最后一维的维度检查
                decoder_input = torch.tensor(batch_y).float().to(self.device)  # 为了实现深拷贝，不然改变decoder_input的值batch_y的值也会改变
                mask = torch.zeros(len).type(torch.bool)
                mask[self.args.label_len:] = True
                decoder_input[:, mask, :, 0] = 0.0

                # 1.执行encoder
                encoder_time = batch_x[..., 1:]
                encoder_target = batch_x[..., 0].unsqueeze(-1)
                encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, \
                encoder_time, encoder_target = self.model.encoder_forward(encoder_time, encoder_target)

                # 2.执行decoder
                for j in range(self.args.pred_len):
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                cross_time_attn_scores, cross_target_attn_scores, cross_merge_attn_scores, outputs = \
                                    self.model.decoder_forward(encoder_time, encoder_target,
                                                               decoder_input[:, :-self.args.label_len, :, :])
                            else:
                                _, _, _, outputs = \
                                    self.model.decoder_forward(encoder_time, encoder_target,
                                                               decoder_input[:, :-self.args.label_len, :, :])
                            decoder_input[:, self.args.label_len + j, :, 0] = outputs[:, j, :, 0]
                    else:
                        if self.args.output_attention:
                            cross_time_attn_scores, cross_target_attn_scores, cross_merge_attn_scores, outputs = \
                                self.model.decoder_forward(encoder_time, encoder_target,
                                                           decoder_input[:, :-self.args.label_len, :, :])
                        else:
                            _, _, _, outputs = \
                                self.model.decoder_forward(encoder_time, encoder_target,
                                                           decoder_input[:, :-self.args.label_len, :, :])
                        decoder_input[:, self.args.label_len + j, :, 0] = outputs[:, j, :, 0]
                decoder_output = decoder_input[:, self.args.label_len:, :, 0]

                outputs = decoder_output.detach().cpu().numpy()
                batch_y = batch_y[:, self.args.label_len:, :, 0].detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size, pred_len,
                                                                                                n_nodes)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, n_nodes)).reshape(batch_size, pred_len,
                                                                                                n_nodes)
                mae, mse, rmse, mape, mspe = metric(outputs, batch_y)
                maes.append(mae), mses.append(mse), rmses.append(rmse), mapes.append(mape), mspes.append(mspe)
                print("\tmae: {0:.7f} | mse: {1:.7f} | rmse: {2:.7f} | mape: {3:.7f} | mspe: {4:.7f}".format(mae, mse, rmse, mape, mspe))
                # print('pred:', outputs)
                # print('true:', batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                inputs = batch_x[:, :, :, 0].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    batch_size, seq_len, n_nodes = inputs.shape
                    inputs = test_data.inverse_transform(inputs.reshape(-1, n_nodes)).reshape(batch_size, seq_len,
                                                                                              n_nodes)
                x_trues.append(inputs)
                batch_x_mark = batch_x_mark.detach().cpu().numpy()
                batch_y_mark = batch_y_mark.detach().cpu().numpy()
                x_marks.append(batch_x_mark)
                y_marks.append(batch_y_mark)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        x_trues = np.array(x_trues)
        x_marks = np.array(x_marks)
        y_marks = np.array(y_marks)
        maes = np.array(maes)
        mses = np.array(mses)
        rmses = np.array(rmses)
        mspes = np.array(mspes)
        mapes = np.array(mapes)
        print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape, mses.shape, rmses.shape, mspes.shape, mapes.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        x_trues = x_trues.reshape(-1, x_trues.shape[-2], x_trues.shape[-1])
        x_marks = x_marks.reshape(-1, x_marks.shape[-2], x_marks.shape[-1])
        y_marks = y_marks.reshape(-1, y_marks.shape[-2], y_marks.shape[-1])
        print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape, mses.shape, rmses.shape, mspes.shape, mapes.shape)

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # folder_path = '/kaggle/working/'  # 使用kaggle跑代码时的输出路径

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x_trues.npy', x_trues)
        np.save(folder_path + 'x_marks.npy', x_marks)
        np.save(folder_path + 'y_marks.npy', y_marks)
        np.save(folder_path + 'maes.npy', maes)
        np.save(folder_path + 'mses.npy', mses)
        np.save(folder_path + 'rmses.npy', rmses)
        np.save(folder_path + 'mapes.npy', mapes)
        np.save(folder_path + 'mspes.npy', mspes)

        return

