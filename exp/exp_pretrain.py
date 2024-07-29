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


class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)

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

    def patchify(self, series, num_patches, n_nodes=False):
        """
        time: (batch_size, seq_len, 2)
        target: (batch_size, seq_len, n_nodes, 1)

        patch_time: (batch_size, num_patches, patch_size * 2)
        patch_target: (batch_size, num_patches, n_nodes, patch_size * 1)
        """
        batch_size, seq_len, n_feats = series.shape[0], series.shape[1], series.shape[-1]
        series = series.transpose(1, -2)
        if n_nodes:
            series = series.reshape(-1, series.shape[-2], series.shape[-1])
        assert seq_len % num_patches == 0

        patch_size = seq_len // num_patches
        patch_series = series.reshape(-1, num_patches, patch_size, n_feats)
        patch_series = patch_series.reshape(-1, num_patches, patch_size * n_feats)

        if n_nodes:
            patch_series = patch_series.reshape(batch_size, -1, num_patches, patch_size * n_feats)

        return patch_series.transpose(1, -2)

    def unpatchify(self, patches, n_nodes=False):
        """
        patch_time: (batch_size, num_patches, patch_size * 2)
        patch_target: (batch_size, num_patches, n_nodes, patch_size * 1)

        time: (batch_size, seq_len, 2)
        target: (batch_size, seq_len, n_nodes, 1)
        """
        patches = patches.transpose(1, -2)
        batch_size = patches.shape[0]
        if n_nodes:
            patches = patches.reshape(-1, patches.shape[-2], patches.shape[-1])
        _, num_patches, patch_size_feats = patches.shape
        patch_size = self.args.seq_len // num_patches
        n_feats = patch_size_feats // patch_size

        patches = patches.reshape(-1, num_patches, patch_size, n_feats)
        unpatches = patches.reshape(-1, num_patches * patch_size, n_feats)

        if n_nodes:
            unpatches = unpatches.reshape(batch_size, -1, num_patches * patch_size, n_feats)

        return unpatches.transpose(1, -2)

    def forward_loss(self, x_time, x_target, time_outputs, target_outputs, mask):
        """
        time_outputs: [batch_size, num_patches, patch_size * n_feats]
        target_outputs: [batch_size, num_patches, n_nodes, patch_size * n_feats]
        mask: [num_patches]
        """
        batch_size, num_patches, n_nodes, _ = x_target.shape

        mask_target = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_nodes, num_patches).transpose(1, 2)
        mask_time = mask.unsqueeze(0).expand(batch_size, num_patches)

        mask0 = (x_target != 0.0)
        mask0 = mask0.float()
        mask0 /= torch.mean((mask0))
        mask0 = torch.where(torch.isnan(mask0), torch.zeros_like(mask0), mask0)

        loss_target = torch.abs(target_outputs - x_target)
        loss_target = loss_target * mask0
        loss_target = loss_target.mean(dim=-1)  # [batch_size, n_nodes, num_patches] mean loss per patch
        loss_target = torch.where(torch.isnan(loss_target), torch.zeros_like(loss_target), loss_target)
        loss_target = (loss_target * mask_target).sum() / mask_target.sum()

        loss_time = torch.abs(time_outputs - x_time).mean(dim=-1)
        loss_time = (loss_time * mask_time).sum() / mask_time.sum()

        loss = loss_time + loss_target

        return loss, loss_time, loss_target

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
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
                        (self.args.batch_size, self.args.seq_len, self.args.num_nodes, 3)
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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim,
                                                                  mode='min', factor=0.1, patience=10,
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

        maes, maes_time, maes_target = [], [], []
        preds = []
        trues = []
        x_marks = []
        masks = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            maes, maes_time, maes_target = [], [], []
            preds = []
            trues = []
            x_marks = []
            masks = []

            self.model.train()
            epoch_time = time.time()

            if best_loss < 5:
                epoch = self.args.train_epochs - 1

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                time_outputs, target_outputs, mask_record = self.model(batch_x)
                """
                time_outputs, y_time: [batch_size, num_patches, patch_size * 2]
                target_outputs, y_target: [batch_size, num_patches, n_nodes, patch_size * 1]
                mask: [num_patches]
                """
                _, num_patches, _ = time_outputs.shape
                target_outputs = self.unpatchify(target_outputs, True)
                batch_size, seq_len, n_nodes, n_feats = target_outputs.shape

                target_outputs = target_outputs.squeeze(-1)  # B, L, N
                x_target = batch_x[:, :, :, 0]
                x_time = batch_x[:, :, 0, 1:]
                if train_data.scale and self.args.inverse:
                    target_outputs = train_data.inverse_transform(target_outputs.reshape(-1, n_nodes)).reshape(
                        batch_size, seq_len, n_nodes)
                    x_target = train_data.inverse_transform(x_target.reshape(-1, n_nodes)).reshape(
                        batch_size, seq_len, n_nodes)

                if epoch == self.args.train_epochs - 1 and self.device == 0:
                    batch_x_mark = batch_x_mark.detach().cpu().numpy()
                    masks.append(mask_record.detach().cpu().numpy())
                    x_marks.append(batch_x_mark)
                    preds.append(target_outputs.detach().cpu().numpy())
                    trues.append(x_target.detach().cpu().numpy())

                target_outputs = target_outputs.unsqueeze(-1)
                x_target = x_target.unsqueeze(-1)

                target_outputs = self.patchify(target_outputs, num_patches, True)
                x_target = self.patchify(x_target, num_patches, True)
                x_time = self.patchify(x_time, num_patches)

                loss, loss_time, loss_target = self.forward_loss(x_time, x_target, time_outputs, target_outputs, mask_record)
                train_loss.append(loss.item())

                # for name, param in self.model.named_parameters():
                #     if param.grad is None:
                #         print(name)
                # 用一个列表存储所有参数的名字
                param_names = []

                for name, param in self.model.named_parameters():
                    param_names.append(name)

                # 列出未接收到梯度的参数索引
                unused_param_indices = [
                    26, 27, 42, 43, 44, 45, 46, 47, 48, 49,
                    69, 70, 85, 86, 87, 88, 89, 90, 91, 92,
                    112, 113, 128, 129, 130, 131, 132, 133,
                    134, 135, 155, 156, 171, 172, 173, 174,
                    175, 176, 177, 178
                ]

                # 打印未接收到梯度的参数名字
                for idx in unused_param_indices:
                    print(f"Parameter index {idx}: {param_names[idx]}")

                # 设置环境变量以获取更多调试信息
                os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    if self.device == 0:
                        print_log(log,
                                  "\tepoch: {1} | iters: {0} | loss: {2:.7f} | speed: {3:.4f}s/iter | left time: {4:.4f}s".
                                  format(i + 1, epoch + 1, loss.item(), speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if epoch == self.args.train_epochs - 1 and self.device == 0:
                    maes_time.append(loss_time.detach().item())
                    maes_target.append(loss_target.detach().item())
                    maes.append(loss.detach().item())

                loss.backward()
                model_optim.step()

            if self.device == 0:
                print_log(log, "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if self.device == 0:
                print_log(log, "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Best Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, best_loss))

            if self.device == 0:
                lr_scheduler.step(train_loss)
                current_lr = model_optim.param_groups[0]['lr']
                print_log(log, "Epoch: {} current lr: {}".format(epoch + 1, current_lr))
                writer.add_scalar(scalar_value=train_loss, global_step=epoch+1, tag='Loss/train')

            if train_loss < best_loss:
                best_loss = train_loss
                if self.device == 0:
                    print_log(log, f'Saving state ...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),  # 如果使用了学习率调度器
                    'loss': train_loss,
                }, path + '/' + 'checkpoint.pth')

            if epoch == self.args.train_epochs - 1:
                break

        # ==================保存训练过程中间结果===================================================
        if self.device == 0:
            # folder_path = './train_results/' + setting + '/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            folder_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
            preds = np.array(preds)
            trues = np.array(trues)
            x_marks = np.array(x_marks)
            maes = np.array(maes)
            maes_time = np.array(maes_time)
            maes_target = np.array(maes_target)
            masks = np.array(masks)

            print('shape:', preds.shape, trues.shape, x_marks.shape, maes.shape, maes_time.shape, maes_target.shape, masks.shape)
            preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
            trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
            x_marks = x_marks.reshape((-1, x_marks.shape[-2], x_marks.shape[-1]))
            masks = masks.reshape((-1, masks.shape[-1]))
            print('shape:', preds.shape, trues.shape, x_marks.shape, maes.shape, maes_time.shape, maes_target.shape, masks.shape)

            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'x_marks.npy', x_marks)
            np.save(folder_path + 'maes.npy', maes)
            np.save(folder_path + 'maes_time.npy', maes_time)
            np.save(folder_path + 'maes_target.npy', maes_target)
            np.save(folder_path + 'masks.npy', masks)
            # ==================保存训练过程中间结果===================================================

            best_model_path = path + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model
