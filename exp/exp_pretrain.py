import pickle

import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
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
        with open(self.args.adj_path, 'rb') as f:
            pickle_data = pickle.load(f, encoding="latin1")
        adj_mx = pickle_data[2]
        adj = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
        # num_nodes = len(pickle_data[0])
        supports = [torch.tensor(i).to(self.device) for i in adj]
        # .float(): 将模型的参数和张量转换为浮点数类型
        model = self.model_dict[self.args.model].Model(self.args, supports=supports, device=self.device).float()

        msg = model.load_state_dict(torch.load(self.args.best_model_path), strict=True)
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def patchify(self, series, num_patches):
        """
        time: (batch_size, n_nodes, seq_len, 4)
        target: (batch_size, n_nodes, seq_len, 1)

        patch_time: (batch_size, n_nodes, num_patches, patch_size * 4)
        patch_target: (batch_size, n_nodes, num_patches, patch_size * 1)
        """
        batch_size, n_nodes, seq_len, n_feats = series.shape
        p = num_patches
        assert seq_len % p == 0

        patch_size = seq_len // p
        patch_series = series.reshape(batch_size, n_nodes, p, patch_size, n_feats)
        patch_series = series.reshape(batch_size, n_nodes, p, patch_size * n_feats)

        return patch_series

    def unpatchify(self, patches):
        """
        patch_time: (batch_size, n_nodes, num_patches, patch_size * 4)
        patch_target: (batch_size, n_nodes, num_patches, patch_size * 1)

        time: (batch_size, n_nodes, seq_len, 4)
        target: (batch_size, n_nodes, seq_len, 1)
        """
        batch_size, n_nodes, num_patches, patch_size_feats = patches.shape
        p = num_patches
        patch_size = self.args.seq_len // p
        n_feats = patch_size_feats // patch_size

        patches = patches.reshape(batch_size, n_nodes, p, patch_size, n_feats)
        unpatches = patches.reshape(batch_size, n_nodes, p * patch_size, n_feats)

        return unpatches

    def forward_loss(self, y_time, y_target, time_outputs, target_outputs, mask):
        """
        time_outputs: [batch_size, n_nodes, num_patches, patch_size * n_feats]
        target_outputs: [batch_size, n_nodes, num_patches, patch_size * n_feats]
        mask: [batch_size, n_nodes, num_patches]
        """
        batch_size, n_nodes, num_patches, _ = y_time.shape

        mask0 = (y_target != 0.0)
        mask0 = mask0.float()
        mask0 /= torch.mean((mask0))
        mask0 = torch.where(torch.isnan(mask0), torch.zeros_like(mask0), mask0)

        loss_target = torch.abs(target_outputs - y_target)
        loss_target = loss_target * mask0
        loss_target = loss_target.mean(dim=-1)  # [batch_size, n_nodes, num_patches] mean loss per patch
        loss_target = torch.where(torch.isnan(loss_target), torch.zeros_like(loss_target), loss_target)
        loss_target = (loss_target * mask).sum() / mask.sum()

        loss_time = torch.abs(time_outputs - y_time).mean(dim=-1)
        loss_time = loss_time * mask
        loss_time = (loss_time * mask).sum() / mask.sum()

        loss = loss_time + loss_target

        return loss, loss_time, loss_target

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # path = '/mnt/workspace/'  # 使用阿里天池跑代码的路径
        path = '/kaggle/working/' # 使用kaggle跑实验时的路径

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("Model = %s" % str(self.model))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # tensorboard_path = os.path.join('./runs/{}/'.format(setting))
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)
        # tensorboard_path = '/mnt/workspace/'  # 使用阿里天池跑实验时的路径
        tensorboard_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        writer = SummaryWriter(log_dir=tensorboard_path)

        step = 0

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
            train_pbar = tqdm(train_loader, position=0, leave=True)  # 可视化训练的过程

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_pbar):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        time_outputs, target_outputs, mask_record = self.model(batch_x)

                else:
                    time_outputs, target_outputs, mask_record = self.model(batch_x)

                """
                time_outputs, y_time: [batch_size, n_nodes, num_patches, patch_size * n_feats]
                target_outputs, y_target: [batch_size, n_nodes, num_patches, patch_size * n_feats]
                mask: [batch_size, n_nodes, num_patches]
                """
                _, _, num_patches, _ = time_outputs.shape
                target_outputs = self.unpatchify(target_outputs)
                batch_size, n_nodes, seq_len, n_feats = target_outputs.shape

                if n_feats == 1:
                    target_outputs = target_outputs.squeeze(-1)

                target_outputs = target_outputs.transpose(1, 2)
                y_target = batch_x[:, :, :, 0]
                y_time = batch_x[:, :, :, 1:]
                if train_data.scale and self.args.inverse:
                    target_outputs = train_data.inverse_transform(target_outputs.reshape(-1, n_nodes)).reshape(
                        batch_size,
                        seq_len, n_nodes)
                    y_target = train_data.inverse_transform(y_target.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                   seq_len, n_nodes)

                if epoch == self.args.train_epochs - 1:
                    batch_x_mark = batch_x_mark.detach().cpu().numpy()
                    mask = mask_record.unsqueeze(-1).repeat(1, 1, 1, self.args.patch_size)
                    mask = self.unpatchify(mask)
                    mask = mask.transpose(1, 2)
                    if mask.shape[-1] == 1:
                        mask = mask.squeeze(-1)
                    masks.append(mask.detach().cpu().numpy())
                    x_marks.append(batch_x_mark)
                    preds.append(target_outputs.detach().cpu().numpy())
                    trues.append(y_target.detach().cpu().numpy())

                if n_feats == 1:
                    target_outputs = target_outputs.unsqueeze(-1)
                    y_target = y_target.unsqueeze(-1)

                target_outputs = target_outputs.transpose(1, 2)
                y_target = y_target.transpose(1, 2)
                y_time = y_time.transpose(1, 2)

                target_outputs = self.patchify(target_outputs, num_patches)
                y_target = self.patchify(y_target, num_patches)
                y_time = self.patchify(y_time, num_patches)

                loss, loss_time, loss_target = self.forward_loss(y_time, y_target, time_outputs, target_outputs, mask_record)
                train_loss.append(loss.item())

                if epoch == self.args.train_epochs - 1:
                    maes_time.append(loss_time.detach().item())
                    maes_target.append(loss_target.detach().item())
                    maes.append(loss.detach().item())

                train_pbar.set_description(f'Epoch [{epoch + 1}/{self.args.train_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            print(f'Saving model ...')
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            writer.add_scalar(scalar_value=train_loss, global_step=step, tag='Loss/train')
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
        # ==================保存训练过程中间结果===================================================
        folder_path = './train_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # folder_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        preds = np.array(preds)
        trues = np.array(trues)
        x_marks = np.array(x_marks)
        maes = np.array(maes)
        maes_time = np.array(maes_time)
        maes_target = np.array(maes_target)
        masks = np.array(masks)

        print('shape:', preds.shape, trues.shape, x_marks.shape, maes.shape, maes_time.shape, maes_target.shape, masks.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        x_marks = x_marks.reshape(-1, x_marks.shape[-2], x_marks.shape[-1])
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
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
