import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class temporalEmbedding(nn.Module):
    def __init__(self, D):
        super(temporalEmbedding, self).__init__()
        self.ff_te = FeedForward([31, D, D])

    def forward(self, TE, T=288, W=7):
        '''
        TE:[B,T,2]
        '''
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % W, W)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B,T,295/55]
        TE = TE.unsqueeze(dim=2)  # [B,T,1,295]
        TE = self.ff_te(TE)  # [B,T,1,F]

        return TE  # [B,T,1,F]


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


class sparseSpatialAttention(nn.Module):
    def __init__(self, hidden_size, log_samples):
        super(sparseSpatialAttention, self).__init__()
        self.qfc = FeedForward([hidden_size, hidden_size])
        self.kfc = FeedForward([hidden_size, hidden_size])
        self.vfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])

        self._hidden_size = hidden_size
        self._log_samples = log_samples

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size, hidden_size, hidden_size], True)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x, adj, eigvec, eigvalue):
        '''
        [B,T,N,D]
        '''
        # add spatial positional encoding
        x_ = x + torch.matmul(eigvec.transpose(0, 1).squeeze(-1), torch.diag_embed(eigvalue))

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        B, T, N, D = Q.shape

        # use gat results to reduce Q
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), adj, :]
        V_expand = V.unsqueeze(-3).expand(B, T, N, N, D)
        V_sample = V_expand[:, :, torch.arange(N).unsqueeze(1), adj, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1))
        GAT_results = torch.matmul(Q_K_sample, V_sample).squeeze(-2)
        M = self.proj(GAT_results).squeeze(-1)
        samples = int(self._log_samples * math.log(N, 2))
        M_top = M.topk(samples, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(T)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        Q_K /= (self._hidden_size ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        # copy operation
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_top.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)

        value = self.ofc(value) + x_
        value = self.ln(value)
        return self.ff(value)


class temporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(temporalAttention, self).__init__()
        self.qfc = FeedForward([hidden_size, hidden_size])
        self.kfc = FeedForward([hidden_size, hidden_size])
        self.vfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])
        self._hidden_size = hidden_size

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size, hidden_size, hidden_size], True)

    def forward(self, x, te, Mask=True):
        '''
        x:[B,T,N,F]
        te:[B,T,N,F]
        '''
        x += te

        query = self.qfc(x).permute(0, 2, 1, 3)  # [B,N,T,F]
        key = self.kfc(x).permute(0, 2, 3, 1)
        value = self.vfc(x).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)  # [B,N,T,T]
        attention /= (self._hidden_size ** 0.5)  # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(batch_size, num_vertexs, 1, 1)  # [B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attention).to(x.device)  # [B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1)  # [B,N,T,T]

        value = torch.matmul(attention, value).permute(0, 2, 1, 3)  # [B,N,T,d]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class temporalConvNet(nn.Module):
    def __init__(self, hidden_size, kernel_size=2, dropout=0.2, levels=1):
        super(temporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(hidden_size, hidden_size, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh.transpose(1, 3)).transpose(1, 3)
        return xh


class adaptiveFusion(nn.Module):
    def __init__(self, hidden_size):
        super(adaptiveFusion, self).__init__()
        self.qlfc = FeedForward([hidden_size, hidden_size])
        self.khfc = FeedForward([hidden_size, hidden_size])
        self.vhfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])
        self._hidden_size = hidden_size

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size, hidden_size, hidden_size], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        tp: [B,T,1,F]
        '''
        T = xl.shape[1]
        xl += te[:, -T:]
        xh += te[:, -T:]

        query = self.qlfc(xl).permute(0, 2, 1, 3)  # [B,N,T,F]
        keyh = torch.relu(self.khfc(xh)).permute(0, 2, 3, 1)
        valueh = torch.relu(self.vhfc(xh)).permute(0, 2, 1, 3)

        attentionh = torch.matmul(query, keyh)  # [B,N,T,T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(batch_size, num_vertexs, 1, 1)  # [B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)  # [B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)
        attentionh /= (self._hidden_size ** 0.5)  # scaled
        attentionh = F.softmax(attentionh, -1)  # [B,N,T,T]

        value = torch.matmul(attentionh, valueh).permute(0, 2, 1, 3)
        value = self.ofc(value)
        value = value + xl  # + xh

        value = self.ln(value)

        return self.ff(value)


class dualEncoder(nn.Module):
    def __init__(self, hidden_size, log_samples, adj_gat, graphwave):
        super(dualEncoder, self).__init__()
        self.tcn = temporalConvNet(hidden_size)
        self.tatt = temporalAttention(hidden_size)

        self.ssal = sparseSpatialAttention(hidden_size, log_samples)
        self.ssah = sparseSpatialAttention(hidden_size, log_samples)

        eigvalue = torch.from_numpy(graphwave[0].astype(np.float32))
        self.eigvalue = nn.Parameter(eigvalue, requires_grad=True)
        self.eigvec = torch.from_numpy(graphwave[1].astype(np.float32)).transpose(0, 1).unsqueeze(-1)
        self.adj = torch.from_numpy(adj_gat)

    def forward(self, xl, xh, te):
        xl = self.tatt(xl, te)
        xh = self.tcn(xh)

        spa_statesl = self.ssal(xl, self.adj.to(xl.device), self.eigvec.to(xl.device), self.eigvalue.to(xl.device))
        spa_statesh = self.ssah(xh, self.adj.to(xl.device), self.eigvec.to(xl.device), self.eigvalue.to(xl.device))
        xl = spa_statesl + xl
        xh = spa_statesh + xh

        return xl, xh


def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

def loadGraph(adj_mx, hs, ls):
    graphwave = get_eigv(adj_mx+np.eye(adj_mx.shape[0]), hs)
    sampled_nodes_number = int(np.around(math.log(adj_mx.shape[0]))+2)*ls
    graph = csr_matrix(adj_mx)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    adj_gat = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
    return adj_gat, graphwave


class Model(nn.Module):
    """
    Paper: When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks
    Link: https://ieeexplore.ieee.org/document/10184591
    Ref Official Code: https://github.com/LMissher/STWave
    Hints: PyWavelets and pytorch_wavelets packages are needed
    """

    def __init__(self, configs, hidden_size=128, log_samples=1, layers=2, wave_type="coif1", wave_levels=2):
        super().__init__()
        self.start_emb_l = FeedForward([configs.output_dim, hidden_size, hidden_size])
        self.start_emb_h = FeedForward([configs.output_dim, hidden_size, hidden_size])
        self.te_emb = temporalEmbedding(hidden_size)
        self.num_nodes = configs.num_nodes

        adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.len_row = 32
        self.len_column = 32
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in range(self.len_row):
            for j in range(self.len_column):
                index = i * self.len_column + j  # grid_id
                for d in dirs:
                    nei_i = i + d[0]
                    nei_j = j + d[1]
                    if nei_i >= 0 and nei_i < self.len_row and nei_j >= 0 and nei_j < self.len_column:
                        nei_index = nei_i * self.len_column + nei_j  # neighbor_grid_id
                        adj_mx[index][nei_index] = 1
                        adj_mx[nei_index][index] = 1

        adj_gat, gwv = loadGraph(adj_mx, 128, 1)

        self.dual_encoder = nn.ModuleList(
            [dualEncoder(hidden_size, log_samples, adj_gat, gwv) for i in range(layers)])
        self.adaptive_fusion = adaptiveFusion(hidden_size)

        self.pre_l = nn.Conv2d(configs.seq_len, configs.pred_len, (1, 1))
        self.pre_h = nn.Conv2d(configs.seq_len, configs.pred_len, (1, 1))

        self.end_emb = FeedForward([hidden_size, hidden_size, configs.output_dim])
        self.end_emb_l = FeedForward([hidden_size, hidden_size, configs.output_dim])

        self.td = configs.steps_per_day
        self.dw = 7
        self.id = configs.input_dim
        self.wt = wave_type
        self.wl = wave_levels

    def forward(self, history_data, future_data):
        '''
        x:[B,T,N,D]
        '''
        x = history_data
        te = torch.cat([x[:, :, 0, 2:3] * self.td, x[:, :, 0, 3:] * self.dw], -1)
        ADD = torch.arange(te.shape[1]).to(x.device).unsqueeze(0).unsqueeze(2) + 1
        TEYTOD = (te[:, -1:, 0:1] + ADD) % self.td
        TEYDOW = (torch.floor((te[:, -1:, 0:1] + ADD) / self.td) + te[..., 1:2]) % self.dw
        te = torch.cat([te, torch.cat([TEYTOD, TEYDOW], -1)], 1)
        te = te[..., [1, 0]]

        inputs = x[..., :self.id]
        xl, xh = disentangle(inputs[..., 0:2].cpu().numpy(), self.wt, self.wl)

        xl, xh, TE = self.start_emb_l(xl.to(x.device)), self.start_emb_h(xh.to(x.device)), self.te_emb(te, self.td,
                                                                                                       self.dw)

        for enc in self.dual_encoder:
            xl, xh = enc(xl, xh, TE[:, :xl.shape[1], :, :])

        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)

        hat_y = self.adaptive_fusion(hat_y_l, hat_y_h, TE[:, xl.shape[1]:, :, :])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)

        label_yl, _ = disentangle(future_data[..., 0:1].cpu().numpy(), self.wt, self.wl)

        return hat_y, hat_y_l, label_yl.to(x.device)


def disentangle(x, w, j):
    T = x.shape[1]
    x = x.transpose(0, 3, 2, 1)  # [B,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef) - 1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef) - 1):
        coefh.append(coef[i + 1])
    xl = pywt.waverec(coefl, w).transpose(0, 3, 2, 1)[:, :T]
    xh = pywt.waverec(coefh, w).transpose(0, 3, 2, 1)[:, :T]

    return torch.from_numpy(xl), torch.from_numpy(xh)