import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F


class MLP_Base(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.fc1 = nn.Linear(n_nodes, n_nodes)
        self.fc2 = nn.Linear(n_nodes, n_nodes)

    def forward(self, x):
        # x:[batch_size, seq_len, n_nodes]
        x1 = self.fc1(x)
        x1 = torch.sin(x1)

        x2 = self.fc2(x)
        x2 = torch.cos(x2)
        return x1, x2


class MLP(nn.Module):
    def __init__(self, num_layers, n_nodes):
        super().__init__()
        self.num_layers = num_layers
        self.mlps = nn.ModuleList(
            MLP_Base(n_nodes) for _ in range(num_layers)
        )
        self.end_fc = nn.Linear(2*num_layers, 1)

    def forward(self, x):
        # x:[batch_size, seq_len, n_nodes]
        res = []
        for mlp in self.mlps:
            x1, x2 = mlp(x)
            res.append(x1.unsqueeze(-1))
            res.append(x2.unsqueeze(-1))
        res = torch.cat(res, dim=-1)
        x = self.end_fc(res)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.n_nodes = configs.num_nodes
        self.fc = MLP(6, self.n_nodes)

    def forward(self, x):
        # x:[batch_size, seq_len, n_nodes, input_dim+tod+dow=3]
        tod = x[:, :, :, 1]  # [batch_size, seq_len, n_nodes]
        batch_size, seq_len, n_nodes, _ = x.shape
        res = self.fc(tod)
        return res
