import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.start_conv = nn.Conv2d(in_channels=configs.input_dim,
                                    out_channels=configs.feed_forward_dim,
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=configs.feed_forward_dim, hidden_size=configs.feed_forward_dim, num_layers=configs.num_layers, batch_first=True, dropout=configs.dropout)
        
        self.end_linear1 = nn.Linear(configs.feed_forward_dim, configs.feed_forward_dim//4)
        self.end_linear2 = nn.Linear(configs.feed_forward_dim//4, configs.pred_len * configs.output_dim)
        self.output_dim = configs.output_dim
        self.pred_len = configs.pred_len

    def forward(self, history_data):
        """Feedforward function of LSTM.

        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: [B, L, N, c]
        """
        x = history_data.transpose(1, 3)
        b, c, n, l = x.shape

        x = x.transpose(1,2).reshape(b*n, c, 1, l)
        x = self.start_conv(x).squeeze().transpose(1, 2)

        out, _ = self.lstm(x)
        x = out[:, -1, :]  # B*N, D

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)  # B*N, pred_len*output_dim
        x = x.reshape(b, n, self.pred_len, self.output_dim).transpose(1, 2)
        return x
