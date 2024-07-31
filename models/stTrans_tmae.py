import torch
from torch import nn

from models import GWNET, st_pretrain


class Model(nn.Module):
    def __init__(self, configs, supports):
        super().__init__()
        # iniitalize
        self.mae = st_pretrain.Model(configs)

        self.backend = GWNET.Model(configs.num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=3,
                 out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2)

        # self.time_fc = nn.Linear(self.time_dim, configs.pred_len * 2)

        # load pre-trained model
        self.pre_trained_tmae_path = configs.best_model_path
        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.mae.load_state_dict(checkpoint_dict["model_state_dict"])

        # freeze parameters
        for param in self.mae.parameters():
            param.requires_grad = False

    def forward(self, history_data, long_history_data):
        """
        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """

        # reshape
        short_term_history = history_data  # [B, L, N, 1]
        batch_size, _, num_nodes, _ = history_data.shape

        hidden_time, hidden_target = self.mae(long_history_data)

        # enhance
        out_len = 1
        hidden_target = hidden_target[:, :, -out_len, :]
        hidden_time = hidden_time[:, :, -out_len, :]
        y_hat = self.backend(short_term_history, hidden_states=hidden_target).transpose(1, 2).unsqueeze(-1)
        # y_time = self.time_fc(hidden_time)

        return y_hat, hidden_time

