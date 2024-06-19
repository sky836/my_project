import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GWNET


class Model(nn.Module):
    def __init__(self, device, supports, num_nodes):
        super(Model, self).__init__()
        self.backbones = nn.ModuleList()
        for i in range(num_nodes):
            self.backbones.append(GWNET.Model(device, supports, num_nodes, gcn_bool=False, addaptadj=False))

    def forward(self, input):
        # [batch_size, seq_len, num_nodes, input_dim]
        batch_size, seq_len, num_nodes, input_dim = input.shape

        outputs = torch.zeros(batch_size, seq_len, num_nodes).to(input.device)
        for i in range(num_nodes):
            output = self.backbones[i](input[:, :, i, :].unsqueeze(2))
            outputs[:, :, i] = output.squeeze(-1).squeeze(-1)

        return outputs




