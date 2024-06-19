import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

from layers.Embed import PatchEmbedding, PositionalEncoding
from models.taformerPredict import gcn


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class Model(nn.Module):
    def __init__(self, configs, supports, device):
        super(Model, self).__init__()
        self.model_dim = configs.encoder_Time_embed_dim + configs.encoder_Target_embed_dim
        # 多层encoder
        self.layers = nn.ModuleList(
            [SelfAttentionLayer(self.model_dim, 4 * self.model_dim, configs.n_heads, configs.dropout) for _ in range(configs.e_layers)])
        encoder_patches = configs.seq_len // configs.patch_size
        # patch embedding
        self.patch_embedding = PatchEmbedding(configs.patch_size, configs.time_channel + configs.target_channel,
                                                      self.model_dim, norm_layer=None)
        self.positional_embedding = PositionalEncoding(self.model_dim, dropout=0.1)

        # =============================GCN special=================================
        self.gcn_bool = configs.gcn_bool
        self.supports = supports
        self.addaptadj = configs.addaptadj

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and self.addaptadj:
            if supports is None:
                self.supports = []
            self.nodevec1 = nn.Parameter(torch.randn(configs.n_nodes, 10).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(10, configs.n_nodes).to(device), requires_grad=True).to(device)
            self.supports_len += 1

        if self.gcn_bool:
            self.gconv = gcn(self.model_dim, self.model_dim, dropout=0.3, support_len=self.supports_len)
            self.layer_norm = nn.LayerNorm(self.model_dim)

        # =============================Predict special=================================
        self.pred_len = configs.pred_len
        self.encoder_outputs = nn.ModuleList()
        for i in range(configs.n_nodes):
            self.encoder_outputs.append(nn.Linear(encoder_patches * self.model_dim, configs.pred_len))

    def forward(self, x):
        """
        x:[batch_size, seq_len, n_nodes, 5]
        y: to fit pipeline of exp, there will not use it.
        """
        # [batch_size, seq_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, seq_len]
        x = x.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, n_patches] => [batch_size, n_nodes, n_patches, d_model]
        x = self.patch_embedding(x).transpose(2, 3)
        x = self.positional_embedding(x)

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        batch_size, n_nodes, n_patches, d_model = x.shape
        device = x.device
        res = torch.zeros(batch_size, n_nodes, n_patches, d_model).to(device)
        for layer in self.layers:
            x = layer(x)
            # =================================GCN=====================================
            residual = x
            if self.gcn_bool and self.supports is not None:
                batch_size, n_nodes, n_patches, d_model = x.shape
                # [batch_size, d_model, n_nodes, n_patches]
                x = x.permute(0, 3, 1, 2)
                if self.addaptadj:
                    x = self.gconv(x, new_supports)
                else:
                    x = self.gconv(x, self.supports)
                x = x.permute(0, 2, 3, 1) + residual
                x = self.layer_norm(x)
            # =================================GCN=====================================
            res = res +x  # skip connection

        res = res.reshape(batch_size, n_nodes, -1)
        # [batch_size, n_nodes, pred_len]
        outputs = torch.zeros(batch_size, n_nodes, self.pred_len).to(device)
        for i in range(n_nodes):
            outputs[:, i, :] = self.encoder_outputs[i](res[:, i, :]).squeeze(1)
        outputs = outputs.transpose(1, 2)

        return outputs

