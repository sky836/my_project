import argparse

import numpy as np
import torch.nn as nn
import torch
from torchinfo import summary


class FullAttention(nn.Module):
    def __init__(self, n_heads, d_feature, d_keys=None, d_values=None):
        super(FullAttention, self).__init__()
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.model_dim = d_feature

        self.query_projection = nn.Linear(d_feature, d_feature)
        self.key_projection = nn.Linear(d_feature, d_feature)
        self.value_projection = nn.Linear(d_feature, d_feature)

        self.out_proj = nn.Linear(d_feature, d_feature)

    def forward(self, Q, K, V, attn_mask=None):
        if attn_mask is not None:
            device = Q.device
            attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1).to(device)
        batch_size, n_nodes, q_len, _ = Q.shape
        _, _, k_len, _ = K.shape
        # [batch_size, n_nodes, q_len, d_keys*n_heads] => [batch_size, n_nodes, n_heads, q_len, d_keys]
        queries = self.query_projection(Q).view(batch_size, n_nodes, q_len, self.n_heads, -1).transpose(2, 3)
        keys = self.key_projection(K).view(batch_size, n_nodes, k_len, self.n_heads, -1).transpose(2, 3)
        values = self.value_projection(V).view(batch_size, n_nodes, k_len, self.n_heads, -1).transpose(2, 3)

        # [batch_size, n, n_heads, q_len, k_len]
        attn_score = torch.matmul(queries, keys.transpose(3, 4)) / np.sqrt(self.model_dim)

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask, -1e9)
        # softmax
        attn_score = torch.softmax(attn_score, dim=-1)

        attn_value = torch.matmul(attn_score, values).transpose(2, 3).reshape(batch_size, n_nodes, q_len, -1)
        attn_value = self.out_proj(attn_value)

        return attn_value

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
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

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

        # self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.attn = FullAttention(d_feature=model_dim, n_heads=num_heads)
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
    def __init__(self, configs):
        super().__init__()
        self.num_nodes = configs.num_nodes
        self.in_steps = configs.seq_len
        self.out_steps = configs.pred_len
        self.steps_per_day = configs.steps_per_day
        self.input_dim = configs.input_dim
        self.output_dim = configs.output_dim
        self.input_embedding_dim = configs.input_embedding_dim
        self.tod_embedding_dim = configs.tod_embedding_dim
        self.dow_embedding_dim = configs.dow_embedding_dim
        self.spatial_embedding_dim = configs.spatial_embedding_dim
        self.adaptive_embedding_dim = configs.adaptive_embedding_dim
        self.feed_forward_dim = configs.feed_forward_dim
        self.model_dim = (
            self.input_embedding_dim
            + self.tod_embedding_dim
            + self.dow_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )
        self.num_heads = configs.n_heads
        self.num_layers = configs.num_layers
        self.use_mixed_proj = configs.use_mixed_proj
        self.dropout = configs.dropout

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))
            )

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Taformer')
    parser.add_argument('--adaptive_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--spatial_embedding_dim', type=int, default=0, help='')
    parser.add_argument('--dow_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--tod_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--input_embedding_dim', type=int, default=24, help='')
    parser.add_argument('--feed_forward_dim', type=int, default=256, help='')
    parser.add_argument('--input_dim', type=int, default=3, help='')
    parser.add_argument('--output_dim', type=int, default=1, help='')
    parser.add_argument('--steps_per_day', type=int, default=288, help='')
    parser.add_argument('--num_layers', type=int, default=3, help='')
    parser.add_argument('--use_mixed_proj', type=bool, default=True, help='')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--num_nodes', type=int, required=False, default=207, help='the nodes of dataset')
    args = parser.parse_args()
    model = Model(args)
    summary(model, [1, 12, 207, 3])
