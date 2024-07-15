import numpy as np
import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

from models.swin_transformer import PatchEmbed


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        # self.mlp = linear(c_in,c_out)
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # x:[batch_size, seq_len, n_nodes, d_model]
        # 在seq_len维度进行gcn，而不是在d_model维度：[n_nodes, seq_len]*[n_nodes, n_nodes] => [n_nodes, seq_len]
        # x = x.transpose(1, 3)
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        # h = h.transpose(1, 3)
        return h


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False, target_dim=None):
        super().__init__()

        self.model_dim = model_dim
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.value_dim = self.head_dim

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)

        if target_dim is None:
            self.FC_V = nn.Linear(model_dim, model_dim)

            self.out_proj = nn.Linear(model_dim, model_dim)
        else:
            self.FC_V = nn.Linear(target_dim, target_dim)

            self.out_proj = nn.Linear(target_dim, target_dim)
            self.value_dim = target_dim // num_heads

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
        value = torch.cat(torch.split(value, self.value_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score_no_softmax = attn_score

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        if self.target_dim is not None:
            attn_score = attn_score.unsqueeze(1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out, attn_score_no_softmax, value


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

    def forward(self, Q, K, V, dim=-2):
        Q = Q.transpose(dim, -2)
        K = K.transpose(dim, -2)
        V = V.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = Q
        out, _, _ = self.attn(Q, K, V)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class FC(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, dropout):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.ln(x + residual)
        return x


class MergeAttentionLayer(nn.Module):
    def __init__(
        self, time_dim, target_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn_time = AttentionLayer(time_dim, num_heads, mask)
        self.attn_target = AttentionLayer(target_dim, num_heads, mask)
        self.target_proj = nn.Linear(target_dim, target_dim)

        self.fc_time = FC(time_dim, feed_forward_dim, dropout)
        self.fc_target = FC(target_dim, feed_forward_dim, dropout)

        self.ln1 = nn.LayerNorm(time_dim)
        self.ln2 = nn.LayerNorm(target_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, time_features, target_features, dim=-2):
        batch_size = time_features.shape[0]
        time_features, target_features = time_features.transpose(dim, -2), target_features.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual_time, residual_target = time_features, target_features
        time_features, attn_time_no_softmax, _ = self.attn_time(time_features, time_features, time_features)
        _, attn_target_no_softmax, value = self.attn_target(target_features, target_features, target_features)
        merge_attn = attn_time_no_softmax.unsqueeze(1) + attn_target_no_softmax
        merge_attn = torch.softmax(merge_attn, dim=-1)
        target_features = merge_attn @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        target_features = torch.cat(
            torch.split(target_features, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        target_features = self.target_proj(target_features)

        time_features = self.dropout1(time_features)
        time_features = self.ln1(time_features + residual_time)
        target_features = self.dropout2(target_features)
        target_features = self.ln2(target_features + residual_target)

        time_features = self.fc_time(time_features)
        target_features = self.fc_target(target_features)

        time_features = time_features.transpose(dim, -2)
        target_features = target_features.transpose(dim, -2)
        return time_features, target_features


class CrossAttentionLayer(nn.Module):
    def __init__(
        self, time_dim, target_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.crossAtten = AttentionLayer(time_dim, num_heads, target_dim=target_dim)
        self.fc = FC(target_dim, feed_forward_dim, dropout)

        self.ln = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y_time, x_time, x_target):
        y_time, x_time, x_target = y_time.transpose(1, -2), x_time.transpose(1, -2), x_target.transpose(1, -2)
        residual_y, residual_x = y_time, x_time
        y_target, cross_attn_no_softmax, _ = self.crossAtten(y_time, x_time, x_target)
        y_target = self.dropout(y_target)
        y_target = self.ln(y_target)

        y_target = self.fc(y_target)
        y_target = y_target.transpose(1, -2)
        return y_target


class Decoder_layer(nn.Module):
    def __init__(self, time_dim, target_dim, supports, supports_len, num_heads, feed_forward_dim, dec_layers, dropout):
        super().__init__()
        self.time_dim = time_dim
        self.target_dim = target_dim
        self.dec_layers = dec_layers

        self.self_attn_time = nn.ModuleList(
            [
                SelfAttentionLayer(time_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(dec_layers)
            ]
        )

        self.cross_attn_time = nn.ModuleList(
            [
                SelfAttentionLayer(time_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(dec_layers)
            ]
        )

        self.cross_target = CrossAttentionLayer(time_dim, target_dim, feed_forward_dim, num_heads, dropout)

        # GCN special
        # self.supports = supports
        # self.supports_len = supports_len
        # self.gconvs = nn.ModuleList(
        #     [
        #         gcn(self.target_dim, self.target_dim, dropout, support_len=self.supports_len)
        #         for _ in range(dec_layers)
        #     ]
        # )
        # self.self_attn_layers_s = nn.ModuleList(
        #     [
        #         SelfAttentionLayer(target_dim, feed_forward_dim, num_heads, dropout)
        #         for _ in range(dec_layers)
        #     ]
        # )

    def forward(self, y_time, x_time, x_target):
        # y_time: (batch_size, in_steps, num_nodes, d_model)
        for i in range(self.dec_layers):
            y_time = self.self_attn_time[i](y_time, y_time, y_time, dim=1)
            y_time = self.cross_attn_time[i](y_time, x_time, x_time, dim=1)

        y_target = self.cross_target(y_time, x_time, x_target)

        # for i in range(self.dec_layers):
        #     y_target = self.self_attn_layers_s[i](y_target, y_target, y_target, dim=2)
        #     y_target = self.gconvs[i](y_target, self.supports)

        return y_target


class Model(nn.Module):
    def __init__(self, configs, supports=None):
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
            configs.input_embedding_dim
            + configs.tod_embedding_dim
            + configs.dow_embedding_dim
            + configs.spatial_embedding_dim
            + configs.adaptive_embedding_dim
        )
        self.time_dim = (configs.tod_embedding_dim + configs.dow_embedding_dim)
        self.target_dim = (configs.input_embedding_dim + configs.adaptive_embedding_dim)
        self.num_heads = configs.n_heads
        self.num_layers = configs.num_layers
        self.dec_layers = configs.d_layers
        self.use_mixed_proj = configs.use_mixed_proj
        self.patch_size = configs.patch_size
        self.num_nodes = configs.num_nodes
        self.supports = supports
        self.dropout = configs.dropout
        if self.supports is None:
            self.supports = []

        # self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.patch_emb = PatchEmbed(seq_len=self.in_steps, patch_size=self.patch_size,
                                    in_chans=self.input_dim, embed_dim=self.input_embedding_dim, norm_layer=nn.LayerNorm)
        self.num_patches = self.patch_emb.num_patches
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
                nn.Parameter(torch.empty(self.num_patches, self.num_nodes, self.adaptive_embedding_dim))
            )

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                (self.num_patches) * self.target_dim + self.spatial_embedding_dim, self.out_steps * self.output_dim
            )
            # self.output_proj = nn.Linear(
            #     self.target_dim + self.spatial_embedding_dim, out_steps * output_dim
            # )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.target_dim, self.output_dim)

        # ===================================encoding special=============================================
        self.merge_attn_layers = nn.ModuleList(
            [
                MergeAttentionLayer(self.time_dim, self.target_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        # GCN special
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        self.adp = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes))
        nn.init.xavier_uniform_(self.adp)
        self.supports_len += 1
        self.supports = self.supports + [self.adp]

        # self.gconvs = nn.ModuleList(
        #     [
        #         gcn(self.target_dim, self.target_dim, dropout, support_len=self.supports_len)
        #         for _ in range(num_layers)
        #     ]
        # )

        # self.self_attn_layers_s = nn.ModuleList(
        #     [
        #         SelfAttentionLayer(self.target_dim, self.feed_forward_dim, self.num_heads, self.dropout)
        #         for _ in range(self.num_layers)
        #     ]
        # )

        # ===================================decoding special=============================================
        self.decoder = Decoder_layer(self.time_dim, self.target_dim, self.supports, self.supports_len, self.num_heads,
                                     self.feed_forward_dim, self.dec_layers, self.dropout)

    def encoding(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
            tod = tod[..., 0]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
            dow = dow[..., 0]
        x = x[..., : self.input_dim]

        # x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        x = self.patch_emb(x)
        patch_size = self.patch_emb.patch_size
        target_features = [x]
        time_features = []
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            time_features.append(tod_emb[:, ::patch_size])
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            time_features.append(dow_emb[:, ::patch_size])

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            target_features.append(adp_emb)
        target_features = torch.cat(target_features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        time_features = torch.cat(time_features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for i in range(self.num_layers):
            time_features, target_features = self.merge_attn_layers[i](time_features, target_features, dim=1)
            # target_features = self.gconvs[i](target_features, self.supports)

        # for i in range(self.num_layers):
        #     target_features = self.self_attn_layers_s[i](target_features, target_features, target_features, dim=2)
            # target_features = self.gconvs[i](target_features, self.supports)
        # (batch_size, in_steps, num_nodes, model_dim)
        return time_features, target_features

    def decoding(self, time_enc, target_enc, y):
        # y: (batch_size, in_steps, num_nodes, tod+dow=2)
        batch_size = y.shape[0]
        if self.tod_embedding_dim > 0:
            tod = y[..., 1]
            tod = tod[..., 0]
        if self.dow_embedding_dim > 0:
            dow = y[..., 2]
            dow = dow[..., 0]

        time_features = []
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            time_features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            time_features.append(dow_emb)
        time_features = torch.cat(time_features, dim=-1)  # (batch_size, in_steps, num_nodes, time_dim)

        y_target = self.decoder(time_features, time_enc, target_enc)

        return y_target

    def forward(self, x, y):
        batch_size, _, num_nodes, _ = x.shape
        time_features, target_features = self.encoding(x)

        y_target = self.decoding(time_features, target_features, y)
        if self.num_patches != self.out_steps:
            target_features = target_features.transpose(1, 2).reshape(batch_size, num_nodes, -1)
            y_target = y_target.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        target_features = torch.cat((target_features, y_target), dim=-1)  # (batch_size, in_steps, num_nodes, model_dim * 2)

        if self.num_patches == self.out_steps:
            target_features = target_features.transpose(1, 2).reshape(batch_size, num_nodes, -1)

        if self.spatial_embedding_dim > 0:
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            target_features = torch.cat([target_features, node_emb], dim=-1)  # B, N, nP*dm*2+dN


        # target_features = self.predict(target_features)
        out = target_features

        out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = Model(207, 12, 12)
    summary(model, [(1, 12, 207, 3), (1, 12, 207, 3)])
