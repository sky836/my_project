import torch
from torch import nn
from torch.nn.init import trunc_normal_

from layers.Attention_Family import TimeAttentionLayer, DecoderTimeAttention
from layers.Embed import PatchEmbedding, PositionalEncoding


# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs:[batch_size, n_nodes, seq_len, d_model]
        """
        residual = inputs
        outputs = self.fc(inputs)
        return self.layer_norm(residual + outputs)
        # return residual + outputs


class EncoderLayer(nn.Module):
    def __init__(self, d_model, configs, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.timeAttention = TimeAttentionLayer(n_heads=4, d_model=d_model, d_keys=48, d_values=48)
        self.time_fc = PoswiseFeedForwardNet(d_model)
        self.target_fc = PoswiseFeedForwardNet(d_model)

    def forward(self, time_features, target_features):
        """
        time_features:[batch_size, n_nodes, n_patches, d_model]
        target_features:[batch_size, n_nodes, n_patches, d_model]
        """
        time_attn_score, target_attn_score, merge_attn_score, merge_attn_value, time_attn_value \
            = self.timeAttention(time_features, target_features, time_features, target_features)

        # time_attn_value = self.dropout(time_attn_value) + time_features
        # merge_attn_value = self.dropout(merge_attn_value) + target_features
        #
        # time_attn_value = self.time_norm(time_attn_value)
        # merge_attn_value = self.target_norm(merge_attn_value)

        time_attn_value = self.time_fc(time_attn_value)
        merge_attn_value = self.target_fc(merge_attn_value)

        return time_attn_score, target_attn_score, merge_attn_score, merge_attn_value, time_attn_value


class Encoder(nn.Module):
    def __init__(self, n_layers, patch_size, time_channel, d_model, target_channel, configs, dropout=0.1):
        super(Encoder, self).__init__()
        # 1. patch embedding
        self.patchTime_embedding = PatchEmbedding(patch_size, time_channel, d_model, norm_layer=None)
        self.patchTarget_embedding = PatchEmbedding(patch_size, target_channel, d_model, norm_layer=None)
        # 2. positional embedding
        # self.positionalTime_embedding = PositionalEncoding(d_model, dropout=dropout)
        self.positionalTarget_embedding = PositionalEncoding(d_model, dropout=dropout)
        # 3. 多层encoder
        self.layers = nn.ModuleList([EncoderLayer(d_model, configs) for _ in range(n_layers)])

    def forward(self, time_features, target_features):
        """
        time_features:[batch_size, seq_len, n_nodes, n_feats]
        target_features:[batch_size, seq_len, n_nodes, n_feats]
        """
        # time_features, target_features = x[:, :, :, 1:].transpose(1, 2), x[:, :, :, 0].unsqueeze(-1).transpose(1, 2)

        # [batch_size, seq_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, seq_len]
        time_features, target_features = time_features.permute(0, 2, 3, 1), target_features.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, n_patches] => [batch_size, n_nodes, n_patches, d_model]
        time_features = self.patchTime_embedding(time_features).transpose(2, 3)
        target_features = self.patchTarget_embedding(target_features).transpose(2, 3)

        # positional embedding
        # time_features = self.positionalTime_embedding(time_features)
        target_features = self.positionalTarget_embedding(target_features)

        time_attn_scores, target_attn_scores, merge_attn_scores = [], [], []
        encoder_target_outputs, encoder_time_outputs = [], []

        for layer in self.layers:
            time_attn_score, target_attn_score, merge_attn_score, target_features, time_features \
                = layer(time_features, target_features)
            time_attn_scores.append(time_attn_score.detach().cpu().numpy())
            target_attn_scores.append(target_attn_score.detach().cpu().numpy())
            merge_attn_scores.append(merge_attn_score.detach().cpu().numpy())
            encoder_time_outputs.append(time_features)
            encoder_target_outputs.append(target_features)

        return time_attn_scores, target_attn_scores, merge_attn_scores, encoder_time_outputs, encoder_target_outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 1. 第一层自注意力
        self.decoder_self_attn = DecoderTimeAttention(n_heads=4, d_model=d_model, d_keys=48, d_values=48)
        # 2. 第二层交互注意力
        self.decoder_cross_attn = DecoderTimeAttention(n_heads=4, d_model=d_model, d_keys=48, d_values=48)

        self.time_fc = PoswiseFeedForwardNet(d_model)
        self.target_fc = PoswiseFeedForwardNet(d_model)

    def forward(self, decoder_time_input, encoder_output_time, encoder_output_target):
        self_attn_score, self_time_value = self.decoder_self_attn(decoder_time_input, decoder_time_input)

        cross_attn, cross_time_value, cross_target = self.decoder_cross_attn(self_time_value, encoder_output_time, encoder_output_target)

        cross_target = self.target_fc(cross_target)
        cross_time_value = self.time_fc(cross_time_value)

        return self_attn_score, cross_time_value, cross_attn, cross_target


class Decoder(nn.Module):
    def __init__(self, n_layers, label_patch_size, time_channel, encoder_d_model, d_model, configs, dropout=0.1):
        super(Decoder, self).__init__()
        # encoder嵌入维度映射为decoder嵌入维度
        self.encTime_2_dec_emb = nn.Linear(encoder_d_model, d_model, bias=True)
        self.encTarget_2_dec_emb = nn.Linear(encoder_d_model, d_model, bias=True)
        # value embedding
        self.patchTime_embedding = PatchEmbedding(label_patch_size, time_channel, d_model, norm_layer=None)
        # self.patchTarget_embedding = PatchEmbedding(label_patch_size, target_channel, d_model, norm_layer=None)
        # positional encoding
        # self.positionalTime_embedding = PositionalEncoding(d_model, dropout=dropout)
        # self.positionalTarget_embedding = PositionalEncoding(d_model, dropout=dropout)
        # 多层decoder
        self.layers = nn.ModuleList([DecoderLayer(d_model, configs) for _ in range(n_layers)])

    # 这是decoder的mask attention要用的，当前时间后的token看不到，当前时间前的token可以看的
    # inputs的大小为[batch_size, seq_len] [batch_size, pred_len, n_nodes, x]
    def get_attn_subsequent_mask(self, inputs):
        batch_size, pred_len, n_nodes, _ = inputs.shape
        attn_shape = [batch_size, n_nodes, pred_len, pred_len]
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        return subsequent_mask.type(torch.bool)  # 大小为[batch_size, n_nodes, pred_len, pred_len]

    def forward(self, decoder_time, encoder_output_times, encoder_output_targets):
        """
        decoder_time:[batch_size, pred_len, n_nodes, n_feats]
        encoder_output_time:[batch_size, n_patches, n_nodes, d_model]
        encoder_output_target:[batch_size, n_patches, n_nodes, d_model]
        """
        # subsequent_mask = self.get_attn_subsequent_mask(decoder_time)
        # decoder_time, decoder_target = decoder_time.transpose(1, 2), decoder_target.transpose(1, 2)
        # decoder value embedding
        # [batch_size, pred_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, seq_len]
        decoder_time = decoder_time.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, n_patches] => [batch_size, n_nodes, n_patches, d_model]
        decoder_time = self.patchTime_embedding(decoder_time).transpose(2, 3)
        # decoder_target = self.patchTarget_embedding(decoder_target).transpose(2, 3)
        # decoder positional embedding
        # decoder_time = self.positionalTime_embedding(decoder_time)
        # decoder_target = self.positionalTarget_embedding(decoder_target)

        decoder_time_output = decoder_time
        decoder_target_outputs = []
        for i, layer in enumerate(self.layers):
            encoder_output_time, encoder_output_target = self.encTime_2_dec_emb(encoder_output_times[i]), \
                                                         self.encTarget_2_dec_emb(encoder_output_targets[i])
            self_attn_score, decoder_time_output, cross_attn, decoder_target_output \
                = layer(decoder_time_output, encoder_output_time, encoder_output_target)
            decoder_target_outputs.append(decoder_target_output)

        return self_attn_score, decoder_time_output, cross_attn, decoder_target_outputs


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = Encoder(configs.e_layers, configs.patch_size, configs.time_channel, configs.encoder_embed_dim,
                               configs.target_channel, configs)
        self.decoder = Decoder(configs.d_layers, configs.label_patch_size, configs.time_channel,
                               configs.encoder_embed_dim, configs.decoder_embed_dim, configs.target_channel, configs)
        encoder_patches = configs.seq_len // configs.patch_size

        # ========================predict special======================================================
        self.encoder_outputs = nn.ModuleList()
        self.decoder_outputs = nn.ModuleList()
        self.pred_len = configs.pred_len
        decoder_patches = configs.pred_len // configs.label_patch_size
        for i in range(configs.n_nodes):
            self.encoder_outputs.append(nn.Linear(encoder_patches * configs.encoder_embed_dim, configs.pred_len))
            self.decoder_outputs.append(nn.Linear(decoder_patches * configs.decoder_embed_dim, configs.pred_len))
        # ========================predict special======================================================

        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(0.5)
        self.layers = configs.e_layers

    def forward(self, x, y):
        """
        x:[batch_size, seq_len, n_nodes, 5]
        y:[batch_size, pred_len, n_nodes, 5]
        """
        decoder_time = y[..., 1:]  # [batch_size, pred_len, n_nodes, n_feats]
        encoder_time = x[..., 1:]
        encoder_target = x[..., 0].unsqueeze(-1)

        # 2. encoder
        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, encoder_time_outputs, encoder_target_outputs = \
            self.encoder(encoder_time, encoder_target)
        # 3. decoder
        # decoder_target_output: [batch_size, n_patches, n_nodes, d_model]
        decoder_self_attn_score, decoder_time_output, cross_attn, decoder_target_outputs = \
            self.decoder(decoder_time, encoder_time_outputs, encoder_target_outputs)

        encoder_target_output, decoder_target_output = encoder_target_outputs[0], decoder_target_outputs[0]
        for i in range(1, self.layers):
            encoder_target_output = encoder_target_output + encoder_target_outputs[i]
            decoder_target_output = decoder_target_output + decoder_target_outputs[i]
        batch_size, n_nodes, _, _ = encoder_target_output.shape
        encoder_target_output, decoder_target_output = \
            encoder_target_output.reshape(batch_size, n_nodes, -1), \
            decoder_target_output.reshape(batch_size, n_nodes, -1)
        # =============================predicts===========================================
        device = encoder_target_output.device
        # [batch_size, n_nodes, pred_len]
        encoder_outputs = torch.zeros(batch_size, n_nodes, self.pred_len).to(device)
        decoder_outputs = torch.zeros(batch_size, n_nodes, self.pred_len).to(device)
        for i in range(n_nodes):
            encoder_outputs[:, i, :] = self.encoder_outputs[i](encoder_target_output[:, i, :]).squeeze(1)
            decoder_outputs[:, i, :] = self.decoder_outputs[i](decoder_target_output[:, i, :]).squeeze(1)
        # =============================predicts===========================================
        target_output = self.fuse_weight * encoder_outputs + (1 - self.fuse_weight) * decoder_outputs
        target_output = target_output.transpose(1, 2)

        return encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, decoder_self_attn_score, cross_attn, target_output

    # def forward_loss(self, y, pred_target, pred_time):
    #     batch_size, pred_len, n_nodes, _ = y.shape
    #     y_time = y[:, :, :, 1:]
    #     y_target = y[:, :, :, 0]
    #
    #     # 在pred_len的最后一个位置插入一个end_token
    #     decoder_time = torch.cat([y_time, self.time_begin_token], dim=1)
    #     decoder_target = torch.cat([y_target, self.target_begin_token], dim=1)
    #
    #     loss = (pred_target - decoder_target) ** 2  # [batch_size, pred_len, n_nodes, 1]
    #     loss = loss.sum() / (batch_size*pred_len*n_nodes)
    #
    #     time_loss = (decoder_time[:, [0, pred_len-1], :, :] - pred_time[:, [0, pred_len-1], :, :]) ** 2
    #     time_loss /= 2*4
    #
    #     loss += time_loss
    #
    #     return loss
