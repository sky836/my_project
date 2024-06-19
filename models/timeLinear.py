import torch
from torch import nn

from layers.Embed import PatchEmbedding


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


class Encoder(nn.Module):
    def __init__(self, patch_size, time_channel, d_model):
        super(Encoder, self).__init__()
        self.patchTime_embedding = PatchEmbedding(patch_size, time_channel, d_model, norm_layer=None)

    def forward(self, encoder_target):
        # [batch_size, seq_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, seq_len]
        encoder_target = encoder_target.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, seq_len] => [batch_size, n_nodes, seq_len, d_model]
        encoder_target = self.patchTime_embedding(encoder_target).permute(0, 1, 3, 2)
        return encoder_target


class Decoder(nn.Module):
    def __init__(self, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_time, encoder_target, decoder_time):
        """
        encoder_time:[batch_size, seq_len, n_nodes, 4]
        encoder_target:[batch_size, n_nodes, seq_len, d_model]
        decoder_time:[batch_size, pred_len, n_nodes, 4]
        """
        encoder_time, decoder_time = encoder_time.transpose(1, 2), decoder_time.transpose(1, 2)
        cross_timeSimlarity = torch.matmul(decoder_time, encoder_time.transpose(2, 3))
        norm_encoder_time = encoder_time.norm(dim=-1, keepdim=True)
        norm_decoder_time = decoder_time.norm(dim=-1, keepdim=True)
        cross_timeSimlarity = cross_timeSimlarity / (norm_decoder_time * norm_encoder_time.transpose(2, 3))
        cross_timeSimlarity = torch.softmax(cross_timeSimlarity, dim=-1)
        cross_timeSimlarity = self.dropout(cross_timeSimlarity)
        target_output = torch.matmul(cross_timeSimlarity, encoder_target)
        return cross_timeSimlarity, target_output


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = Encoder(configs.patch_size, configs.target_channel, configs.d_model)
        self.decoder = Decoder()
        self.fc = PoswiseFeedForwardNet(configs.d_model)
        self.output_layer = nn.Linear(configs.d_model, configs.target_channel)

    def forward(self, encoder_time, encoder_target, decoder_time):
        """
        encoder_time:[batch_size, seq_len, n_nodes, 4]
        encoder_target:[batch_size, seq_len, n_nodes, 1]
        decoder_time:[batch_size, pred_len, n_nodes, 4]
        return:
        cross_target:[batch_size, pred_len, n_nodes, target_channel]
        """
        # [batch_size, n_nodes, seq_len, d_model]
        encoder_target = self.encoder(encoder_target)
        # cross_timeSimlarity: [batch_size, n_nodes, pred_len, seq_len]
        # cross_target: [batch_size, n_nodes, pred_len, d_model]
        cross_timeSimlarity, cross_target = self.decoder(encoder_time, encoder_target, decoder_time)
        cross_target = self.fc(cross_target)
        cross_target = self.output_layer(cross_target)
        return cross_timeSimlarity, cross_target.transpose(1, 2)