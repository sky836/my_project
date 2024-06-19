import torch
from torch import nn

from layers.Embed import PatchEmbedding, PositionalEncoding
from models.taformerPredict import Encoder


class Model(nn.Module):
    def __init__(self, configs, supports, device):
        super(Model, self).__init__()
        # ========================encoder special======================================================
        self.layers = configs.e_layers
        self.encoder = Encoder(configs.e_layers, configs.encoder_embed_dim, configs.n_nodes, device,
                               supports, configs.gcn_bool, configs.addaptadj)
        encoder_patches = configs.seq_len // configs.patch_size
        # 1. patch embedding
        self.enc_patchTime_embedding = PatchEmbedding(configs.patch_size, configs.time_channel,
                                                      configs.encoder_embed_dim, norm_layer=None)
        self.enc_patchTarget_embedding = PatchEmbedding(configs.patch_size, configs.target_channel,
                                                        configs.decoder_embed_dim, norm_layer=None)
        # 2. positional embedding
        self.positionalTarget_embedding = PositionalEncoding(configs.encoder_embed_dim, dropout=0.1)

        # ========================pretrain special======================================================
        # encoder嵌入维度映射为decoder嵌入维度
        self.encTime_2_dec_emb = nn.Linear(configs.encoder_embed_dim, configs.decoder_embed_dim, bias=True)
        self.encTarget_2_dec_emb = nn.Linear(configs.encoder_embed_dim, configs.decoder_embed_dim, bias=True)
        self.num_patches = encoder_patches
        self.seq_len = configs.seq_len
        self.pretrain_layers = configs.pretrain_layers
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, configs.encoder_embed_dim))
        self.decoder_pretrain = Encoder(configs.pretrain_layers, configs.encoder_embed_dim, configs.n_nodes, device,
                                        supports, configs.gcn_bool, configs.addaptadj)
        self.decoder_time_output = nn.Linear(configs.decoder_embed_dim, configs.patch_size * configs.time_channel)
        self.decoder_target_output = nn.Linear(configs.decoder_embed_dim, configs.patch_size * configs.target_channel)

    def random_masking(self, time, target, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [batch_size, n_nodes, n_patches, d_model], sequence
        """
        batch_size, n_nodes, n_patches, d_time = time.shape  # batch, length, dim
        _, _, _, d_target = target.shape  # batch, length, dim
        len_keep = int(n_patches * (1 - mask_ratio))

        noise = torch.rand(batch_size, n_nodes, n_patches, device=time.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        # keep the first subset
        ids_keep = ids_shuffle[:, :, :len_keep]
        masked_token_index = ids_shuffle[:, :, len_keep:]
        time_unmasked = torch.gather(time, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d_time))
        target_unmasked = torch.gather(target, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d_target))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, n_nodes, n_patches], device=time.device)
        mask[:, :, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return time_unmasked, target_unmasked, mask, ids_restore, masked_token_index

    def encoding(self, x):
        encoder_time = x[..., 1:]
        encoder_target = x[..., 0].unsqueeze(-1)

        # [batch_size, seq_len, n_nodes, n_feats] => [batch_size, n_nodes, n_feats, seq_len]
        time_features, target_features = encoder_time.permute(0, 2, 3, 1), encoder_target.permute(0, 2, 3, 1)
        # [batch_size, n_nodes, d_model, n_patches] => [batch_size, n_nodes, n_patches, d_model]
        time_features = self.enc_patchTime_embedding(time_features).transpose(2, 3)
        target_features = self.enc_patchTarget_embedding(target_features).transpose(2, 3)

        # positional embedding
        target_features = self.positionalTarget_embedding(target_features)

        time_unmasked, target_unmasked, mask_record, ids_restore, masked_token_index = self.random_masking(
            time_features, target_features, mask_ratio=0.75)
        encoder_time_input = time_unmasked
        encoder_target_input = target_unmasked

        encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, encoder_time_outputs, encoder_target_outputs = \
            self.encoder(encoder_time_input, encoder_target_input)

        return mask_record, ids_restore, masked_token_index, encoder_time_attn_scores, encoder_target_attn_scores, encoder_merge_attn_scores, encoder_time_outputs, encoder_target_outputs

    def decoding_pretrain(self, unmasked_time, unmasked_target, ids_restore, masked_token_index):
        encoder_target_output = unmasked_target[0]
        encoder_time_output = unmasked_time[0]
        for i in range(1, self.layers):
            encoder_target_output = encoder_target_output + unmasked_target[i]
            encoder_time_output = encoder_time_output + unmasked_time[i]

        batch_size, num_nodes, _, _ = encoder_target_output.shape

        # encoder 2 decoder layer
        encoder_time_output = self.encTime_2_dec_emb(encoder_time_output)
        encoder_target_output = self.encTarget_2_dec_emb(encoder_target_output)

        # add mask tokens
        target_masked = self.positionalTarget_embedding(
            self.mask_token.expand(batch_size, num_nodes, masked_token_index.shape[-1],
                                   encoder_target_output.shape[-1]),
            index=masked_token_index
        )
        time_masked = self.mask_token.expand(batch_size, num_nodes, masked_token_index.shape[-1],
                                             encoder_time_output.shape[-1])

        target_full = torch.cat([encoder_target_output, target_masked], dim=2)
        time_full = torch.cat([encoder_time_output, time_masked], dim=2)

        target_full = torch.gather(target_full, dim=2,
                                   index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, target_full.shape[-1]))
        time_full = torch.gather(time_full, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, time_full.shape[-1]))

        _, _, _, time_outputs, target_outputs = self.decoder_pretrain(time_full, target_full)

        decoder_time_output = time_outputs[0]
        decoder_target_output = target_outputs[0]
        for i in range(1, self.pretrain_layers):
            decoder_time_output = decoder_time_output + time_outputs[i]
            decoder_target_output = decoder_target_output + target_outputs[i]

        # [batch_size, n_nodes, num_patches, d_time] => [batch_size, n_nodes, num_patches, patch_size * n_feats]
        time_outputs = self.decoder_time_output(decoder_time_output)
        target_outputs = self.decoder_target_output(decoder_target_output)

        return time_outputs, target_outputs

    def forward(self, x):
        """
        x:[batch_size, seq_len, n_nodes, 5]
        """
        # 2. encoder
        mask_record, ids_restore, masked_token_index, encoder_time_attn_scores, encoder_target_attn_scores, \
        encoder_merge_attn_scores, encoder_time_outputs, encoder_target_outputs = self.encoding(x)
        # 3. decoder
        time_outputs, target_outputs = self.decoding_pretrain(encoder_time_outputs, encoder_target_outputs, ids_restore,
                                                              masked_token_index)
        return time_outputs, target_outputs, mask_record
