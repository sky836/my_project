import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

from models.st_transformer import MergeAttentionLayer
from models.swin_transformer import PatchEmbed


class Model(nn.Module):
    def __init__(self, configs, mode='no pre-train'):
        super(Model, self).__init__()
        self.num_nodes = configs.num_nodes
        self.in_steps = configs.label_len
        self.steps_per_day = configs.steps_per_day
        self.input_dim = configs.input_dim
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
        self.time_dim = (configs.tod_embedding_dim * 2 + configs.dow_embedding_dim)
        self.target_dim = (configs.input_embedding_dim + configs.spatial_embedding_dim * 2)
        self.num_heads = configs.n_heads
        self.num_layers = configs.num_layers
        self.dec_layers = configs.d_layers
        self.use_mixed_proj = configs.use_mixed_proj
        self.patch_size = configs.patch_size
        self.num_nodes = configs.num_nodes
        self.dropout = configs.dropout
        self.mode = mode

        self.patch_emb = PatchEmbed(seq_len=self.in_steps, patch_size=self.patch_size,
                                    in_chans=self.input_dim, embed_dim=self.input_embedding_dim,
                                    norm_layer=nn.LayerNorm)
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
        self.time_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.num_patches, self.tod_embedding_dim))
        )
        self.series_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.num_patches, self.spatial_embedding_dim))
        )

        # ===================================encoding special=============================================
        self.merge_attn_layers = nn.ModuleList(
            [
                MergeAttentionLayer(self.time_dim, self.target_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        # ===================================pretrain special=============================================
        self.mask_ratio = configs.mask_ratio
        self.mask_token_T = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, 1, self.time_dim - self.tod_embedding_dim)))
        self.mask_token_S = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(1, 1, 1, self.target_dim - self.spatial_embedding_dim)))
        self.decoder = MergeAttentionLayer(self.time_dim, self.target_dim, self.feed_forward_dim, self.num_heads, self.dropout)
        self.pretrain_output_T = nn.Linear(self.time_dim, self.patch_size * (self.input_dim - 1))
        self.pretrain_output_S = nn.Linear(self.target_dim, self.patch_size)

    def maskgenerator(self, device, mask_ratio, l):
        len_keep = int(l * (1 - mask_ratio))
        noise = torch.rand(l, device=device)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle)
        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        masked_token_index = ids_shuffle[len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([l], device=device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        return mask, ids_restore, masked_token_index, ids_keep

    def random_masking(self, time, target, mask_ratio):
        """
        target: [batch_size, n_patches, n_nodes, d_model], sequence
        time: [batch_size, n_patches, d_model], sequence
        """
        batch_size, n_patches, n_nodes, target_dim = target.shape
        _, _, time_dim = time.shape
        device = time.device

        mask, ids_restore, masked_token_index, ids_keep = self.maskgenerator(device, mask_ratio, n_patches)

        time_unmasked = time[:, ids_keep]
        target_unmasked = target[:, ids_keep]

        return mask, ids_restore, masked_token_index, time_unmasked, target_unmasked

    def encoding(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size, in_steps, num_nodes, _ = x.shape

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
            tod = tod[..., 0]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
            dow = dow[..., 0]
        x = x[..., : self.input_dim]

        x = self.patch_emb(x)
        patch_size = self.patch_emb.patch_size
        target_features = [x]
        time_features = []
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, tod_embedding_dim)
            time_features.append(tod_emb[:, ::patch_size])
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, dow_embedding_dim)
            time_features.append(dow_emb[:, ::patch_size])
        time_features.append(self.time_embedding.expand(
            size=(batch_size, *self.time_embedding.shape)
        ))
        node_emb = self.node_emb.expand(
            size=(batch_size, self.num_patches, *self.node_emb.shape)
        )
        target_features.append(node_emb)
        target_features = torch.cat(target_features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        time_features = torch.cat(time_features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        series_emb = self.series_embedding.expand(
            size=(batch_size, num_nodes, *self.series_embedding.shape)
        )
        target_features = target_features.transpose(1, 2)
        target_features = torch.cat([target_features, series_emb], dim=-1)
        target_features = target_features.transpose(1, 2)

        mask, ids_restore, masked_token_index, time_unmasked, target_unmasked = \
            self.random_masking(time_features, target_features, self.mask_ratio)

        for i in range(self.num_layers):
            time_unmasked, target_unmasked = self.merge_attn_layers[i](time_unmasked, target_unmasked, dim=1)

        return mask, ids_restore, masked_token_index, time_unmasked, target_unmasked

    def decoding(self, ids_restore, masked_token_index, time_unmasked, target_unmasked):
        batch_size, _, num_nodes, _ = target_unmasked.shape
        time_masked = self.mask_token_T.repeat(batch_size, self.num_patches-time_unmasked.shape[1], 1)
        target_masked = self.mask_token_S.repeat(batch_size, self.num_patches-target_unmasked.shape[1], self.num_nodes, 1)

        time_feat = self.time_embedding.unsqueeze(0).expand(batch_size, *self.time_embedding.shape)
        series_feat = self.series_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_nodes, *self.series_embedding.shape)

        time_feat = time_feat[:, masked_token_index]
        series_feat = series_feat[:, :, masked_token_index]

        time_masked = torch.cat([time_masked,  time_feat], dim=-1)
        target_masked = torch.cat([target_masked.transpose(1, 2), series_feat], dim=-1).transpose(1, 2)

        time_full = torch.cat([time_unmasked, time_masked], dim=1)
        target_full = torch.cat([target_unmasked, target_masked], dim=1)

        ids_restore_T = ids_restore.unsqueeze(0).unsqueeze(-1)
        ids_restore_S = ids_restore.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ids_restore_T = ids_restore_T.expand(batch_size, self.num_patches, self.time_dim)
        ids_restore_S = ids_restore_S.expand(batch_size, self.num_patches, self.num_nodes, self.target_dim)

        time_full = torch.gather(time_full, dim=1, index=ids_restore_T)
        target_full = torch.gather(target_full, dim=1, index=ids_restore_S)

        time_full, target_full = self.decoder(time_full, target_full, dim=1)

        if self.mode == "pre-train":
            time_full = self.pretrain_output_T(time_full).view(batch_size, self.num_patches, self.patch_size * (self.input_dim - 1))
            target_full = self.pretrain_output_S(target_full).view(batch_size, self.num_patches, self.num_nodes, self.patch_size)

        return time_full, target_full

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        mask, ids_restore, masked_token_index, time_unmasked, target_unmasked = self.encoding(x)

        time_full, target_full = self.decoding(ids_restore, masked_token_index, time_unmasked, target_unmasked)
        return time_full, target_full, mask

