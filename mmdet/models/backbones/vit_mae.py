from mae.models_vit import VisionTransformer
from torch import Tensor

from mmengine.model import BaseModule

from mmdet.registry import MODELS

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from mae.util.pos_embed import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_with_resolution,
)


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTMAE(BaseModule):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        self_attention=False,
        absolute_scale=False,
        target_size=[],
        init_cfg=None
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.multiscale = len(target_size) > 1
        self.patch_embed = PatchEmbedUnSafe(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.self_attention = self_attention
        self.absolute_scale = absolute_scale
        self.target_size = target_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        
        # for base module init weights
        self.init_cfg = init_cfg
       
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def upsample_decoder(self, x, target_dim):
        """
        x: (N, L, num_patches**2, decoder_embed_dim)
        padded: (N, decoder_num_patches**2, decoder_embed_dim)
        """
        p = target_dim
        x = x.unsqueeze(dim=1)
        n, _, l_low, _ = x.shape
        l_low_dim = int(l_low**0.5)
        x = torch.nn.functional.interpolate(
            input=x.reshape(n, 1, l_low_dim, l_low_dim, self.decoder_embed_dim),
            size=(p, p, self.decoder_embed_dim),
            mode="nearest",
        ).view(n, 1, p**2, self.decoder_embed_dim)
        padded = x.squeeze(dim=1)
        return padded

    def find_closest_multiple(self, target_resolution):
        n = target_resolution + self.patch_embed.patch_size[0] / 2
        n = n - (n % self.patch_embed.patch_size[0])
        return int(n)

    def plot_decoder_vector(self, x):
        B, total_patches, _ = x.shape
        num_patches_per_axis = int(total_patches**0.5)
        patch_size = self.patch_embed.patch_size[0]
        embed_dim = self.decoder_embed_dim

        output_raster = torch.zeros(
            B, num_patches_per_axis * embed_dim, num_patches_per_axis
        )

        data = x.reshape(
            B, num_patches_per_axis, num_patches_per_axis, embed_dim
        )  # 4, 7, 7, 512

        data = data.permute(0, 3, 1, 2)  # 4, 512, 7, 7

        for img in range(B):
            for i in range(embed_dim):
                output_raster[
                    img, i * num_patches_per_axis : (i + 1) * num_patches_per_axis, :
                ] = data[img, i, :, :]

        return output_raster

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio=0.0, input_res=None):
        # embed patches
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        input_res = input_res.cpu()

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            input_res,
            cls_token=True,
            device=x.device,
        )

        # get_2d_sincos_pos_embed(
        # pos_emb = torch.from_numpy(pos_emb).float().to(x.device) # n X L X d_emb
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x

    @staticmethod
    def subsample(seq, mask):
        if mask is None:
            return seq
        n, l = seq.shape[:2]
        _, l_mask = mask.shape
        x_arr = torch.arange(n).view(n, 1).repeat(1, l_mask)
        seq = seq[x_arr, mask]
        return seq

    def set_target_size(self, target_size):
        self.target_size = target_size
        self.multiscale = len(target_size) > 1

    def forward(
        self,
        imgs,
        input_res,
        mask_ratio=0.0,
    ):
        # images, targets: B X C X H0 X W0;  B X C X H1 X W1
        # input_res, target_res: []
        B, C, H, W = imgs.shape
        latent = self.forward_encoder(
            imgs, mask_ratio, input_res
        )

        patches_H = H // self.patch_embed.patch_size[0]
        patches_W = W // self.patch_embed.patch_size[1]
        latent = latent[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, patches_H, patches_W)
        return [latent]
        
MODELS.register_module(module=ViTMAE)
