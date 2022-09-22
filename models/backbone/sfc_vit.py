""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@PointNeXt team
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from tokenize import group

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..helpers import build_model_with_cfg, named_apply 
from ..layers import GroupEmbed,  SFCEmbed, Mlp, DropPath, trunc_normal_
from ..registry import register_model
from ..backbone2d.vit import _load_weights, _init_vit_weights, Block
from ..layers.helpers import MultipleSequential
_logger = logging.getLogger(__name__)


class Block3D(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, drop=0, attn_drop=0, drop_path=0, act_layer=..., norm_layer=...):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, features, pos):
        # TODO: make this more point cloud specific? now, only features + positional encoding. 
        # can it also taken position as input?
        
        # TODO: try to remove this +pos
        # features = super().forward(features)
        features = super().forward(features)
        # features = super().forward(features+pos)
        return features, pos


# TODO: separate to backbone model + classification head (here, only a linear layer). 
# TODO: embed_dim too large, drops performance. 
# TODO: how to resolve this? 
# TODO: transformer with small data.  
class SFCViT3D(nn.Module):
    """ SFC Vision Transformer for 3D Classification. 
    """
    def __init__(self,
                 num_groups=256, group_size=32, in_chans=6, num_classes=40, 
                 global_pool=True,  
                 embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=SFCEmbed, norm_layer=None,
                 act_layer=None, weight_init='', 
                 anisotropic=True, 
                 subsample='fps', # random, FPS
                 group='ballquery',
                 radius = 0.1 
                 ):
        """
        Args:
            num_group (int, tuple): number of patches (groups in 3d)
            group_size (int, tuple): the size (# points) of each group
            in_chans (int): number of input channels. Default: 6. (xyz + rgb)
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # TODO: how to only parse the required kwargs.
        self.group_embed = embed_layer(num_groups, in_chans=in_chans, embed_dim=embed_dim, 
                                       anisotropic=anisotropic
                                       )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Sequential( # TODO: try different 
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )  

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = MultipleSequential(*[
            Block3D(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # global pooling layer
        self.global_pool = global_pool

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    # def forward_features(self, xyz, features):
    #     center_xyz, features, grouped_xyz, grouped_features = self.group_embed(xyz, features)
    #     cls_token = self.cls_token.expand(features.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     if self.dist_token is None:
    #         features = torch.cat((cls_token, features), dim=1)
    #     else:
    #         features = torch.cat((cls_token, self.dist_token.expand(features.shape[0], -1, -1), features), dim=1)
    #     pos_embed = torch.cat((self.cls_pos.expand(features.shape[0], -1, -1), self.pos_embed(center_xyz)), dim=1)
        
    #     features = self.pos_drop(features + pos_embed)  # TODO: EVEN HERE, without self-attention, is already a strong baseline.

    #     features, pos_embed = self.blocks(features, pos_embed)
    #     features, _ = torch.max(features[:, 1:, :], dim=1)
        
    #     return features
    def forward_features(self, xyz, features):
        center_xyz, features, grouped_xyz, grouped_features = self.group_embed(xyz, features)
        cls_token = self.cls_token.expand(features.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            features = torch.cat((cls_token, features), dim=1)
        else:
            features = torch.cat((cls_token, self.dist_token.expand(features.shape[0], -1, -1), features), dim=1)
        pos_embed = torch.cat((self.cls_pos.expand(features.shape[0], -1, -1), self.pos_embed(center_xyz)), dim=1)
        
        features = self.pos_drop(features + pos_embed)

        # # TODO: test here. 
        features, pos_embed = self.blocks(features, pos_embed)
        # TODO:
        # return torch.max(features, dim=1)[0]
        if self.dist_token is None:
            if self.global_pool:
                outcome, _= torch.max(features[:, 1:, :], dim=1)  # global pool without cls token
                # outcome = self.norm(features) # TODO: this will ruin the whole network, why? NAN loss. 
            else:
                features = self.norm(features)
                outcome = features[:, 0]        
            return self.pre_logits(outcome)
        else:
            return features[:, 0], features[:, 1]

    def forward(self, xyz, features):
        x = self.forward_features(xyz, features)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # TODO: use DeepGCN's classification layer. 
            x = self.head(x)
        return x


# TODO: register model