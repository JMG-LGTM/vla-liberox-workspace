# SPDX-License-Identifier: Apache-2.0
# Adapted from the Q-Planning authors' LeRobot QFunction, commit 2e3a628.
"""Distributional Q over action chunks and frozen visual patch grids."""
import torch
from torch import nn
from torch.nn import functional as F

from .data import GAMMA
from .config import DEFAULT_CONFIG

def bins(device):
    return torch.linspace(-.01, 1.01, 101, device=device, dtype=torch.float32)


def hl_gauss(values):
    centers = bins(values.device)
    delta = (centers - values.float().clamp(-.01, 1.01).unsqueeze(-1)) / .0075
    return (-.5 * delta.square()).softmax(-1)


def expectation(logits):
    return (logits.float().softmax(-1) * bins(logits.device)).sum(-1)


def td_target(reward, bootstrap, q_next):
    return reward + GAMMA * torch.where(bootstrap, q_next, 0.)


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=.1, batch_first=True)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(dim) for _ in range(3)]
        self.dropout = nn.Dropout(.1)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(.1), nn.Linear(4 * dim, dim))

    def forward(self, x, position, context, pad):
        v = self.norm1(x)
        qk = v + position
        x = x + self.dropout(self.self_attn(qk, qk, v, key_padding_mask=pad, need_weights=False)[0])
        query = self.norm2(x) + position
        x = x + self.dropout(self.cross_attn(query, context, context, need_weights=False)[0])
        return x + self.dropout(self.ff(self.norm3(x)))


class Critic(nn.Module):
    def __init__(self, layers=18, dim=1024, heads=16, patches=256, config=DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.action = nn.Linear(config.action_dim, dim)
        self.query_pos = nn.Parameter(torch.empty(config.horizon, dim))
        self.cls = nn.Parameter(torch.empty(1, 1, dim))
        self.patch_pos = nn.Parameter(torch.empty(len(config.camera_keys), patches, dim))
        self.view = nn.Embedding(len(config.camera_keys), dim)
        for p in (self.query_pos, self.cls, self.patch_pos):
            nn.init.normal_(p, std=.02)
        self.layers = nn.ModuleList([DecoderLayer(dim, heads) for _ in range(layers)])
        for p in self.layers.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p)
        self.head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Dropout(.1),nn.Linear(dim,101))

    def forward(self, features, actions, mask):
        if features.dtype != torch.float32:
            raise TypeError(f"Q features must enter as float32, received {features.dtype}")
        if features.ndim != 4 or features.shape[1:] != self.patch_pos.shape:
            raise ValueError('visual features do not match the configured views and patch grid')
        if (actions.shape != (len(features),self.config.horizon,self.config.action_dim)
                or mask.shape != actions.shape[:2] or mask.dtype != torch.bool):
            raise ValueError('action batch or padding mask does not match the Q configuration')
        tokens = self.action(actions)
        context = (features + (self.patch_pos + self.view.weight[:, None]).to(features.dtype)).flatten(1, 2)
        x = torch.cat((self.cls.expand(len(tokens), -1, -1).to(tokens.dtype), tokens), dim=1)
        pos = F.pad(self.query_pos, (0,0,1,0)).to(tokens.dtype)
        pad = ~F.pad(mask, (1,0), value=True)
        for layer in self.layers:
            x = layer(x, pos, context, pad)
        return self.head(x[:, 0])
