from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from utils.diffusion import EDMLoss

import copy

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, channels_num, max_positions=10000, endpoint=False):
        super().__init__()
        self.channels_num = channels_num
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.channels_num//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.channels_num // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class GGSD_Diffusion(nn.Module):
    def __init__(self, input_dim, layer_dim, heads):
        super().__init__()
        self.layer_dim = layer_dim

        self.init_proj = nn.Linear(input_dim, layer_dim)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_dim * 2),
            nn.BatchNorm1d(layer_dim * 2),
            nn.BatchNorm1d(layer_dim)
        ])

        self.map_noise = PositionalEmbedding(num_channels=layer_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(layer_dim, layer_dim),
            nn.SiLU(),
            nn.Linear(layer_dim, layer_dim)
        )

        self.gat_layers = nn.ModuleList([
            self.create_gat_layer(layer_dim, layer_dim * 2, heads=heads),
            self.create_gat_layer(layer_dim * 2, layer_dim * 2, heads=heads),
            self.create_gat_layer(layer_dim * 2, layer_dim, heads=heads)
        ])

        self.act = nn.SiLU()
        self.out_linear = nn.Linear(layer_dim, input_dim)
    
        self.adjust_linears = nn.ModuleList([
            nn.Linear(layer_dim, layer_dim * 2),
            nn.Linear(layer_dim * 2, layer_dim * 2),
            nn.Linear(layer_dim * 2, layer_dim)
        ])


    def create_gat_layer(self, in_dim, out_dim, heads):
        return pyg_nn.GATConv(in_dim, out_dim // heads, heads=heads, concat=True)


    def forward(self, subgraph_x, edge_index, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        subgraph_x = self.init_proj(subgraph_x) + emb

        for i, gat in enumerate(self.gat_layers):
            subgraph_initial_x = subgraph_x.clone()  
            subgraph_x = gat(subgraph_x, edge_index)  
            subgraph_initial_x = self.adjust_linears[i](subgraph_initial_x)
            subgraph_x += subgraph_initial_x  
            subgraph_x = self.batch_norms[i](subgraph_x)  
            subgraph_x = self.act(subgraph_x)  

        subgraph_x = self.out_linear(subgraph_x)
        return subgraph_x


class Precond(nn.Module):
    def __init__(self,
        gnn_denoiser,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.gnn_denoiser_F = gnn_denoiser

    def forward(self, subgraph_x, edge_index, sigma):

        subgraph_x = subgraph_x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        subgraph_x_in = c_in * subgraph_x
        # print('edge_index: ', edge_index.dtype)  torch.int64
        subgraph_F_x = self.denoise_fn_F((subgraph_x_in).to(dtype), edge_index, c_noise.flatten())

        assert subgraph_F_x.dtype == dtype
        subgraph_D_x = c_skip * subgraph_x + c_out * subgraph_F_x.to(torch.float32)
        return subgraph_D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class Model(nn.Module):
    def __init__(self, gnn_denoiser, hid_dim, P_mean, P_std, sigma_data, gamma, opts=None, pfgmpp = False):
        super().__init__()

        self.gnn_denoiser_D = Precond(gnn_denoiser, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=gamma, opts=None)

    def forward(self, subgraph_x, edge_index):
        loss = self.loss_fn(self.gnn_denoiser_D, subgraph_x, edge_index)
        return loss.mean(-1).mean()
