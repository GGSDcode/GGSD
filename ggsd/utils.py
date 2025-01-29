import numpy as np
import torch
import torch.nn as nn


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