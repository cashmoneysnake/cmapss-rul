"""Small Transformer encoder: every cycle can look at every other cycle through attention."""
import math

import torch
from torch import nn


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.embed = nn.Linear(n_features, hidden)       # sensors -> hidden-size vector per cycle
        # sinusoidal position code, so the model knows the ORDER of the cycles
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        pe = torch.zeros(max_len, hidden)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x) + self.pe[: x.shape[1]]        # (batch, window, hidden)
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)        # read off the most recent cycle
