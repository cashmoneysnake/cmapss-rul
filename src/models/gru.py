"""GRU regressor: like the LSTM but with a simpler memory cell (fewer parameters)."""
import torch
from torch import nn


class GRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)                       # (batch, window, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)
