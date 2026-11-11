"""LSTM regressor: reads a window cycle by cycle, outputs one RUL number."""
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features) -> out: (batch, window, hidden)
        out, _ = self.lstm(x)
        # keep only the memory after the LAST cycle, then map it to one number
        return self.head(out[:, -1, :]).squeeze(-1)
