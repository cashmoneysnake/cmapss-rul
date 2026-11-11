"""1D-CNN regressor: slides small filters along the time axis instead of keeping a memory."""
import torch
from torch import nn


class CNNRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),               # average each filter over the 30 cycles
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d wants (batch, channels, time), so swap sensors and time
        z = self.conv(x.transpose(1, 2)).squeeze(-1)   # (batch, hidden)
        return self.head(z).squeeze(-1)
