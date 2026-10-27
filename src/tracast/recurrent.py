"""LSTM encoder with the same direct forecasting head as the GRU baseline."""

from torch import nn


class LSTM(nn.Module):
    def __init__(self, sensors, hidden=32, horizon=12):
        super().__init__()
        self.encoder = nn.LSTM(sensors, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * sensors)
        self.sensors, self.horizon = sensors, horizon

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1]).reshape(-1, self.horizon, self.sensors)
