import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, sensors, hidden=32, horizon=12):
        super().__init__()
        self.gru = nn.GRU(sensors, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * sensors)
        self.sensors, self.horizon = sensors, horizon

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1]).reshape(-1, self.horizon, self.sensors)


class Metrics:
    def __init__(self, horizon):
        self.absolute = torch.zeros(horizon, dtype=torch.float64)
        self.squared = torch.zeros(horizon, dtype=torch.float64)
        self.count = torch.zeros(horizon, dtype=torch.int64)

    def update(self, prediction, target, mask):
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Metric shapes differ")
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        error = torch.where(mask, prediction.double() - target.double(), 0)
        self.absolute += error.abs().sum((0, 2))
        self.squared += error.square().sum((0, 2))
        self.count += mask.sum((0, 2))

    def result(self):
        if (self.count == 0).any():
            raise ValueError("No observed targets for a forecast horizon")
        return {
            "mae": (self.absolute.sum() / self.count.sum()).item(),
            "rmse": (self.squared.sum() / self.count.sum()).sqrt().item(),
            "per_horizon": [
                {
                    "minutes": 5 * (i + 1),
                    "count": int(self.count[i]),
                    "mae": (self.absolute[i] / self.count[i]).item(),
                    "rmse": (self.squared[i] / self.count[i]).sqrt().item(),
                }
                for i in range(len(self.count))
            ],
        }
