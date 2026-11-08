"""The dense GRU and linear forecasters, and the masked metrics."""

import torch
from torch import nn


class GRU(nn.Module):
    """Direct multi-horizon forecaster from the last hidden state.

    Reads sensors + extra input channels and ignores any others, so one batch can
    feed models with and without calendar inputs.
    """

    def __init__(self, sensors, hidden=32, horizon=12, extra=0):
        super().__init__()
        self.gru = nn.GRU(sensors + extra, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * sensors)
        self.sensors, self.horizon, self.extra = sensors, horizon, extra

    def forward(self, x):
        _, state = self.gru(x[..., : self.sensors + self.extra])
        return self.head(state[-1]).reshape(-1, self.horizon, self.sensors)


class LinearAR(nn.Module):
    """One linear layer from the whole input window to every forecast."""

    def __init__(self, sensors, history=12, horizon=12, extra=0):
        super().__init__()
        self.layer = nn.Linear(history * (sensors + extra), horizon * sensors)
        self.sensors, self.horizon, self.extra = sensors, horizon, extra

    def forward(self, x):
        x = x[..., : self.sensors + self.extra]
        return self.layer(x.flatten(start_dim=1)).reshape(
            -1, self.horizon, self.sensors
        )


class Metrics:
    """Masked MAE and RMSE, overall and for each forecast horizon."""

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


class DailyMetrics:
    """Error sums per forecast-origin day.

    Feed windows in dataset order (start is the index of the batch's first window),
    which an unshuffled loader does.
    """

    def __init__(self, codes, labels):
        self.codes = torch.as_tensor(codes, dtype=torch.int64)
        self.labels = list(labels)
        days = len(self.labels)
        self.absolute = torch.zeros(days, dtype=torch.float64)
        self.squared = torch.zeros(days, dtype=torch.float64)
        self.count = torch.zeros(days, dtype=torch.int64)

    def update(self, prediction, target, mask, start):
        days = self.codes[start : start + prediction.shape[0]]
        if len(days) != prediction.shape[0]:
            raise ValueError("Daily records do not cover this batch")
        error = torch.where(mask, prediction.double() - target.double(), 0)
        self.absolute.index_add_(0, days, error.abs().sum((1, 2)))
        self.squared.index_add_(0, days, error.square().sum((1, 2)))
        self.count.index_add_(0, days, mask.sum((1, 2)))

    def result(self):
        return {
            "days": self.labels,
            "abs": self.absolute.tolist(),
            "sq": self.squared.tolist(),
            "count": self.count.tolist(),
        }
