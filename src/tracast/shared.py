"""Shared-weight forecasters: one small recurrent net applied to every sensor.

The dense models (GRU, LSTM, Graph-GRU) read all sensors at once, so their input
and output layers grow with the sensor count. These models run the same recurrent
net on each sensor separately, so the parameter count doesn't depend on how many
sensors there are. Each sensor sees its own recent speeds, the calendar inputs and,
for the graph version, the weighted average speed of its upstream and downstream
neighbors. Without the graph a sensor knows nothing about its neighbors at all, so
comparing the two is a clean test of whether the graph helps.
"""

import torch
from torch import nn

from .graph_model import normalized


class SharedRecurrent(nn.Module):
    def __init__(
        self,
        sensors,
        hidden=32,
        horizon=12,
        extra=0,
        weights=None,
        kind="gru",
        embedding=0,
    ):
        super().__init__()
        if kind not in ("gru", "lstm"):
            raise ValueError("kind must be gru or lstm")
        self.sensors, self.horizon, self.extra = sensors, horizon, extra
        self.graph = weights is not None
        if self.graph:
            weights = torch.as_tensor(weights, dtype=torch.float32).clone()
            if weights.shape != (sensors, sensors):
                raise ValueError("Adjacency must match the sensor count")
            self.register_buffer("incoming", normalized(weights.T))
            self.register_buffer("outgoing", normalized(weights))
        self.node = nn.Parameter(torch.zeros(sensors, embedding)) if embedding else None
        if self.node is not None:
            nn.init.normal_(self.node, std=0.1)
        features = 1 + (2 if self.graph else 0) + extra + embedding
        cell = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cell(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        batch, steps = x.shape[0], x.shape[1]
        speeds = x[..., : self.sensors]
        parts = [speeds.unsqueeze(-1)]
        if self.graph:
            parts.append(
                torch.einsum("ij,btj->bti", self.incoming, speeds).unsqueeze(-1)
            )
            parts.append(
                torch.einsum("ij,btj->bti", self.outgoing, speeds).unsqueeze(-1)
            )
        if self.extra:
            calendar = x[..., self.sensors : self.sensors + self.extra]
            parts.append(calendar.unsqueeze(2).expand(-1, -1, self.sensors, -1))
        if self.node is not None:
            parts.append(self.node.expand(batch, steps, -1, -1))
        # (batch, time, sensor, feature) -> one sequence per sensor
        sequences = torch.cat(parts, dim=-1).permute(0, 2, 1, 3)
        sequences = sequences.reshape(batch * self.sensors, steps, -1)
        _, state = self.rnn(sequences)
        last = state[0] if isinstance(state, tuple) else state
        forecast = self.head(last[-1])
        return forecast.reshape(batch, self.sensors, self.horizon).permute(0, 2, 1)
