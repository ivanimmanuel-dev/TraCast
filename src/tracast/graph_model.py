"""One-hop directed neighbor aggregation followed by the existing GRU encoder."""

import torch
from torch import nn

from .model import GRU


def normalized(weights):
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Adjacency must be square")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Adjacency must be finite and nonnegative")
    if torch.diag(weights).any():
        raise ValueError("Self values use a separate feature")
    degree = weights.sum(dim=1, keepdim=True)
    return weights / degree.clamp(min=1e-12)


class GraphGRU(nn.Module):
    def __init__(self, weights, hidden=32, horizon=12, embedding=4, extra=0):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32).clone()
        self.register_buffer("incoming", normalized(weights.T))
        self.register_buffer("outgoing", normalized(weights))
        self.sensors, self.extra = weights.shape[0], extra
        self.spatial = nn.Linear(3, embedding)
        self.encoder = GRU(self.sensors, hidden, horizon)
        # Node embeddings, then any calendar channels, feed the recurrent layer.
        self.encoder.gru = nn.GRU(
            self.sensors * embedding + extra, hidden, batch_first=True
        )

    def spatial_features(self, x):
        incoming = torch.einsum("ij,btj->bti", self.incoming, x)
        outgoing = torch.einsum("ij,btj->bti", self.outgoing, x)
        return torch.stack((x, incoming, outgoing), dim=-1)

    def forward(self, x):
        speeds = x[..., : self.sensors]
        features = torch.relu(self.spatial(self.spatial_features(speeds)))
        calendar = x[..., self.sensors : self.sensors + self.extra]
        inputs = torch.cat([features.flatten(start_dim=2), calendar], dim=-1)
        # encoder.forward only reads sensor channels, so call its gru and head
        _, state = self.encoder.gru(inputs)
        return self.encoder.head(state[-1]).reshape(
            -1, self.encoder.horizon, self.sensors
        )
