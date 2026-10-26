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
    def __init__(self, weights, hidden=32, horizon=12, embedding=4):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32).clone()
        self.register_buffer("incoming", normalized(weights.T))
        self.register_buffer("outgoing", normalized(weights))
        self.sensors = weights.shape[0]
        self.spatial = nn.Linear(3, embedding)
        self.encoder = GRU(self.sensors, hidden, horizon)
        self.encoder.gru = nn.GRU(self.sensors * embedding, hidden, batch_first=True)

    def spatial_features(self, x):
        incoming = torch.einsum("ij,btj->bti", self.incoming, x)
        outgoing = torch.einsum("ij,btj->bti", self.outgoing, x)
        return torch.stack((x, incoming, outgoing), dim=-1)

    def forward(self, x):
        features = torch.relu(self.spatial(self.spatial_features(x)))
        return self.encoder(features.flatten(start_dim=2))
