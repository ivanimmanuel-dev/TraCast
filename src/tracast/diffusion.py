"""A DCRNN-style diffusion-convolutional GRU (Li, Yu, Shahabi and Liu, ICLR 2018).

Traffic flows along roads, so the graph is directed and a sensor is influenced by
sensors several hops away. Diffusion convolution captures both: it mixes a sensor's
features with the features reached by random walks that follow the road direction
(forward, D_out^-1 W) and by random walks against it (reverse, D_in^-1 W^T), up to a
chosen number of hops. The DCGRU cell uses that convolution inside the GRU gates
instead of an ordinary linear layer, so the recurrent state itself is diffused.

What is different from the paper's full model: this one encodes the input window and
predicts all horizons at once from the last hidden state (the same direct heads the
shared models use, so comparisons are like for like), instead of the autoregressive
decoder with scheduled sampling. The learned weights depend on the number of features
and hops, never on the number of sensors, so one model can be trained on any graph
and clients with different local graphs can share parameters.

Written from the paper's equations; the diffusion hops follow the paper's definition
(hop 0 is the sensor's own features, and `diffusion_steps` hops are taken in each
direction). The adjacency may be any nonnegative matrix. Sensors with no outgoing (or
no incoming) edges get a zero row in that direction's transition matrix, so they
simply receive nothing through it.
"""

import re

import numpy as np
import torch
from torch import nn

MODES = ("both", "forward", "reverse", "symmetric")


def random_walk(weights):
    """Row-normalized weights, D^-1 W. A row with no weight stays all zeros."""
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Adjacency must be square")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Adjacency must be finite and nonnegative")
    degree = weights.sum(dim=1, keepdim=True)
    safe = torch.where(degree > 0, degree, torch.ones_like(degree))
    return torch.where(degree > 0, weights / safe, torch.zeros_like(weights))


def transition_matrices(weights, mode="both"):
    """The random-walk matrices the convolution diffuses along.

    both: forward D_out^-1 W and reverse D_in^-1 W^T (the paper's choice). forward or
    reverse: just one of them. symmetric: one walk on the undirected graph
    max(W, W^T), which ignores the road direction.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if mode == "symmetric":
        return [random_walk(torch.maximum(weights, weights.T))]
    forward, reverse = random_walk(weights), random_walk(weights.T)
    return {"both": [forward, reverse], "forward": [forward], "reverse": [reverse]}[
        mode
    ]


def hop_stack(weights, steps, mode="both"):
    """All the powers P^1 .. P^steps of every transition matrix, as one tensor.

    Shape (hops, N, N) with hops = steps * number of matrices. Hop 0 (the identity)
    is not stored; the convolution adds the input itself.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    matrices = transition_matrices(weights, mode)
    size = matrices[0].shape[0]
    if steps == 0:
        return torch.zeros(0, size, size)
    powers = []
    for matrix in matrices:
        current = matrix
        powers.append(current)
        for _ in range(steps - 1):
            current = current @ matrix
            powers.append(current)
    return torch.stack(powers)


class DiffusionConv(nn.Module):
    """sum_k (theta_k,1 P_f^k + theta_k,2 P_b^k) X, written as one linear layer.

    The input is concatenated with its diffused copies along the feature axis, so the
    layer has (1 + hops) * in_features inputs, whatever the number of sensors is.
    """

    def __init__(self, in_features, out_features, hops, bias_start=0.0):
        super().__init__()
        self.hops = hops
        self.linear = nn.Linear((1 + hops) * in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_start)

    def forward(self, x, powers):
        # x: (batch, sensors, features); powers: (hops, sensors, sensors)
        if self.hops == 0:
            return self.linear(x)
        reached = torch.matmul(powers[:, None], x[None])  # (hops, batch, sensors, f)
        blocks = [x] + list(reached.unbind(0))
        return self.linear(torch.cat(blocks, dim=-1))


class DCGRUCell(nn.Module):
    """A GRU cell whose gates are diffusion convolutions of [input, state]."""

    def __init__(self, in_features, hidden, hops):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(in_features + hidden, 2 * hidden, hops, 1.0)
        self.candidate = DiffusionConv(in_features + hidden, hidden, hops, 0.0)

    def forward(self, x, state, powers):
        both = torch.sigmoid(self.gates(torch.cat([x, state], dim=-1), powers))
        reset, update = both.chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * state], dim=-1), powers)
        )
        return update * state + (1 - update) * candidate


class DCRNN(nn.Module):
    """Encoder of DCGRU layers plus a per-sensor head for every forecast horizon.

    Input x is (batch, time, sensors + extra): the speeds, then any calendar
    channels (shared by all sensors). Output is (batch, horizon, sensors).
    """

    def __init__(
        self,
        weights,
        hidden=32,
        horizon=12,
        extra=0,
        diffusion_steps=2,
        mode="both",
        layers=1,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        weights = torch.as_tensor(weights, dtype=torch.float32)
        powers = hop_stack(weights, diffusion_steps, mode)
        self.register_buffer("powers", powers)
        self.sensors, self.extra, self.horizon = weights.shape[0], extra, horizon
        self.hidden, self.diffusion_steps, self.mode = hidden, diffusion_steps, mode
        hops = powers.shape[0]
        self.cells = nn.ModuleList(
            DCGRUCell((1 + extra) if i == 0 else hidden, hidden, hops)
            for i in range(layers)
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        batch, steps = x.shape[0], x.shape[1]
        speeds = x[..., : self.sensors]
        calendar = x[..., self.sensors : self.sensors + self.extra]
        states = [x.new_zeros(batch, self.sensors, self.hidden) for _ in self.cells]
        for t in range(steps):
            features = [speeds[:, t].unsqueeze(-1)]
            if self.extra:
                features.append(
                    calendar[:, t].unsqueeze(1).expand(-1, self.sensors, -1)
                )
            current = torch.cat(features, dim=-1)
            for i, cell in enumerate(self.cells):
                states[i] = cell(current, states[i], self.powers)
                current = states[i]
        forecast = self.head(states[-1])  # (batch, sensors, horizon)
        return forecast.permute(0, 2, 1)


def parse_name(base):
    """The options in a model name like dcrnn_k1_fwd_h16, or None if it isn't one.

    k<n>: diffusion hops per direction (default 2). fwd, bwd, sym: which transition
    matrices (default both directions). perm: relabel the sensors in the graph at
    random, a control that keeps the graph's shape but breaks which sensor is
    connected to which. h<n>: hidden units (default: the run's). l<n>: layers.
    """
    parts = base.split("_")
    if parts[0] != "dcrnn":
        return None
    options = {
        "steps": 2,
        "mode": "both",
        "permute": False,
        "hidden": None,
        "layers": 1,
    }
    modes = {"fwd": "forward", "bwd": "reverse", "sym": "symmetric"}
    for token in parts[1:]:
        if re.fullmatch(r"k\d", token):
            options["steps"] = int(token[1:])
        elif token in modes:
            options["mode"] = modes[token]
        elif token == "perm":
            options["permute"] = True
        elif re.fullmatch(r"h\d+", token) and int(token[1:]) > 0:
            options["hidden"] = int(token[1:])
        elif re.fullmatch(r"l[1-4]", token):
            options["layers"] = int(token[1:])
        else:
            return None
    return options


def permuted(weights, seed=0):
    """The same graph with its sensors relabelled by a seeded random permutation."""
    weights = np.asarray(weights)
    order = np.random.default_rng(seed).permutation(weights.shape[0])
    return weights[np.ix_(order, order)].copy()
