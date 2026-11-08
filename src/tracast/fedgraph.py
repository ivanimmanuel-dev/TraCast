"""Federated training of the shared graph forecasters on METR-LA.

Every client owns a disjoint set of sensors and sees only their readings and the
road graph among them (the edges that leave its sensor set are dropped, so no
client reads another client's sensors). The models exchange only
their learned parameters, whose shapes don't depend on how many sensors a client has,
so clients with different numbers of sensors can still be averaged.

This module holds everything that doesn't need Flower: the partitions, the local
graphs, one client's local update, evaluation, a plain in-process FedAvg/FedProx
loop that uses Flower's own aggregation function, and the pooled (centralized) and
local-only baselines. The Flower apps in fedgraph_flower.py call the same functions,
so both paths can be compared number for number.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler

from . import topology
from .baselines import HistoricalAverage
from .data import Scaler, Windows, load, missing_report, split
from .experiment import seed_all
from .model import DailyMetrics, Metrics

MODELS = ("sgru", "sgraph", "dcrnn", "sgru_id")
# a learned vector per sensor makes the parameters depend on the sensor count, so
# those models can be trained pooled but never averaged across clients
POOLED_ONLY = ("sgru_id",)
PARTITIONS = ("geographic", "balanced", "congestion", "random")
ALGORITHMS = ("fedavg", "fedprox")
ARMS = ("central_full", "central_local", "federated", "local_only")


@dataclass
class GraphFedConfig:
    sensors: int = 207
    rows: int = 34272
    history: int = 12
    horizon: int = 12
    hidden: int = 32
    embedding: int = 4
    model: str = "sgraph"
    diffusion_steps: int = 2
    diffusion_mode: str = "both"
    clients: int = 8
    partition: str = "geographic"
    partition_seed: int = 0
    rounds: int = 30
    local_epochs: int = 1
    local_stride: int = 3
    batch_size: int = 64
    lr: float = 0.001
    algorithm: str = "fedavg"
    proximal_mu: float = 0.0
    seed: int = 42
    threads: int = 4
    threshold: float = 0.1
    congestion_mph: float = 40.0
    budgets: list = field(default_factory=list)
    personalize_rounds: int = 0

    def validate(self):
        for name in (
            "sensors",
            "rows",
            "history",
            "horizon",
            "hidden",
            "clients",
            "rounds",
            "local_epochs",
            "local_stride",
            "batch_size",
            "threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.partition_seed) is not int or self.partition_seed < 0:
            raise ValueError("partition_seed must be a nonnegative integer")
        if type(self.diffusion_steps) is not int or self.diffusion_steps < 0:
            raise ValueError("diffusion_steps must be a nonnegative integer")
        if self.diffusion_mode not in ("both", "forward", "reverse", "symmetric"):
            raise ValueError("Unknown diffusion_mode")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(self.proximal_mu) or self.proximal_mu < 0:
            raise ValueError("proximal_mu must be nonnegative and finite")
        if self.algorithm == "fedavg" and self.proximal_mu != 0:
            raise ValueError("FedAvg has no proximal term; use algorithm='fedprox'")
        if type(self.personalize_rounds) is not int or self.personalize_rounds < 0:
            raise ValueError("personalize_rounds must be a nonnegative integer")
        if any(type(b) is not int or not 1 <= b <= self.rounds for b in self.budgets):
            raise ValueError("budgets must be round counts up to rounds")
        if not 16 <= self.sensors <= 207 or not 2 <= self.clients <= self.sensors:
            raise ValueError("Need 16 to 207 sensors and at least two clients")
        if not self.history + self.horizon < self.rows <= 34272:
            raise ValueError("rows must cover history + horizon and fit METR-LA")

    @property
    def extra(self):
        return 3

    def mu(self):
        return self.proximal_mu if self.algorithm == "fedprox" else 0.0


# --- partitions -----------------------------------------------------------------------


def kmeans(points, k, seed=0, restarts=10, iterations=100):
    """Lloyd's algorithm with k-means++ starts; the best of a few restarts.

    Deterministic for a given seed, and returns one label per point. Cluster labels
    are renumbered by centroid (west to east, then south to north) so the result
    doesn't depend on the order the restarts happened to find the clusters in.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not 1 <= k <= len(points):
        raise ValueError("Invalid points or cluster count")
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        centers = [points[rng.integers(len(points))]]
        while len(centers) < k:
            distance = np.min(
                ((points[:, None] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1
            )
            if distance.sum() == 0:
                centers.append(points[rng.integers(len(points))])
                continue
            centers.append(points[rng.choice(len(points), p=distance / distance.sum())])
        centers = np.asarray(centers)
        for _ in range(iterations):
            distance = ((points[:, None] - centers[None]) ** 2).sum(-1)
            labels = distance.argmin(axis=1)
            updated = centers.copy()
            for c in range(k):
                members = points[labels == c]
                if len(members):
                    updated[c] = members.mean(axis=0)
                else:
                    # an empty cluster takes the point farthest from its center
                    updated[c] = points[distance.min(axis=1).argmax()]
            if np.allclose(updated, centers):
                break
            centers = updated
        distance = ((points[:, None] - centers[None]) ** 2).sum(-1)
        labels = distance.argmin(axis=1)
        inertia = float(distance.min(axis=1).sum())
        if inertia < best_inertia - 1e-12 and len(set(labels)) == k:
            best, best_inertia = (labels, centers), inertia
    if best is None:
        raise ValueError("k-means could not find the requested number of clusters")
    labels, centers = best
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    renumber = np.empty(k, dtype=int)
    renumber[order] = np.arange(k)
    return renumber[labels]


def congestion_scores(frame, config):
    """Share of observed training speeds below the congestion threshold, per sensor."""
    train = split(frame, config.history, config.horizon)["train"]
    observed = train.mask(train.eq(0))
    count = observed.notna().sum()
    if (count == 0).any():
        raise ValueError("A sensor has no observed training values")
    return (observed.lt(config.congestion_mph).sum() / count).to_numpy()


def partition_groups(frame, locations, config):
    """Sensor index groups, one per client, from the training period and coordinates.

    geographic: k-means on the sensor coordinates, so clients own areas of the
    network and have different numbers of sensors. balanced: sensors dealt out by
    congestion rank, so every client gets a similar mix. congestion: consecutive
    blocks of that ranking, so each client sees a different traffic regime.
    random: a seeded shuffle.
    """
    count, clients = frame.shape[1], config.clients
    if config.partition == "geographic":
        lat0 = float(np.mean([locations[str(s)]["latitude"] for s in frame.columns]))
        points = np.array(
            [
                [
                    locations[str(s)]["longitude"] * np.cos(np.radians(lat0)),
                    locations[str(s)]["latitude"],
                ]
                for s in frame.columns
            ]
        )
        labels = kmeans(points, clients, config.partition_seed)
        groups = [np.flatnonzero(labels == c) for c in range(clients)]
    elif config.partition in ("balanced", "congestion"):
        scores = congestion_scores(frame, config)
        ranked = sorted(range(count), key=lambda i: (scores[i], str(frame.columns[i])))
        if config.partition == "balanced":
            groups = [np.asarray(ranked[i::clients]) for i in range(clients)]
        else:
            groups = [np.asarray(g) for g in np.array_split(ranked, clients)]
    else:
        order = np.random.default_rng(config.partition_seed).permutation(count)
        groups = [order[i::clients] for i in range(clients)]
    groups = [np.sort(g).astype(int) for g in groups]
    covered = np.concatenate(groups)
    if len(covered) != count or len(set(covered.tolist())) != count:
        raise ValueError("Partitions must be disjoint and cover every sensor")
    if any(len(g) == 0 for g in groups):
        raise ValueError("Every client needs at least one sensor")
    return groups


def local_graphs(weights, groups):
    """Each client's permitted graph: the edges among its own sensors only."""
    return [weights[np.ix_(g, g)].copy() for g in groups]


def block_graph(weights, groups):
    """One matrix holding every client's local graph and no edge between clients."""
    block = np.zeros_like(weights)
    for g in groups:
        block[np.ix_(g, g)] = weights[np.ix_(g, g)]
    return block


def graph_report(weights, groups):
    """What the partition does to the graph: edges kept, edges cut, sensors left alone."""
    total, kept = np.count_nonzero(weights), 0
    total_weight, kept_weight = float(weights.sum()), 0.0
    clients = []
    for i, g in enumerate(groups):
        local = weights[np.ix_(g, g)]
        kept += int(np.count_nonzero(local))
        kept_weight += float(local.sum())
        isolated = int(((local.sum(0) == 0) & (local.sum(1) == 0)).sum())
        clients.append(
            {
                "client": i,
                "sensors": len(g),
                "local_edges": int(np.count_nonzero(local)),
                "locally_isolated_sensors": isolated,
            }
        )
    return {
        "directed_edges": int(total),
        "edges_kept": kept,
        "edges_kept_fraction": kept / total if total else 1.0,
        "weight_kept_fraction": kept_weight / total_weight if total_weight else 1.0,
        "clients": clients,
    }


# --- client data ----------------------------------------------------------------------


class ClientData:
    """One client's sensors, local graph, and chronological windows.

    The scaler is fit on this client's own training rows, and the graph holds only
    edges among its own sensors. Nothing here is read from another client.
    """

    def __init__(self, frame, weights, config, client_id=0, scaler=None):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (frame.shape[1], frame.shape[1]):
            raise ValueError("The graph must match the client's sensors")
        parts = split(frame, config.history, config.horizon)
        self.client_id = client_id
        self.sensors = list(map(str, frame.columns))
        self.weights = weights
        self.scaler = scaler if scaler is not None else Scaler().fit(parts["train"])
        self.datasets = {
            name: Windows(part, self.scaler, config.history, config.horizon, True)
            for name, part in parts.items()
        }
        self.parts = parts
        self.mean = torch.from_numpy(self.scaler.mean)
        self.std = torch.from_numpy(self.scaler.std)
        self.examples = len(self.datasets["train"]) * frame.shape[1]

    def info(self):
        return {
            "client": self.client_id,
            "sensors": self.sensors,
            "sensor_count": len(self.sensors),
            "local_edges": int(np.count_nonzero(self.weights)),
            "train_examples": self.examples,
            "missing": {k: missing_report(v) for k, v in self.parts.items()},
            "normalization": "per-sensor mean and std from this client's training rows",
        }


def prepare(config, data, graph_dir):
    """Load METR-LA, pick the sensors, and cut them into client groups.

    Returns a dict with the full frame and graph (used only to build the clients and
    the centralized reference) and the per-client index groups.
    """
    config.validate()
    frame = load(data, 207, config.rows)
    ids, weights, meta = topology.load(
        graph_dir, frame.columns, config.sensors, config.threshold
    )
    frame = frame.loc[:, ids]
    groups = partition_groups(frame, meta["locations"], config)
    return {
        "frame": frame,
        "ids": ids,
        "weights": weights,
        "groups": groups,
        "graph_meta": {k: v for k, v in meta.items() if k not in ("adjacency",)},
    }


def build_clients(prep, config):
    frame, weights, groups = prep["frame"], prep["weights"], prep["groups"]
    graphs = local_graphs(weights, groups)
    return [
        ClientData(frame.iloc[:, g], graphs[i], config, i) for i, g in enumerate(groups)
    ]


def pooled_client(prep, config, edges):
    """Everything in one place, for the centralized reference arms.

    edges="full" keeps the whole road graph; edges="local" keeps only the edges that
    stay inside a client, so the pooled model sees exactly the graph the federated
    clients see (this separates the effect of the cut graph from the effect of
    federated optimization).
    """
    weights = prep["weights"]
    if edges == "local":
        weights = block_graph(weights, prep["groups"])
    elif edges != "full":
        raise ValueError("edges must be full or local")
    return ClientData(prep["frame"], weights, config, client_id=0)


# --- models ---------------------------------------------------------------------------


def build_model(config, weights):
    """The forecaster for a client whose local graph is weights.

    The learned parameters never depend on the number of sensors, only the
    (non-learned) graph buffers do.
    """
    weights = np.asarray(weights, dtype=np.float32)
    sensors = weights.shape[0]
    if config.model == "sgru":
        from .shared import SharedRecurrent

        return SharedRecurrent(
            sensors, config.hidden, config.horizon, config.extra, None
        )
    if config.model == "sgru_id":
        from .shared import SharedRecurrent

        return SharedRecurrent(
            sensors,
            config.hidden,
            config.horizon,
            config.extra,
            None,
            embedding=config.embedding,
        )
    if config.model == "sgraph":
        from .shared import SharedRecurrent

        return SharedRecurrent(
            sensors, config.hidden, config.horizon, config.extra, weights
        )
    from .diffusion import DCRNN

    return DCRNN(
        weights,
        config.hidden,
        config.horizon,
        config.extra,
        diffusion_steps=config.diffusion_steps,
        mode=config.diffusion_mode,
    )


def parameter_dict(model):
    """The learned parameters only (graph buffers are local and never exchanged)."""
    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in model.named_parameters()
    )


def load_parameters(model, params):
    names = [name for name, _ in model.named_parameters()]
    if list(params) != names:
        raise ValueError("Parameter names differ from the model's")
    with torch.no_grad():
        for (name, target), source in zip(model.named_parameters(), params.values()):
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"Parameter {name} has the wrong shape")
            target.copy_(source)


def initial_parameters(config, sensors=2):
    """Same start weights for every arm: seeded, and independent of the graph.

    sensors only matters for the models with a vector per sensor.
    """
    seed_all(config.seed, config.threads)
    return parameter_dict(
        build_model(config, np.zeros((sensors, sensors), dtype=np.float32))
    )


def parameter_count(params):
    return int(sum(v.numel() for v in params.values()))


def parameter_bytes(params):
    return int(sum(v.numel() * v.element_size() for v in params.values()))


def state_hash(params):
    import hashlib

    digest = hashlib.sha256()
    for name, value in params.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# --- training and evaluation ----------------------------------------------------------


def local_update(model, start, data, config, seed, mu=0.0):
    """One client's local training from the global parameters.

    Fresh Adam, local_epochs passes that each use a random 1/local_stride of the
    client's training windows, normalized masked MAE, gradient clipping at 5. With
    mu > 0 the loss gets FedProx's proximal term (mu / 2) * ||w - w_global||^2.
    """
    load_parameters(model, start)
    seed_all(seed, config.threads)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    anchor = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    train = data.datasets["train"]
    sampler = RandomSampler(
        train,
        num_samples=max(1, len(train) // config.local_stride),
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(train, batch_size=config.batch_size, sampler=sampler)
    loss_sum, targets, steps, windows = 0.0, 0, 0, 0
    began = time.perf_counter()
    for _ in range(config.local_epochs):
        for x, y, mask in loader:
            count = int(mask.sum())
            if not count:
                continue
            optimizer.zero_grad(set_to_none=True)
            target = (y - data.mean) / data.std
            loss = (model(x)[mask] - target[mask]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite local loss")
            objective = loss
            if anchor is not None:
                prox = sum(
                    (p - a).pow(2).sum() for p, a in zip(model.parameters(), anchor)
                )
                objective = loss + 0.5 * mu * prox
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * count
            targets += count
            steps += 1
            windows += len(x)
    if not targets:
        raise ValueError("No observed training targets")
    return parameter_dict(model), {
        "num-examples": data.examples,
        "loss-sum": loss_sum,
        "target-count": targets,
        "optimizer-steps": steps,
        "processed-windows": windows,
        "train-normalized-mae": loss_sum / targets,
        "train-seconds": time.perf_counter() - began,
    }


def evaluate(model, data, stage, config, with_days=True, by_sensor=False, only=None):
    """Error sums in mph per horizon (and per forecast-origin day) on one split.

    model=None scores persistence: the last observed speed repeated. With
    by_sensor the record also holds the sums per horizon and sensor, so a pooled
    model can be scored on each client's sensors afterwards (see slice_record).
    only restricts every score to those sensor indices; the model still reads all
    sensors, so graph features keep their neighbors.
    """
    dataset = data.datasets[stage]
    metrics = Metrics(config.horizon)
    codes, labels = dataset.origin_days()
    daily = DailyMetrics(codes, labels) if with_days else None
    sensors = len(data.sensors)
    cells = (config.horizon, sensors)
    cell_abs = torch.zeros(cells, dtype=torch.float64)
    cell_sq = torch.zeros(cells, dtype=torch.float64)
    cell_count = torch.zeros(cells, dtype=torch.int64)
    if model is not None:
        model.eval()
    keep = None
    if only is not None:
        keep = torch.zeros(sensors, dtype=torch.bool)
        keep[torch.as_tensor(list(only), dtype=torch.long)] = True
    start = 0
    with torch.no_grad():
        for x, y, mask in DataLoader(dataset, batch_size=256):
            if keep is not None:
                mask = mask & keep
            if model is None:
                normalized = x[:, -1:, :sensors].expand(-1, config.horizon, -1)
            else:
                normalized = model(x)
            prediction = normalized * data.std + data.mean
            metrics.update(prediction, y, mask)
            if daily is not None:
                daily.update(prediction, y, mask, start)
            if by_sensor:
                error = torch.where(mask, prediction.double() - y.double(), 0)
                cell_abs += error.abs().sum(0)
                cell_sq += error.square().sum(0)
                cell_count += mask.sum(0)
            start += len(x)
    out = {
        "ae": metrics.absolute.tolist(),
        "se": metrics.squared.tolist(),
        "count": metrics.count.tolist(),
    }
    if daily is not None:
        out["daily"] = daily.result()
    if by_sensor:
        out["cells"] = {
            "ae": cell_abs.tolist(),
            "se": cell_sq.tolist(),
            "count": cell_count.tolist(),
        }
    return out


def slice_record(record, indices):
    """The horizon sums of a by_sensor record, restricted to some sensors."""
    cells = {k: np.asarray(v) for k, v in record["cells"].items()}
    pick = np.asarray(indices)
    return {
        "ae": cells["ae"][:, pick].sum(axis=1).tolist(),
        "se": cells["se"][:, pick].sum(axis=1).tolist(),
        "count": cells["count"][:, pick].sum(axis=1).astype(int).tolist(),
    }


def pool(records):
    """Add sufficient statistics across clients and turn them into MAE and RMSE."""
    if not records:
        raise ValueError("No evaluation records")
    horizon = len(records[0]["count"])
    metrics = Metrics(horizon)
    for record in records:
        if any(len(record[key]) != horizon for key in ("ae", "se", "count")):
            raise ValueError("Inconsistent horizon statistics")
        metrics.absolute += torch.tensor(record["ae"], dtype=torch.float64)
        metrics.squared += torch.tensor(record["se"], dtype=torch.float64)
        metrics.count += torch.tensor(record["count"], dtype=torch.int64)
    return metrics.result()


def pool_days(records):
    """Per-day error sums added over clients, for paired bootstrapping over test days."""
    days = [r["daily"] for r in records]
    first = days[0]
    if any(d["days"] != first["days"] for d in days):
        raise ValueError("Clients cover different test days")
    return {
        "days": first["days"],
        "abs": np.sum([d["abs"] for d in days], axis=0).tolist(),
        "sq": np.sum([d["sq"] for d in days], axis=0).tolist(),
        "count": np.sum([d["count"] for d in days], axis=0).astype(int).tolist(),
    }


def slim(record):
    """A record without the per-day detail, for per-client tables."""
    return {k: record[k] for k in ("ae", "se", "count")}


def client_scores(model_for_client, clients, stage, config):
    return [
        evaluate(model_for_client(c), c, stage, config, with_days=stage == "test")
        for c in clients
    ]


# --- baselines that don't train -------------------------------------------------------


def persistence_arm(clients, config):
    records = [evaluate(None, c, "test", config) for c in clients]
    return {
        "test": pool(records),
        "daily": pool_days(records),
        "clients": [
            {"client": c.client_id, "metrics": pool([slim(r)])}
            for c, r in zip(clients, records)
        ],
    }


def historical_average_arm(prep, clients, config):
    """Per sensor, weekday and 5-minute slot training mean (no communication needed)."""
    records = []
    for c in clients:
        train, test = c.parts["train"], c.parts["test"]
        values = train.to_numpy(dtype=np.float64)
        mask = np.isfinite(values) & (values != 0)
        table = HistoricalAverage().fit(train.index, values, mask)
        predicted = table.predict(test.index)
        dataset = c.datasets["test"]
        metrics = Metrics(config.horizon)
        codes, labels = dataset.origin_days()
        daily = DailyMetrics(codes, labels)
        steps = np.arange(config.horizon)
        for start in range(0, len(dataset), 2048):
            origin = np.arange(start, min(start + 2048, len(dataset)))
            rows = origin[:, None] + dataset.history + steps[None, :]
            prediction = torch.from_numpy(predicted[rows])
            target = torch.from_numpy(dataset.y[rows])
            mask_rows = torch.from_numpy(dataset.mask[rows])
            metrics.update(prediction, target, mask_rows)
            daily.update(prediction, target, mask_rows, start)
        records.append(
            {
                "ae": metrics.absolute.tolist(),
                "se": metrics.squared.tolist(),
                "count": metrics.count.tolist(),
                "daily": daily.result(),
            }
        )
    return {
        "test": pool(records),
        "daily": pool_days(records),
        "clients": [
            {"client": c.client_id, "metrics": pool([slim(r)])}
            for c, r in zip(clients, records)
        ],
    }


# --- arms -----------------------------------------------------------------------------


def _models(config, clients):
    return {c.client_id: build_model(config, c.weights) for c in clients}


def _validation_mae(models, clients, config):
    records = [evaluate(models[c.client_id], c, "val", config, False) for c in clients]
    return pool(records), records


def run_federated(config, clients, start, aggregate=None, log=print):
    """FedAvg or FedProx, in process, with the same code the Flower clients run.

    Aggregation is Flower's own aggregate_arrayrecords (weighted by the number of
    local training examples, clients in id order), so this loop and a Flower run can
    be compared bit for bit. The best round on pooled validation MAE is kept.
    """
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    if config.model in POOLED_ONLY:
        raise ValueError(
            f"{config.model} has per-sensor parameters; it can't be federated"
        )
    models = _models(config, clients)
    size = parameter_bytes(start)
    params, history, snapshots = start, [], {}
    best, best_round, best_params = np.inf, None, None
    began = time.perf_counter()
    for r in range(1, config.rounds + 1):
        replies, rows = [], []
        for c in clients:
            updated, stats = local_update(
                models[c.client_id],
                params,
                c,
                config,
                config.seed + 10000 * r + c.client_id,
                config.mu(),
            )
            replies.append(
                RecordDict(
                    {
                        "arrays": ArrayRecord(updated),
                        "metrics": MetricRecord(
                            {"num-examples": stats["num-examples"]}
                        ),
                    }
                )
            )
            rows.append({"client": c.client_id, **stats})
        merged = aggregate_arrayrecords(replies, "num-examples")
        params = OrderedDict(merged.to_torch_state_dict())
        for model in models.values():
            load_parameters(model, params)
        validation, records = _validation_mae(models, clients, config)
        train_loss = sum(x["loss-sum"] for x in rows) / sum(
            x["target-count"] for x in rows
        )
        history.append(
            {
                "round": r,
                "train_normalized_mae": train_loss,
                "validation": {k: validation[k] for k in ("mae", "rmse")},
                "client_validation_mae": [pool([slim(x)])["mae"] for x in records],
                "client_train_seconds": [x["train-seconds"] for x in rows],
                "optimizer_steps": sum(x["optimizer-steps"] for x in rows),
                "state_sha256": state_hash(params),
                "communication_bytes": {
                    "train_download": len(clients) * size,
                    "train_upload": len(clients) * size,
                    "validation_download": len(clients) * size,
                    "round_total": 3 * len(clients) * size,
                },
            }
        )
        if validation["mae"] < best:
            best, best_round = validation["mae"], r
            best_params = OrderedDict((k, v.clone()) for k, v in params.items())
        if config.budgets:
            snapshots[r] = OrderedDict((k, v.clone()) for k, v in params.items())
        log(
            f"federated round {r}: val MAE {validation['mae']:.4f} "
            f"train {train_loss:.4f}"
        )
    result = finish_federated(
        config, clients, models, history, best_round, best_params, size, began
    )
    if config.budgets:
        result["budgets"] = budget_results(
            config, clients, models, history, snapshots, size
        )
    if config.personalize_rounds:
        result["personalized"] = personalize(
            config,
            clients,
            models,
            best_params,
            config.personalize_rounds,
            log,
        )
    return result


def personalize(config, clients, models, start, rounds, log=print):
    """Local fine-tuning of the final federated model, with no further communication.

    Every client continues training the global model on its own data for up to
    `rounds` local rounds (the same local update as everywhere else, with fresh seeds)
    and keeps the round with the best validation MAE on its own data. Round 0 is the
    federated model itself, so a client only moves away from it when its own
    validation data says that is better. The test statistics are added over clients.
    """
    began = time.perf_counter()
    records, rows = [], []
    for c in clients:
        model = models[c.client_id]
        load_parameters(model, start)
        best = pool([evaluate(model, c, "val", config, False)])["mae"]
        best_round, best_params, params = 0, start, start
        train_seconds = 0.0
        for r in range(1, rounds + 1):
            params, stats = local_update(
                model,
                params,
                c,
                config,
                config.seed + 20000 + 100 * r + c.client_id,
            )
            train_seconds += stats["train-seconds"]
            load_parameters(model, params)
            validation = pool([evaluate(model, c, "val", config, False)])["mae"]
            if validation < best:
                best, best_round = validation, r
                best_params = OrderedDict((k, v.clone()) for k, v in params.items())
        load_parameters(model, best_params)
        record = evaluate(model, c, "test", config)
        records.append(record)
        rows.append(
            {
                "client": c.client_id,
                "sensors": len(c.sensors),
                "best_round": best_round,
                "validation_mae": best,
                "metrics": pool([slim(record)]),
                "train_seconds": train_seconds,
            }
        )
        log(f"personalized client {c.client_id}: best round {best_round}")
    return {
        "test": pool(records),
        "daily": pool_days(records),
        "clients": rows,
        "rounds": rounds,
        "train_seconds": sum(r["train_seconds"] for r in rows),
        "wall_seconds": time.perf_counter() - began,
        "scope": "Each client fine-tunes the final federated model on its own data; no communication",
    }


def budget_results(config, clients, models, history, snapshots, size):
    """Test scores a server would have had if it had stopped after b rounds.

    Training in round r never depends on later rounds, so the first b rounds of a
    long run are exactly a b-round run. The model kept is the best on validation
    among those b rounds, as in the main arm.
    """
    table = {}
    for b in config.budgets:
        window = history[:b]
        best_round = min(window, key=lambda h: h["validation"]["mae"])["round"]
        for model in models.values():
            load_parameters(model, snapshots[best_round])
        records = client_scores(lambda c: models[c.client_id], clients, "test", config)
        pooled = pool(records)
        table[str(b)] = {
            "best_round": best_round,
            "mae": pooled["mae"],
            "rmse": pooled["rmse"],
            "communication_bytes": (3 * b + 1) * len(clients) * size,
            "daily": pool_days(records),
        }
    return table


def finish_federated(
    config, clients, models, history, best_round, best_params, size, began
):
    for model in models.values():
        load_parameters(model, best_params)
    records = client_scores(lambda c: models[c.client_id], clients, "test", config)
    rounds = len(history)
    return {
        "best_round": best_round,
        "test": pool(records),
        "daily": pool_days(records),
        "clients": [
            {
                "client": c.client_id,
                "sensors": len(c.sensors),
                "metrics": pool([slim(r)]),
            }
            for c, r in zip(clients, records)
        ],
        "history": history,
        "parameter_count": parameter_count(best_params),
        "parameter_bytes": size,
        "state_sha256": state_hash(best_params),
        "communication": {
            "training_and_validation_bytes": rounds * 3 * len(clients) * size,
            "selected_test_download_bytes": len(clients) * size,
            "total_parameter_bytes": (rounds * 3 + 1) * len(clients) * size,
            "scope": "Uncompressed parameter tensors only; metrics, configs and transport overhead excluded",
        },
        "train_seconds": sum(sum(h["client_train_seconds"]) for h in history),
        "wall_seconds": time.perf_counter() - began,
    }


def run_pooled(config, data, start, label, groups, log=print, return_params=False):
    """One model on all the data, with the same rounds, seeds and optimizer resets.

    Each round is local_epochs passes of local_stride-thinned epochs over every
    sensor at once, so the number of training examples matches the federated arm.
    groups are the clients' sensor indices, only used to score the pooled model on
    each client's sensors.
    """
    model = build_model(config, data.weights)
    params, history = start, []
    best, best_round, best_params = np.inf, None, None
    began = time.perf_counter()
    for r in range(1, config.rounds + 1):
        params, stats = local_update(
            model, params, data, config, config.seed + 10000 * r + config.clients
        )
        load_parameters(model, params)
        validation = pool([evaluate(model, data, "val", config, False)])
        history.append(
            {
                "round": r,
                "train_normalized_mae": stats["train-normalized-mae"],
                "validation": {k: validation[k] for k in ("mae", "rmse")},
                "optimizer_steps": stats["optimizer-steps"],
                "train_seconds": stats["train-seconds"],
                "state_sha256": state_hash(params),
            }
        )
        if validation["mae"] < best:
            best, best_round = validation["mae"], r
            best_params = OrderedDict((k, v.clone()) for k, v in params.items())
        log(f"{label} round {r}: val MAE {validation['mae']:.4f}")
    load_parameters(model, best_params)
    record = evaluate(model, data, "test", config, by_sensor=True)
    result = {
        "best_round": best_round,
        "test": pool([record]),
        "daily": pool_days([record]),
        "clients": [
            {
                "client": i,
                "sensors": len(g),
                "metrics": pool([slice_record(record, g)]),
            }
            for i, g in enumerate(groups)
        ],
        "history": history,
        "parameter_count": parameter_count(best_params),
        "state_sha256": state_hash(best_params),
        "train_seconds": sum(h["train_seconds"] for h in history),
        "wall_seconds": time.perf_counter() - began,
    }
    return (result, best_params) if return_params else result


def run_local_only(config, clients, start, log=print):
    """Every client alone: the same rounds and seeds, no communication.

    Each client keeps its own best-validation round, and the test statistics are
    added across clients, so the number is comparable with the federated one.
    """
    began = time.perf_counter()
    records, rows = [], []
    for c in clients:
        model = build_model(config, c.weights)
        params, best, best_round, best_params = start, np.inf, None, None
        train_seconds = 0.0
        for r in range(1, config.rounds + 1):
            params, stats = local_update(
                model, params, c, config, config.seed + 10000 * r + c.client_id
            )
            train_seconds += stats["train-seconds"]
            load_parameters(model, params)
            validation = pool([evaluate(model, c, "val", config, False)])
            if validation["mae"] < best:
                best, best_round = validation["mae"], r
                best_params = OrderedDict((k, v.clone()) for k, v in params.items())
        load_parameters(model, best_params)
        record = evaluate(model, c, "test", config)
        records.append(record)
        rows.append(
            {
                "client": c.client_id,
                "sensors": len(c.sensors),
                "best_round": best_round,
                "validation_mae": best,
                "metrics": pool([slim(record)]),
                "train_seconds": train_seconds,
            }
        )
        log(f"local-only client {c.client_id}: val MAE {best:.4f} (round {best_round})")
    return {
        "test": pool(records),
        "daily": pool_days(records),
        "clients": rows,
        "parameter_count": parameter_count(start),
        "train_seconds": sum(r["train_seconds"] for r in rows),
        "wall_seconds": time.perf_counter() - began,
        "scope": "Per-client models chosen on each client's own validation data; no communication",
    }


def describe(prep, clients, config):
    """Everything about the setup a reader needs to judge the comparison."""
    return {
        "config": asdict(config),
        "sensors": len(prep["ids"]),
        "clients": [c.info() | {"sensors": c.sensors} for c in clients],
        "graph": graph_report(prep["weights"], prep["groups"]),
        "split": {
            name: {
                "first": str(part.index[0]),
                "last": str(part.index[-1]),
                "rows": len(part),
            }
            for name, part in clients[0].parts.items()
        },
    }


def with_config(config, **changes):
    new = replace(config, **changes)
    new.validate()
    return new


def frame_from_arrays(values, timestamps, sensors):
    return pd.DataFrame(
        values, index=pd.DatetimeIndex(timestamps), columns=list(map(str, sensors))
    )


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
