"""Forecasting sensors the model never trained on.

The shared models have no per-sensor parameters, so a sensor that wasn't in the
training set can be fed straight through. Giving every sensor a learned vector helps
the sensors it knows (see the variants report) but leaves a new sensor without one.
This experiment trains on some sensors, then scores the held-out ones on the same
chronological test period, with these ways of handling a new sensor:

- the plain shared GRU and Graph-GRU, zero-shot (nothing to adapt);
- the sensor-vector GRU with a zero vector, with the average trained vector, with a
  vector fitted on the new sensor's first few days (the rest of the network frozen),
  and with a vector predicted by ridge regression from what is known about the
  sensor (its speeds in those first days and its location).

The references are the same models trained on every sensor (they have seen the
held-out sensors' training rows), persistence, and the historical average. A new
sensor only ever contributes its first few days before the test period: they set its
input scaling and, for the fitted and predicted vectors, its vector. Nothing from its
validation or test rows is used except to score it (the vector fit picks its epoch on
the sensor's validation rows).
"""

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from importlib.metadata import version
from pathlib import Path
import json
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from . import fedgraph as F
from .baselines import HistoricalAverage
from .data import Scaler, Windows
from .experiment import seed_all, write_json
from .model import DailyMetrics, Metrics
from .provenance import git_state
from .resources import peak_memory
from .stats import day_bootstrap

HOLDOUTS = ("region", "random")
KINDS = ("sgru", "sgraph", "sgru_id")
ROWS_PER_DAY = 288

TARGETS = {
    "persistence": "Persistence",
    "ha_first_days": "Historical average, first days only",
    "ha_all_train": "Historical average, all training rows",
    "sgru_zero_shot": "Shared GRU, zero-shot",
    "sgraph_zero_shot": "Shared Graph-GRU, zero-shot",
    "sgru_id_zero": "Sensor-vector GRU, zero vector",
    "sgru_id_mean": "Sensor-vector GRU, mean vector",
    "sgru_id_fitted": "Sensor-vector GRU, vector fitted on first days",
    "sgru_id_ridge": "Sensor-vector GRU, vector predicted from descriptors",
    "oracle_sgru": "Shared GRU trained on all sensors",
    "oracle_sgraph": "Shared Graph-GRU trained on all sensors",
    "oracle_sgru_id": "Sensor-vector GRU trained on all sensors",
}
PAIRS = [
    ("sgraph_zero_shot", "sgru_zero_shot"),
    ("sgraph_zero_shot", "oracle_sgraph"),
    ("sgru_zero_shot", "oracle_sgru"),
    ("sgru_id_fitted", "sgru_id_zero"),
    ("sgru_id_fitted", "sgru_id_mean"),
    ("sgru_id_ridge", "sgru_id_mean"),
    ("sgru_id_fitted", "sgru_id_ridge"),
    ("sgru_id_fitted", "oracle_sgru_id"),
    ("sgru_id_fitted", "sgraph_zero_shot"),
    ("sgru_id_fitted", "oracle_sgru"),
    ("sgru_id_fitted", "oracle_sgraph"),
    ("sgraph_zero_shot", "ha_first_days"),
    ("sgru_id_fitted", "ha_first_days"),
    ("sgraph_zero_shot", "persistence"),
]


@dataclass
class InductiveConfig:
    sensors: int = 64
    rows: int = 34272
    history: int = 12
    horizon: int = 12
    hidden: int = 32
    embedding: int = 4
    holdout: str = "region"
    holdout_fraction: float = 0.25
    holdout_seed: int = 0
    rounds: int = 30
    local_stride: int = 3
    batch_size: int = 64
    lr: float = 0.001
    few_shot_days: int = 7
    fit_epochs: int = 30
    fit_lr: float = 0.02
    ridge_alpha: float = 1.0
    seeds: list = field(default_factory=lambda: [7, 42, 2026])
    threads: int = 3
    threshold: float = 0.1

    def validate(self):
        if self.holdout not in HOLDOUTS:
            raise ValueError(f"holdout must be one of {HOLDOUTS}")
        if not 0 < self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be between 0 and 0.5")
        if not 16 <= self.sensors <= 207:
            raise ValueError("Need 16 to 207 sensors")
        for name in (
            "few_shot_days",
            "fit_epochs",
            "rounds",
            "local_stride",
            "batch_size",
            "hidden",
            "embedding",
            "threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.fit_lr) or self.fit_lr <= 0 or self.ridge_alpha < 0:
            raise ValueError("fit_lr must be positive and ridge_alpha nonnegative")
        if (
            not isinstance(self.seeds, list)
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be distinct integers")
        if self.few_shot_days * ROWS_PER_DAY >= int(self.rows * 0.7) - 2 * 24:
            raise ValueError("few_shot_days must fit well inside the training period")

    def fed(self, model, seed):
        """The pooled-training configuration for one model kind."""
        return F.GraphFedConfig(
            sensors=self.sensors,
            rows=self.rows,
            history=self.history,
            horizon=self.horizon,
            hidden=self.hidden,
            embedding=self.embedding,
            model=model,
            clients=2,
            rounds=self.rounds,
            local_epochs=1,
            local_stride=self.local_stride,
            batch_size=self.batch_size,
            lr=self.lr,
            seed=seed,
            threads=self.threads,
            threshold=self.threshold,
        )


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {f.name for f in fields(InductiveConfig)}
    if unknown:
        raise ValueError(f"Unknown keys: {sorted(unknown)}")
    config = InductiveConfig(**raw)
    config.validate()
    return config


# --- which sensors are new ------------------------------------------------------------


def coordinates(ids, locations):
    lat0 = float(np.mean([locations[str(s)]["latitude"] for s in ids]))
    return np.array(
        [
            [
                locations[str(s)]["longitude"] * np.cos(np.radians(lat0)),
                locations[str(s)]["latitude"],
            ]
            for s in ids
        ]
    )


def choose_holdout(ids, locations, config):
    """Indices of the held-out sensors (sorted), chosen without looking at any speed.

    region: k-means into 1 / fraction areas and hold out the area closest to the
    wanted size, so the new sensors form a part of the network the model never saw.
    random: a seeded random sample, so every new sensor has trained neighbors.
    """
    count = len(ids)
    wanted = max(1, round(config.holdout_fraction * count))
    if config.holdout == "random":
        rng = np.random.default_rng(config.holdout_seed)
        return np.sort(rng.choice(count, size=wanted, replace=False))
    areas = max(2, round(1 / config.holdout_fraction))
    labels = F.kmeans(coordinates(ids, locations), areas, config.holdout_seed)
    sizes = [int((labels == a).sum()) for a in range(areas)]
    best = min(range(areas), key=lambda a: (abs(sizes[a] - wanted), a))
    return np.flatnonzero(labels == best)


def mixed_scaler(frame, trained, held, days, split_row):
    """Input scaling: trained sensors from all training rows, new ones from their
    first days only (all a newly installed sensor has before it goes live)."""
    train = frame.iloc[:split_row]
    known = Scaler().fit(train)
    new = Scaler().fit(train.iloc[: days * ROWS_PER_DAY, list(held)])
    scaler = Scaler()
    scaler.mean = known.mean.copy()
    scaler.std = known.std.copy()
    scaler.mean[held], scaler.std[held] = new.mean, new.std
    return scaler


# --- vectors for new sensors ----------------------------------------------------------


def descriptors(frame, locations, days):
    """Per sensor: mean, SD and congestion share of its first days, then latitude and
    longitude. Zeros count as missing, as everywhere else."""
    early = frame.iloc[: days * ROWS_PER_DAY]
    observed = early.mask(early.eq(0))
    rows = np.column_stack(
        [
            observed.mean().to_numpy(),
            observed.std(ddof=0).to_numpy(),
            (observed.lt(40).sum() / observed.notna().sum()).to_numpy(),
            [locations[str(s)]["latitude"] for s in frame.columns],
            [locations[str(s)]["longitude"] for s in frame.columns],
        ]
    )
    if not np.isfinite(rows).all():
        raise ValueError("A sensor has no observed values in its first days")
    return rows


def ridge_predict(train_x, train_y, new_x, alpha):
    """Closed-form ridge regression on standardized descriptors, with an intercept."""
    center, spread = train_x.mean(axis=0), train_x.std(axis=0)
    spread = np.where(spread < 1e-9, 1.0, spread)
    a = (train_x - center) / spread
    b = (new_x - center) / spread
    a1 = np.column_stack([a, np.ones(len(a))])
    penalty = alpha * np.eye(a1.shape[1])
    penalty[-1, -1] = 0.0  # the intercept isn't shrunk
    weights = np.linalg.solve(a1.T @ a1 + penalty, a1.T @ train_y)
    return np.column_stack([b, np.ones(len(b))]) @ weights


def fit_vectors(model, trained_rows, trained, held, data, config, fed, seed, log):
    """Learn the new sensors' vectors with the rest of the network frozen.

    The windows come from the new sensors' first days only, the loss counts only
    their targets, and the epoch with the best validation MAE on them is kept.
    """
    for name, p in model.named_parameters():
        p.requires_grad_(name == "node")
    seed_all(seed, config.threads)
    with torch.no_grad():
        model.node[held] = trained_rows.mean(dim=0)
    early = data.parts["train"].iloc[: config.few_shot_days * ROWS_PER_DAY]
    windows = Windows(early, data.scaler, config.history, config.horizon, True)
    sampler = RandomSampler(
        windows, generator=torch.Generator().manual_seed(seed), num_samples=len(windows)
    )
    loader = DataLoader(windows, batch_size=config.batch_size, sampler=sampler)
    keep = torch.zeros(len(data.sensors), dtype=torch.bool)
    keep[torch.as_tensor(held, dtype=torch.long)] = True
    optimizer = torch.optim.Adam([model.node], lr=config.fit_lr)
    best, best_node, trace = np.inf, None, []
    for epoch in range(1, config.fit_epochs + 1):
        model.train()
        for x, y, mask in loader:
            sel = mask & keep
            if not sel.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            target = (y - data.mean) / data.std
            loss = (model(x)[sel] - target[sel]).abs().mean()
            loss.backward()
            model.node.grad[trained] = 0.0  # only the new sensors' rows may move
            optimizer.step()
        val = F.pool([F.evaluate(model, data, "val", fed, False, only=held)])["mae"]
        trace.append(val)
        if val < best:
            best, best_node = val, model.node.detach().clone()
        log(f"vector fit epoch {epoch}: held-out validation MAE {val:.4f}")
    with torch.no_grad():
        model.node.copy_(best_node)
    for p in model.parameters():
        p.requires_grad_(True)
    return {"validation_mae": trace, "best_epoch": int(np.argmin(trace)) + 1}


# --- scoring --------------------------------------------------------------------------


def score_held(model, data, held, fed):
    record = F.evaluate(model, data, "test", fed, True, only=held)
    return {"test": F.pool([record]), "daily": F.pool_days([record])}


def historical_average(frame, data, held, fed, days=None):
    """The training-mean table of each held-out sensor, scored on its test windows."""
    train = data.parts["train"]
    train = train if days is None else train.iloc[: days * ROWS_PER_DAY]
    columns = list(held)
    values = train.iloc[:, columns].to_numpy(dtype=np.float64)
    table = HistoricalAverage().fit(
        train.index, values, np.isfinite(values) & (values != 0)
    )
    test = data.datasets["test"]
    predicted = np.zeros((len(data.parts["test"]), len(data.sensors)), np.float32)
    predicted[:, columns] = table.predict(data.parts["test"].index)
    metrics = Metrics(fed.horizon)
    codes, labels = test.origin_days()
    daily = DailyMetrics(codes, labels)
    steps = np.arange(fed.horizon)
    keep = np.zeros(len(data.sensors), dtype=bool)
    keep[held] = True
    for start in range(0, len(test), 2048):
        origin = np.arange(start, min(start + 2048, len(test)))
        rows = origin[:, None] + test.history + steps[None, :]
        prediction = torch.from_numpy(predicted[rows])
        target = torch.from_numpy(test.y[rows])
        mask = torch.from_numpy(test.mask[rows] & keep)
        metrics.update(prediction, target, mask)
        daily.update(prediction, target, mask, start)
    record = {
        "ae": metrics.absolute.tolist(),
        "se": metrics.squared.tolist(),
        "count": metrics.count.tolist(),
        "daily": daily.result(),
    }
    return {"test": F.pool([record]), "daily": F.pool_days([record])}


def with_vectors(model, params, trained, held, rows_for_held):
    """Load trained parameters and put each sensor's vector where it belongs."""
    shared = OrderedDict((k, v) for k, v in params.items() if k != "node")
    with torch.no_grad():
        for (_, target), source in zip(
            [(n, p) for n, p in model.named_parameters() if n != "node"],
            shared.values(),
        ):
            target.copy_(source)
        model.node[trained] = params["node"]
        model.node[held] = rows_for_held


# --- one seed -------------------------------------------------------------------------


def run_seed(config, data_path, graph_dir, seed, out, log=print):
    from . import topology
    from .data import load

    began = time.perf_counter()
    frame = load(data_path, 207, config.rows)
    ids, weights, meta = topology.load(
        graph_dir, frame.columns, config.sensors, config.threshold
    )
    frame = frame.loc[:, ids]
    locations = meta["locations"]
    held = choose_holdout(ids, locations, config)
    trained = np.setdiff1d(np.arange(len(ids)), held)
    split_row = int(len(frame) * 0.7)
    fed = {kind: config.fed(kind, seed) for kind in KINDS}

    data_trained = F.ClientData(
        frame.iloc[:, trained], weights[np.ix_(trained, trained)], fed["sgru"]
    )
    data_oracle = F.ClientData(frame, weights, fed["sgru"])
    data_new = F.ClientData(
        frame,
        weights,
        fed["sgru"],
        scaler=mixed_scaler(frame, trained, held, config.few_shot_days, split_row),
    )
    result = {
        "seed": seed,
        "holdout": {
            "rule": config.holdout,
            "held_out": [ids[i] for i in held],
            "trained_on": len(trained),
        },
        "heldout": {},
        "trained_sensors_mae": {},
        "training": {},
    }

    log(f"seed {seed}: {len(trained)} sensors to train on, {len(held)} held out")
    persistence = F.evaluate(None, data_new, "test", fed["sgru"], True, only=held)
    result["heldout"]["persistence"] = {
        "test": F.pool([persistence]),
        "daily": F.pool_days([persistence]),
    }
    result["heldout"]["ha_first_days"] = historical_average(
        frame, data_new, held, fed["sgru"], config.few_shot_days
    )
    result["heldout"]["ha_all_train"] = historical_average(
        frame, data_oracle, held, fed["sgru"]
    )

    params = {}
    for kind in KINDS:
        start = F.initial_parameters(
            fed[kind], len(trained) if kind in F.POOLED_ONLY else 2
        )
        trained_result, params[kind] = F.run_pooled(
            fed[kind],
            data_trained,
            start,
            f"{kind} (trained sensors only)",
            [np.arange(len(trained))],
            log,
            return_params=True,
        )
        result["trained_sensors_mae"][kind] = trained_result["test"]["mae"]
        result["training"][kind] = {
            "best_round": trained_result["best_round"],
            "train_seconds": trained_result["train_seconds"],
            "parameters": trained_result["parameter_count"],
        }
        start = F.initial_parameters(
            fed[kind], len(ids) if kind in F.POOLED_ONLY else 2
        )
        oracle_result, oracle_params = F.run_pooled(
            fed[kind],
            data_oracle,
            start,
            f"{kind} (all sensors)",
            [trained, held],
            log,
            return_params=True,
        )
        model = F.build_model(fed[kind], weights)
        F.load_parameters(model, oracle_params)
        result["heldout"][f"oracle_{kind}"] = score_held(
            model, data_oracle, held, fed[kind]
        )
        result["training"][f"oracle_{kind}"] = {
            "best_round": oracle_result["best_round"],
            "train_seconds": oracle_result["train_seconds"],
            "parameters": oracle_result["parameter_count"],
        }

    for kind in ("sgru", "sgraph"):
        model = F.build_model(fed[kind], weights)
        F.load_parameters(model, params[kind])
        result["heldout"][f"{kind}_zero_shot"] = score_held(
            model, data_new, held, fed[kind]
        )

    kind = "sgru_id"
    model = F.build_model(fed[kind], weights)
    trained_rows = params[kind]["node"]
    for label, rows in (
        ("zero", torch.zeros(len(held), config.embedding)),
        ("mean", trained_rows.mean(dim=0).expand(len(held), -1).clone()),
    ):
        with_vectors(model, params[kind], trained, held, rows)
        result["heldout"][f"sgru_id_{label}"] = score_held(
            model, data_new, held, fed[kind]
        )

    seed_all(seed, config.threads)
    features = descriptors(frame, locations, config.few_shot_days)
    predicted = ridge_predict(
        features[trained],
        trained_rows.numpy().astype(np.float64),
        features[held],
        config.ridge_alpha,
    )
    with_vectors(
        model, params[kind], trained, held, torch.from_numpy(predicted).float()
    )
    result["heldout"]["sgru_id_ridge"] = score_held(model, data_new, held, fed[kind])

    with_vectors(
        model,
        params[kind],
        trained,
        held,
        trained_rows.mean(dim=0).expand(len(held), -1),
    )
    fit = fit_vectors(
        model, trained_rows, trained, held, data_new, config, fed[kind], seed, log
    )
    result["heldout"]["sgru_id_fitted"] = score_held(model, data_new, held, fed[kind])
    result["vector_fit"] = fit
    result["peak_memory"] = peak_memory()
    result["wall_seconds"] = time.perf_counter() - began
    result["config"] = asdict(config)
    write_json(Path(out) / f"seed-{seed}.json", result)
    return result


# --- summary and CLI ------------------------------------------------------------------


def _stats(values):
    values = [float(v) for v in values]
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def summarize(config, out):
    out = Path(out)
    runs = [
        json.loads((out / f"seed-{s}.json").read_text())
        for s in config.seeds
        if (out / f"seed-{s}.json").exists()
    ]
    if not runs:
        raise ValueError("No finished seeds")
    methods = {
        name: {
            "label": label,
            "mae": _stats([r["heldout"][name]["test"]["mae"] for r in runs]),
            "rmse": _stats([r["heldout"][name]["test"]["rmse"] for r in runs]),
            "horizon_mae": {
                str(h["minutes"]): _stats(
                    [r["heldout"][name]["test"]["per_horizon"][i]["mae"] for r in runs]
                )
                for i, h in enumerate(runs[0]["heldout"][name]["test"]["per_horizon"])
            },
        }
        for name, label in TARGETS.items()
    }
    daily = {n: [r["heldout"][n]["daily"] for r in runs] for n in TARGETS}
    paired = {f"{a}_vs_{b}": day_bootstrap(daily[a], daily[b]) for a, b in PAIRS}
    summary = {
        "config": asdict(config),
        "seeds": [r["seed"] for r in runs],
        "held_out": [r["holdout"]["held_out"] for r in runs],
        "methods": methods,
        "paired_day_bootstrap": paired,
        "paired_note": "Gaps in held-out test MAE (first minus second, negative favors the first) on identical targets; 95% intervals from resampling test days, training seeds averaged first.",
        "trained_sensors_mae": {
            kind: _stats([r["trained_sensors_mae"][kind] for r in runs])
            for kind in KINDS
        },
        "vector_fit_best_epoch": _stats([r["vector_fit"]["best_epoch"] for r in runs]),
        "training": {
            name: {
                "parameters": runs[0]["training"][name]["parameters"],
                "train_seconds": _stats(
                    [r["training"][name]["train_seconds"] for r in runs]
                ),
            }
            for name in runs[0]["training"]
        },
        **git_state(),
    }
    write_json(out / "summary.json", summary)
    figures(summary, out)
    return summary


def figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = summary["methods"]
    names = list(methods)
    colors = {
        "persistence": "#667085",
        "ha_first_days": "#a16207",
        "ha_all_train": "#ca8a04",
        "sgru_zero_shot": "#d97706",
        "sgraph_zero_shot": "#087e8b",
        "sgru_id_zero": "#fca5a5",
        "sgru_id_mean": "#f87171",
        "sgru_id_ridge": "#dc2626",
        "sgru_id_fitted": "#991b1b",
        "oracle_sgru": "#fcd34d",
        "oracle_sgraph": "#5eead4",
        "oracle_sgru_id": "#7f1d1d",
    }
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(
        x,
        [methods[n]["mae"]["mean"] for n in names],
        yerr=[methods[n]["mae"]["sd"] or 0 for n in names],
        color=[colors[n] for n in names],
        capsize=3,
    )
    ax.set_xticks(
        x, [methods[n]["label"] for n in names], rotation=40, ha="right", fontsize=7
    )
    low = min(methods[n]["mae"]["mean"] for n in names)
    ax.set_ylim(low - 0.3, max(methods[n]["mae"]["mean"] for n in names) + 0.15)
    ax.set(
        ylabel="Held-out sensors' test MAE (mph)",
        title="Forecasting sensors the model never trained on",
    )
    fig.savefig(Path(out) / "holdout-accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for n in (
        "persistence",
        "sgru_zero_shot",
        "sgraph_zero_shot",
        "sgru_id_mean",
        "sgru_id_fitted",
        "oracle_sgraph",
    ):
        steps = sorted(methods[n]["horizon_mae"], key=int)
        ax.plot(
            [int(s) for s in steps],
            [methods[n]["horizon_mae"][s]["mean"] for s in steps],
            label=methods[n]["label"],
            color=colors[n],
        )
    ax.set(
        xlabel="Forecast horizon (minutes)",
        ylabel="MAE on held-out sensors (mph)",
        title="Held-out sensors by horizon",
    )
    ax.legend(fontsize=6)
    fig.savefig(Path(out) / "holdout-horizons.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/holdout.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--topology", default="data/topology")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", help="comma-separated seeds (default: all)")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--case", type=int, help="internal: run one seed and stop")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.case is not None:
        run_seed(config, args.data, args.topology, args.case, out)
        return
    if not args.summarize_only:
        write_json(
            out / "config.json",
            asdict(config)
            | git_state()
            | {
                "python": sys.version,
                "platform": platform.platform(),
                "versions": {n: version(n) for n in ("torch", "numpy", "pandas")},
            },
        )
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config.seeds
        for seed in seeds:
            if (out / f"seed-{seed}.json").exists():
                print(f"seed {seed}: already done", flush=True)
                continue
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tracast.inductive",
                    "--config",
                    args.config,
                    "--data",
                    args.data,
                    "--topology",
                    args.topology,
                    "--out",
                    str(out),
                    "--case",
                    str(seed),
                ],
                check=True,
            )
    summarize(config, out)


if __name__ == "__main__":
    main()
