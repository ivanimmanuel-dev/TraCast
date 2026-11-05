"""Matched centralized forecasting trials on a fixed METR-LA sensor graph."""

import argparse
from dataclasses import dataclass, asdict, field
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .baselines import HistoricalAverage, score_table
from .data import SHA256, Scaler, Windows, load, split, missing_report
from .experiment import seed_all, evaluate, write_json, code_manifest
from .federated_task import state_hash
from .graph_model import GraphGRU
from .model import GRU, DailyMetrics, LinearAR
from .provenance import git_state
from .resources import peak_memory
from .stats import day_bootstrap
from . import topology

# A _time suffix gives a model the calendar inputs. ha and persistence don't train.
# dense models read every sensor at once; the s* models share one recurrent net
# across sensors (see shared.py); an _id suffix adds a learned vector per sensor
BASES = (
    "gru",
    "lstm",
    "graph_gru",
    "graph_self",
    "linear",
    "sgru",
    "slstm",
    "sgraph",
    "sgru_id",
    "sgraph_id",
)
STATIC = ("persistence", "ha")
TRIALS = STATIC + BASES + tuple(f"{base}_time" for base in BASES)
SCHEDULERS = ("none", "plateau")


def split_name(name):
    """Return the architecture and whether it receives calendar channels."""
    timed = name.endswith("_time")
    return (name[: -len("_time")] if timed else name), timed


def valid_model(name):
    return name == "ha" or split_name(name)[0] in BASES


@dataclass
class BenchmarkConfig:
    sensors: int = 16
    rows: int = 8064
    history: int = 12
    horizon: int = 12
    hidden: int = 32
    embedding: int = 4
    batch_size: int = 64
    epochs: int = 5
    lr: float = 0.001
    seeds: list[int] = field(default_factory=lambda: [7, 42, 2026])
    threads: int = 4
    threshold: float = 0.1
    models: list[str] = field(default_factory=lambda: ["gru", "graph_gru"])
    # defaults give the original fixed-epoch protocol
    patience: int | None = None
    train_stride: int = 1
    core_sizes: list[int] = field(default_factory=list)
    scheduler: str = "none"
    scheduler_patience: int = 2

    @property
    def uses_time(self):
        return any(split_name(name)[1] for name in self.models)

    def validate(self):
        for name in (
            "sensors",
            "rows",
            "history",
            "horizon",
            "hidden",
            "embedding",
            "batch_size",
            "epochs",
            "threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        if self.sensors > 207 or self.rows > 34272:
            raise ValueError("Subset exceeds METR-LA")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds)
        ):
            raise ValueError("Expected unique integer seeds")
        if not math.isfinite(self.lr) or self.lr <= 0 or not 0 < self.threshold < 1:
            raise ValueError("Invalid learning rate or distance threshold")
        if (
            not self.models
            or len(set(self.models)) != len(self.models)
            or not all(valid_model(name) for name in self.models)
        ):
            raise ValueError("Invalid model list")
        if self.patience is not None and (
            type(self.patience) is not int or self.patience < 1
        ):
            raise ValueError("patience must be null or a positive integer")
        if type(self.train_stride) is not int or self.train_stride < 1:
            raise ValueError("train_stride must be a positive integer")
        if any(
            type(k) is not int or not 1 <= k <= self.sensors for k in self.core_sizes
        ):
            raise ValueError("core_sizes must be sensor counts up to sensors")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}")
        if type(self.scheduler_patience) is not int or self.scheduler_patience < 1:
            raise ValueError("scheduler_patience must be a positive integer")


def model_for(name, config, weights):
    if name in STATIC:
        return None
    base, timed = split_name(name)
    extra = 3 if timed else 0
    if base == "gru":
        return GRU(config.sensors, config.hidden, config.horizon, extra)
    if base == "lstm":
        from .recurrent import LSTM

        return LSTM(config.sensors, config.hidden, config.horizon, extra)
    if base in ("sgru", "slstm", "sgraph", "sgru_id", "sgraph_id"):
        from .shared import SharedRecurrent

        return SharedRecurrent(
            config.sensors,
            config.hidden,
            config.horizon,
            extra,
            weights if base.startswith("sgraph") else None,
            kind="lstm" if base == "slstm" else "gru",
            embedding=config.embedding if base.endswith("_id") else 0,
        )
    if base == "linear":
        return LinearAR(config.sensors, config.history, config.horizon, extra)
    if base in ("graph_gru", "graph_self"):
        graph = weights if base == "graph_gru" else np.zeros_like(weights)
        return GraphGRU(graph, config.hidden, config.horizon, config.embedding, extra)
    raise ValueError(f"Unknown model: {name}")


def prepared(config, data, graph_dir, time_features=None, calibration=0.0):
    if time_features is None:
        time_features = config.uses_time
    frame = load(data, 207, config.rows)
    ids, weights, graph = topology.load(
        graph_dir, frame.columns, config.sensors, config.threshold
    )
    frame = frame.loc[:, ids]
    parts = split(frame, config.history, config.horizon, calibration)
    scaler = Scaler().fit(parts["train"])
    datasets = {
        name: Windows(part, scaler, config.history, config.horizon, time_features)
        for name, part in parts.items()
    }
    protocol = {
        "graph": graph,
        "missing": {name: missing_report(part) for name, part in parts.items()},
        "scaler": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()},
        "splits": {
            name: {
                "rows": len(part),
                "windows": len(datasets[name]),
                "start": str(part.index[0]),
                "end": str(part.index[-1]),
            }
            for name, part in parts.items()
        },
    }
    return datasets, scaler, weights, protocol


def timed_calls(predict, batch_size, scope):
    with torch.inference_mode():
        for _ in range(5):
            predict()
        seconds = []
        for _ in range(30):
            start = time.perf_counter()
            predict()
            seconds.append(time.perf_counter() - start)
    return {
        "batch_size": batch_size,
        "repetitions": 30,
        "warmups": 5,
        "median_batch_seconds": statistics.median(seconds),
        "median_ms_per_window": statistics.median(seconds) * 1000 / batch_size,
        "scope": scope,
    }


def inference(model, x, horizon, sensors):
    if model is not None:
        model.eval()

    def predict():
        if model is None:
            return x[:, -1:, :sensors].expand(-1, horizon, -1)
        return model(x)

    return timed_calls(
        predict,
        len(x),
        "Normalized forward pass only; fixed first test batch; excludes loader and metrics",
    )


def make_loaders(datasets, config, generator):
    """One loader per split; with train_stride k an epoch is a random 1/k of the windows."""
    loaders = {}
    for stage, ds in datasets.items():
        options = {"batch_size": config.batch_size, "num_workers": 0}
        if stage == "train" and config.train_stride > 1:
            options["sampler"] = RandomSampler(
                ds, num_samples=len(ds) // config.train_stride, generator=generator
            )
        else:
            options["shuffle"] = stage == "train"
            options["generator"] = generator if stage == "train" else None
        loaders[stage] = DataLoader(ds, **options)
    return loaders


def fit(model, loaders, mean, std, config, label, out, select="val"):
    """Train with the config's schedule, scoring loaders[select] after every epoch.

    The best-validation weights go to out/best.pt (the caller loads them) and the
    epoch history to out/history.json.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=config.scheduler_patience
        )
        if config.scheduler == "plateau"
        else None
    )
    history = []
    best, stale, stopped_early = float("inf"), 0, False
    train_seconds, validation_seconds = 0.0, 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        rate = optimizer.param_groups[0]["lr"]
        loss_sum, count_sum, steps = 0.0, 0, 0
        start = time.perf_counter()
        for x, y, mask in loaders["train"]:
            count = int(mask.sum())
            if not count:
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            target = (y - mean) / std
            loss = (prediction[mask] - target[mask]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite benchmark loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * count
            count_sum += count
            steps += 1
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        if not count_sum:
            raise ValueError("No supervised training targets")
        start = time.perf_counter()
        validation = evaluate(model, loaders[select], mean, std, config.horizon)
        validation_elapsed = time.perf_counter() - start
        validation_seconds += validation_elapsed
        row = {
            "epoch": epoch,
            "lr": rate,
            "train_normalized_mae": loss_sum / count_sum,
            "observed_targets": count_sum,
            "optimizer_steps": steps,
            "train_seconds": elapsed,
            "validation_seconds": validation_elapsed,
            "validation": validation,
        }
        history.append(row)
        if validation["mae"] < best:
            best, stale = validation["mae"], 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(validation["mae"])
        write_json(out / "history.json", history)
        print(
            f"{label} epoch={epoch} val_MAE={validation['mae']:.6f} train_s={elapsed:.3f}",
            flush=True,
        )
        if config.patience is not None and stale >= config.patience:
            stopped_early = epoch < config.epochs
            break
    return {
        "history": history,
        "stopped_early": stopped_early,
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
    }


def trial(config, data, graph_dir, name, seed, out):
    config.validate()
    seed_all(seed, config.threads)
    datasets, scaler, weights, protocol = prepared(
        config, data, graph_dir, config.uses_time or split_name(name)[1]
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "running"})
    try:
        model = model_for(name, config, weights)
        generator = torch.Generator().manual_seed(seed)
        loaders = make_loaders(datasets, config, generator)
        mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
        if model is not None:
            trained = fit(model, loaders, mean, std, config, f"{name} seed={seed}", out)
            history = trained["history"]
            stopped_early = trained["stopped_early"]
            train_seconds = trained["train_seconds"]
            validation_seconds = trained["validation_seconds"]
        else:
            history, stopped_early = [], False
            train_seconds = validation_seconds = 0.0
        if model is not None:
            model.load_state_dict(
                torch.load(out / "best.pt", map_location="cpu", weights_only=True)
            )
        test_set = datasets["test"]
        cores = [k for k in config.core_sizes if k < config.sensors]
        codes, labels = test_set.origin_days()
        daily = DailyMetrics(codes, labels)
        start = time.perf_counter()
        if name == "ha":
            train_set = datasets["train"]
            fitted = HistoricalAverage().fit(
                train_set.times, train_set.y, train_set.mask
            )
            table = fitted.predict(test_set.times)
            test = score_table(test_set, table, daily, subsets=cores)
        else:
            test = evaluate(
                model,
                loaders["test"],
                mean,
                std,
                config.horizon,
                daily=daily,
                subsets=cores,
            )
        test_seconds = time.perf_counter() - start
        x = next(iter(loaders["test"]))[0]
        if name == "ha":
            rows = (
                np.arange(len(x))[:, None]
                + test_set.history
                + np.arange(config.horizon)[None, :]
            )
            timing = timed_calls(
                lambda: table[rows],
                len(x),
                "Table lookup for the first test batch; excludes fitting, loader, and metrics",
            )
        else:
            timing = inference(model, x, config.horizon, config.sensors)
        result = {
            "model": name,
            "seed": seed,
            "test": test,
            "daily": daily.result(),
            "history": history,
            "stopped_early": stopped_early,
            "best_epoch": (
                min(history, key=lambda r: r["validation"]["mae"])["epoch"]
                if history
                else None
            ),
            "parameter_count": (
                sum(p.numel() for p in model.parameters()) if model is not None else 0
            ),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "test_evaluation_seconds": test_seconds,
            "inference": timing,
            "peak_memory": peak_memory(),
            "state_sha256": state_hash(model) if model is not None else None,
        }
        write_json(out / "result.json", result)
        write_json(out / "protocol.json", protocol)
        write_json(out / "status.json", {"status": "complete"})
        return result
    except BaseException as error:
        write_json(
            out / "status.json",
            {"status": "failed", "type": type(error).__name__, "error": str(error)},
        )
        raise


def aggregate(results):
    summary = {}
    for name in dict.fromkeys(r["model"] for r in results):
        values = [r for r in results if r["model"] == name]

        def moments(items):
            return {
                "mean": float(np.mean(items)),
                "sd": float(np.std(items, ddof=1)) if len(items) > 1 else None,
            }

        summary[name] = {
            "n": len(values),
            "seeds": [r["seed"] for r in values],
            "parameter_count": values[0]["parameter_count"],
            **{
                metric: moments([r["test"][metric] for r in values])
                for metric in ("mae", "rmse")
            },
            "train_seconds": moments([r["train_seconds"] for r in values]),
            "inference_ms_per_window": moments(
                [r["inference"]["median_ms_per_window"] for r in values]
            ),
            "peak_process_mib": moments(
                [r["peak_memory"]["bytes"] / 1024**2 for r in values]
            ),
            "horizons": [
                {
                    "minutes": values[0]["test"]["per_horizon"][i]["minutes"],
                    **{
                        metric: moments(
                            [r["test"]["per_horizon"][i][metric] for r in values]
                        )
                        for metric in ("mae", "rmse")
                    },
                }
                for i in range(len(values[0]["test"]["per_horizon"]))
            ],
        }
    for name, entry in summary.items():
        values = [r for r in results if r["model"] == name]
        if values[0]["history"]:
            entry["best_epoch"] = moments([r["best_epoch"] for r in values])
            entry["epochs_run"] = moments([len(r["history"]) for r in values])
    paired = {}
    reference = {r["seed"]: r for r in results if r["model"] == "gru"}
    for name in summary:
        if name == "gru" or name in STATIC:
            continue
        values = [
            r["test"]["mae"] - reference[r["seed"]]["test"]["mae"]
            for r in results
            if r["model"] == name and r["seed"] in reference
        ]
        if values:
            paired[name] = {
                "metric": "MAE difference versus GRU; negative favors candidate",
                "differences": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    report = {
        "models": summary,
        "paired_mae": paired,
        "uncertainty": "Sample SD across training seeds; common fixed test block",
    }
    if results and all("daily" in r for r in results):
        records = {
            name: [r["daily"] for r in results if r["model"] == name]
            for name in summary
        }
        for name, entry in summary.items():
            entry["test_mae_ci95_days"] = day_bootstrap(records[name])["ci95"]
        learned = [name for name in summary if name not in STATIC]
        pairs = [
            (name, "gru") for name in learned if name != "gru" and "gru" in summary
        ]
        if "graph_gru" in summary and "graph_self" in summary:
            pairs.append(("graph_gru", "graph_self"))
        for name in learned:
            plain = split_name(name)[0]
            if name != plain and plain in summary:
                pairs.append((name, plain))
        # with calendar inputs, compare against the GRU that also has them
        if "gru_time" in summary:
            pairs += [
                (name, "gru_time")
                for name in learned
                if name.endswith("_time") and name != "gru_time"
            ]
        if "graph_gru_time" in summary and "graph_self_time" in summary:
            pairs.append(("graph_gru_time", "graph_self_time"))
        if "ha" in summary:
            pairs += [(name, "ha") for name in learned]
            pairs.append(("ha", "persistence"))
        report["day_bootstrap"] = {
            f"{first}_vs_{second}": day_bootstrap(records[first], records[second])
            for first, second in dict.fromkeys(pairs)
            if second in records
        }
        report["day_bootstrap_note"] = (
            "Negative estimates favor the first model. Intervals resample test days "
            "with replacement after averaging training seeds, so they describe "
            "variation across the held-out dates, not across initializations."
        )
    return report


def plots(out, results, summary, protocol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary["models"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, value in models.items():
        for metric, ax in zip(("mae", "rmse"), axes):
            x = [r["minutes"] for r in value["horizons"]]
            y = np.array([r[metric]["mean"] for r in value["horizons"]])
            sd = np.array([r[metric]["sd"] or 0 for r in value["horizons"]])
            ax.plot(x, y, label=name)
            ax.fill_between(x, y - sd, y + sd, alpha=0.12)
            ax.set(
                xlabel="Forecast horizon (minutes)",
                ylabel="mph",
                title=f"Test {metric.upper()}",
            )
            ax.legend(fontsize=8)
    fig.savefig(out / "horizons.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in models:
        values = [r for r in results if r["model"] == name and r["history"]]
        if not values:
            continue
        for metric, ax in zip(("train_normalized_mae", "validation"), axes):
            curves = [
                [
                    h[metric] if metric != "validation" else h[metric]["mae"]
                    for h in r["history"]
                ]
                for r in values
            ]
            # early stopping gives runs different lengths, so each epoch
            # averages the runs still training
            columns = [
                [curve[i] for curve in curves if len(curve) > i]
                for i in range(max(map(len, curves)))
            ]
            y = np.array([np.mean(column) for column in columns])
            sd = np.array(
                [
                    np.std(column, ddof=1) if len(column) > 1 else 0.0
                    for column in columns
                ]
            )
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, label=name)
            ax.fill_between(x, y - sd, y + sd, alpha=0.12)
            ax.set(
                xlabel="Epoch",
                ylabel="Normalized MAE" if metric != "validation" else "MAE (mph)",
                title=(
                    "Training loss"
                    if metric != "validation"
                    else "Validation convergence"
                ),
            )
            ax.legend(fontsize=8)
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)
    ids = protocol["graph"]["sensor_ids"]
    weights = np.asarray(protocol["graph"]["adjacency"])
    coords = protocol["graph"]["locations"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(weights, cmap="viridis", vmin=0, vmax=1)
    axes[0].set(
        title="Directed distance-kernel weights",
        xlabel="Destination index",
        ylabel="Source index",
    )
    for i, start in enumerate(ids):
        for j, end in enumerate(ids):
            if weights[i, j] > 0:
                axes[1].plot(
                    [coords[start]["longitude"], coords[end]["longitude"]],
                    [coords[start]["latitude"], coords[end]["latitude"]],
                    color="#7a8793",
                    alpha=0.28,
                    linewidth=0.6,
                )
    axes[1].scatter(
        [coords[s]["longitude"] for s in ids],
        [coords[s]["latitude"] for s in ids],
        s=25,
        color="#176b87",
    )
    for i, sensor in enumerate(ids):
        axes[1].annotate(
            str(i),
            (coords[sensor]["longitude"], coords[sensor]["latitude"]),
            fontsize=7,
        )
    axes[1].set(
        title="Selected sensors and source graph relationships",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    fig.savefig(out / "topology.png", dpi=150)
    plt.close(fig)


def run(config, data, graph_dir, out):
    config.validate()
    topology.fetch(graph_dir)
    _, _, _, protocol = prepared(config, data, graph_dir)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", asdict(config))
    source = code_manifest()
    identity = {
        "config": asdict(config),
        "data_sha256": SHA256,
        "topology_files": topology.FILES,
        "code": source,
    }
    write_json(
        out / "manifest.json",
        {
            **identity,
            "run_id": hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest(),
            **git_state(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": sys.argv,
            "versions": {
                n: version(n) for n in ("numpy", "pandas", "torch", "matplotlib")
            },
            "timing": "Sequential fresh processes; training updates only; forward-pass microbenchmark separate from full test scoring",
        },
    )
    write_json(out / "protocol.json", protocol)
    write_json(out / "status.json", {"status": "running"})
    results = []
    try:
        # persistence and ha don't train, so they run once
        static = ["persistence"] + [m for m in config.models if m in STATIC]
        learned = [m for m in config.models if m not in STATIC]
        tasks = [(m, config.seeds[0]) for m in static] + [
            (m, s) for s in config.seeds for m in learned
        ]
        for name, seed in tasks:
            target = out / f"{name}-seed-{seed}"
            command = [
                sys.executable,
                "-m",
                "tracast.benchmark",
                "--config",
                str(out / "config.json"),
                "--data",
                str(Path(data).resolve()),
                "--topology",
                str(Path(graph_dir).resolve()),
                "--out",
                str(target),
                "--trial",
                name,
                "--seed",
                str(seed),
            ]
            with (out / f"{name}-seed-{seed}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            result = json.loads((target / "result.json").read_text())
            results.append(result)
            print(
                f"{name} seed={seed} test_MAE={result['test']['mae']:.6f} train_s={result['train_seconds']:.2f}",
                flush=True,
            )
        summary = aggregate(results)
        write_json(out / "results.json", results)
        write_json(out / "summary.json", summary)
        plots(out, results, summary, protocol)
        write_json(out / "status.json", {"status": "complete"})
        return summary
    except BaseException as error:
        write_json(
            out / "status.json",
            {"status": "failed", "type": type(error).__name__, "error": str(error)},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--topology", default="data/topology")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trial", choices=TRIALS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = BenchmarkConfig(**json.loads(Path(args.config).read_text()))
    if args.trial:
        trial(config, args.data, args.topology, args.trial, args.seed, args.out)
    else:
        run(config, args.data, args.topology, args.out)


if __name__ == "__main__":
    main()
