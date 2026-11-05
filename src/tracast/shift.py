"""Does a forecaster get worse as traffic drifts, and does adapting it help?

The test period is replayed day by day, the way the model would run for real. At the
start of each day a strategy may use data that has already been observed and nothing
else: window j is "known" at the first forecast of a day only if its last target row
is no later than that forecast's last input row. The strategies:

* static          the model as trained, never touched
* bias_ema        a running average of the recent residuals, added to the forecast
                  (no gradients, almost free)
* head_daily      every day, fine-tune only the output layer on the last few days of
                  known windows
* retrain_weekly  every week, fine-tune the whole network on everything known so far

Two settings: "standard" (the usual 70% train) and "stale", where the model is trained
on only the first 40% of the timeline, so a long stretch of unseen data sits between
what it learned from and the test period. Hyperparameters of the cheap strategies are
picked by replaying the validation part, never the test part.
"""

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .benchmark import BenchmarkConfig, fit, make_loaders, model_for, split_name
from .data import Scaler, Windows, load
from .experiment import seed_all, write_json
from .provenance import git_state
from . import topology
from .stats import day_bootstrap

STRATEGIES = ("static", "bias_ema", "head_daily", "retrain_weekly")


@dataclass
class ShiftConfig:
    benchmark: dict
    model: str = "gru_time"
    seeds: list[int] = field(default_factory=lambda: [7, 42, 2026])
    settings: dict = field(default_factory=lambda: {"standard": 0.7, "stale": 0.4})
    etas: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05, 0.1])
    head_days: int = 14
    head_lrs: list[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.002])
    head_epochs: int = 1
    retrain_every: int = 7
    retrain_epochs: int = 3
    retrain_lr: float = 0.0005

    def training(self):
        return BenchmarkConfig(
            **{**self.benchmark, "models": [self.model], "seeds": self.seeds}
        )


def load_config(path):
    return ShiftConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


# --- the replay stream ------------------------------------------------------------


def window_range(rows):
    """Window indices whose inputs and targets all lie inside rows [lo, hi)."""
    lo, hi = rows
    return np.arange(lo, hi - 23)  # history 12 + horizon 12 rows per window


def known_windows(first_window):
    """Windows whose last target row has been observed when window first_window is
    forecast: their last target row j + 23 is at most first_window's last input row
    first_window + 11, so j <= first_window - 12."""
    return first_window - 12


def day_ranges(dataset, idx):
    """(first, stop) window ranges of each calendar day of the forecast origin."""
    times = dataset.times[dataset.history + idx]
    codes, _ = pd.factorize(times.normalize())
    ranges = []
    for code in np.unique(codes):
        rows = idx[codes == code]
        ranges.append((int(rows[0]), int(rows[-1]) + 1))
    return ranges


def prepare(config, data, graph_dir, train_fraction):
    bench = config.training()
    frame = load(data, 207, bench.rows)
    ids, weights, _ = topology.load(
        graph_dir, frame.columns, bench.sensors, bench.threshold
    )
    frame = frame.loc[:, ids]
    n = len(frame)
    train_end = int(n * train_fraction)
    val_end = int(n * 0.8) if train_fraction >= 0.7 else train_end + int(n * 0.05)
    test_start = int(n * 0.8)
    scaler = Scaler().fit(frame.iloc[:train_end])
    timed = split_name(config.model)[1]
    # one dataset over the whole timeline: inputs are forward-filled from the past,
    # which is what a deployed model would have
    dataset = Windows(frame, scaler, bench.history, bench.horizon, timed)
    return {
        "frame": frame,
        "scaler": scaler,
        "dataset": dataset,
        "weights": weights,
        "bench": bench,
        "rows": {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (test_start, n),
        },
    }


def targets(dataset, idx):
    rows = idx[:, None] + dataset.history + np.arange(dataset.horizon)[None, :]
    return dataset.y[rows].astype(np.float64), dataset.mask[rows]


def predict(model, dataset, idx, mean, std):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), 512):
            chunk = idx[start : start + 512]
            x = torch.from_numpy(
                np.stack([dataset.x[i : i + dataset.history] for i in chunk])
            )
            out.append((model(x) * std + mean).numpy().astype(np.float64))
    return np.concatenate(out)


def finetune(model, dataset, idx, mean, std, epochs, lr, generator, head_only=False):
    """A few epochs of the usual masked L1 loss on the given windows."""
    if not len(idx):
        return
    params = list(model.head.parameters()) if head_only else list(model.parameters())
    frozen = []
    if head_only:
        keep = {id(p) for p in params}
        for p in model.parameters():
            if id(p) not in keep and p.requires_grad:
                p.requires_grad_(False)
                frozen.append(p)
    optimizer = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(
        Subset(dataset, idx.tolist()), batch_size=64, shuffle=True, generator=generator
    )
    model.train()
    for _ in range(epochs):
        for x, y, mask in loader:
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)[mask] - ((y - mean) / std)[mask]).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    for p in frozen:
        p.requires_grad_(True)


# --- strategies ---------------------------------------------------------------------


def bias_ema(static, target, mask, eta):
    """Add a running average of past residuals, per horizon step and sensor.

    The step-k target of window j arrives before window j + 1 + k is forecast, so at
    window i only windows up to i - 1 - k can have fed the average.
    """
    windows, horizons, sensors = static.shape
    residual = target - static
    bias = np.zeros((horizons, sensors))
    out = np.empty_like(static)
    steps = np.arange(horizons)
    for i in range(windows):
        feedback = i - 1 - steps
        valid = feedback >= 0
        for k in steps[valid]:
            j = feedback[k]
            seen = mask[j, k]
            bias[k, seen] += eta * (residual[j, k, seen] - bias[k, seen])
        out[i] = static[i] + bias
    return out


def replay(model, dataset, idx, mean, std, config, kind, seed, lr=None):
    """Run a model-updating strategy over the windows idx, one day at a time."""
    model = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    pred = np.empty((len(idx), dataset.horizon, len(mean)))
    adapt, updates, start_idx = 0.0, 0, int(idx[0])
    for day, (lo, hi) in enumerate(day_ranges(dataset, idx)):
        known = known_windows(lo)
        begin = time.perf_counter()
        if kind == "head_daily":
            low = max(0, known - config.head_days * 288 + 1)
            finetune(
                model,
                dataset,
                np.arange(low, known + 1),
                mean,
                std,
                config.head_epochs,
                lr,
                generator,
                head_only=True,
            )
            updates += 1
        elif kind == "retrain_weekly" and day % config.retrain_every == 0:
            finetune(
                model,
                dataset,
                np.arange(0, known + 1),
                mean,
                std,
                config.retrain_epochs,
                config.retrain_lr,
                generator,
            )
            updates += 1
        adapt += time.perf_counter() - begin
        pred[lo - start_idx : hi - start_idx] = predict(
            model, dataset, np.arange(lo, hi), mean, std
        )
    return pred, adapt, updates


def score(pred, target, mask, codes):
    """Overall and per-day error sums over observed targets."""
    error = np.where(mask, np.abs(pred - target), 0.0)
    days = np.unique(codes)
    sums = [float(error[codes == d].sum()) for d in days]
    counts = [int(mask[codes == d].sum()) for d in days]
    return {
        "mae": float(error.sum() / mask.sum()),
        "rmse": float(
            np.sqrt((np.where(mask, (pred - target) ** 2, 0.0)).sum() / mask.sum())
        ),
        "per_horizon_mae": [
            float(error[:, h].sum() / mask[:, h].sum()) for h in range(pred.shape[1])
        ],
        "daily": {"days": [int(d) for d in days], "abs": sums, "count": counts},
    }


# --- drift in the data itself ---------------------------------------------------------


def js_divergence(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a):
        keep = a > 0
        return float((a[keep] * np.log2(a[keep] / m[keep])).sum())

    return 0.5 * (kl(p) + kl(q))


def diagnostics(frame, train_end, block_days=14):
    """Block by block: mean speed, congested share, missing share, and how far the
    speed distribution sits from the training period's (Jensen-Shannon, bits)."""
    values = frame.to_numpy(dtype=np.float64)
    missing = np.isnan(values) | (values == 0)
    speeds = np.where(missing, np.nan, values)
    bins = np.arange(0, 90, 10)
    base = np.histogram(speeds[:train_end][~missing[:train_end]], bins=bins)[0].astype(
        float
    )
    step, rows = block_days * 288, []
    for start in range(0, len(frame), step):
        stop = min(start + step, len(frame))
        seen = ~missing[start:stop]
        hist = np.histogram(speeds[start:stop][seen], bins=bins)[0].astype(float)
        rows.append(
            {
                "start": str(frame.index[start].date()),
                "rows": stop - start,
                "mean_speed": float(np.nanmean(speeds[start:stop])),
                "congested_share": float((speeds[start:stop][seen] < 40).mean()),
                "missing_share": float(missing[start:stop].mean()),
                "js_vs_train_bits": js_divergence(hist, base),
                "in_train": stop <= train_end,
            }
        )
    return rows


def daily_regimes(frame, dataset, idx, train_end):
    """Per test day: how far its speeds are from the training distribution (bits)."""
    values = frame.to_numpy(dtype=np.float64)
    missing = np.isnan(values) | (values == 0)
    speeds = np.where(missing, np.nan, values)
    bins = np.arange(0, 90, 10)
    base = np.histogram(speeds[:train_end][~missing[:train_end]], bins=bins)[0].astype(
        float
    )
    out = []
    for lo, hi in day_ranges(dataset, idx):
        rows = slice(lo + dataset.history, hi + dataset.history)
        seen = ~missing[rows]
        hist = np.histogram(speeds[rows][seen], bins=bins)[0].astype(float)
        stamp = frame.index[lo + dataset.history]
        out.append(
            {
                "date": str(stamp.date()),
                "weekend": bool(stamp.dayofweek >= 5),
                "js_vs_train_bits": js_divergence(hist, base),
                "missing_share": float(missing[rows].mean()),
            }
        )
    return out


# --- one setting, one seed --------------------------------------------------------------


def run_setting(config, data, graph_dir, name, fraction, seed, out):
    seed_all(seed, config.training().threads)
    prep = prepare(config, data, graph_dir, fraction)
    dataset, bench = prep["dataset"], prep["bench"]
    mean, std = (
        torch.from_numpy(prep["scaler"].mean),
        torch.from_numpy(prep["scaler"].std),
    )
    train_idx, val_idx, test_idx = (
        window_range(prep["rows"][k]) for k in ("train", "val", "test")
    )
    out.mkdir(parents=True, exist_ok=True)

    model = model_for(config.model, bench, prep["weights"])
    generator = torch.Generator().manual_seed(seed)
    loaders = make_loaders(
        {
            "train": Subset(dataset, train_idx.tolist()),
            "val": Subset(dataset, val_idx.tolist()),
        },
        bench,
        generator,
    )
    fit(model, loaders, mean, std, bench, f"{name} seed={seed}", out)
    model.load_state_dict(
        torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    )

    # pick the cheap strategies' settings by replaying validation, never test
    val_target, val_mask = targets(dataset, val_idx)
    val_static = predict(model, dataset, val_idx, mean, std)
    etas = {
        eta: float(
            np.abs(bias_ema(val_static, val_target, val_mask, eta) - val_target)[
                val_mask
            ].mean()
        )
        for eta in config.etas
    }
    eta = min(etas, key=etas.get)
    head = {}
    for lr in config.head_lrs:
        pred, _, _ = replay(
            model, dataset, val_idx, mean, std, config, "head_daily", seed, lr
        )
        head[lr] = float(np.abs(pred - val_target)[val_mask].mean())
    lr = min(head, key=head.get)

    target, mask = targets(dataset, test_idx)
    codes, _ = pd.factorize(dataset.times[dataset.history + test_idx].normalize())
    begin = time.perf_counter()
    static = predict(model, dataset, test_idx, mean, std)
    static_seconds = time.perf_counter() - begin
    results = {
        "static": {
            **score(static, target, mask, codes),
            "adapt_seconds": 0.0,
            "updates": 0,
            "predict_seconds": static_seconds,
        }
    }
    begin = time.perf_counter()
    biased = bias_ema(static, target, mask, eta)
    results["bias_ema"] = {
        **score(biased, target, mask, codes),
        "adapt_seconds": time.perf_counter() - begin,
        "updates": len(test_idx),
    }
    for kind in ("head_daily", "retrain_weekly"):
        pred, seconds, updates = replay(
            model, dataset, test_idx, mean, std, config, kind, seed, lr
        )
        results[kind] = {
            **score(pred, target, mask, codes),
            "adapt_seconds": seconds,
            "updates": updates,
        }
    result = {
        "setting": name,
        "train_fraction": fraction,
        "seed": seed,
        "rows": prep["rows"],
        "chosen": {
            "eta": eta,
            "head_lr": lr,
            "eta_val_mae": etas,
            "head_val_mae": head,
        },
        "strategies": results,
        "regimes": daily_regimes(
            prep["frame"], dataset, test_idx, prep["rows"]["train"][1]
        ),
        "drift": diagnostics(prep["frame"], prep["rows"]["train"][1]),
    }
    write_json(out / "result.json", result)
    return result


def run(config, data, graph_dir, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json", {**json.loads(json.dumps(vars(config))), **git_state()}
    )
    summary = {"settings": {}, **git_state()}
    for name, fraction in config.settings.items():
        runs = []
        for seed in config.seeds:
            folder = out / f"{name}-seed-{seed}"
            done = folder / "result.json"
            runs.append(
                json.loads(done.read_text())
                if done.exists()
                else run_setting(config, data, graph_dir, name, fraction, seed, folder)
            )
        summary["settings"][name] = aggregate(runs)
    write_json(out / "summary.json", summary)
    return summary


def aggregate(runs):
    """Mean over seeds, plus paired test-day intervals against the static model."""
    out = {
        "seeds": [r["seed"] for r in runs],
        "rows": runs[0]["rows"],
        "strategies": {},
    }
    for kind in STRATEGIES:
        entries = [r["strategies"][kind] for r in runs]
        mae = [e["mae"] for e in entries]
        entry = {
            "mae": float(np.mean(mae)),
            "mae_sd": float(np.std(mae, ddof=1)) if len(mae) > 1 else None,
            "rmse": float(np.mean([e["rmse"] for e in entries])),
            "per_horizon_mae": np.mean(
                [e["per_horizon_mae"] for e in entries], axis=0
            ).tolist(),
            "adapt_seconds": float(np.mean([e["adapt_seconds"] for e in entries])),
            "updates": entries[0]["updates"],
            "daily_mae": (
                np.mean(
                    [
                        np.array(e["daily"]["abs"]) / np.array(e["daily"]["count"])
                        for e in entries
                    ],
                    axis=0,
                )
            ).tolist(),
        }
        if kind != "static":
            entry["vs_static"] = day_bootstrap(
                [
                    e["daily"] | {"days": [str(d) for d in e["daily"]["days"]]}
                    for e in entries
                ],
                [
                    r["strategies"]["static"]["daily"]
                    | {
                        "days": [
                            str(d) for d in r["strategies"]["static"]["daily"]["days"]
                        ]
                    }
                    for r in runs
                ],
            )
        out["strategies"][kind] = entry
    # where does adapting help: the most drifted test days against the least
    regimes = runs[0]["regimes"]
    shift = np.array([d["js_vs_train_bits"] for d in regimes])
    order = np.argsort(shift)
    third = max(1, len(order) // 3)
    groups = {
        "least_shifted_third": order[:third],
        "most_shifted_third": order[-third:],
        "weekday": [i for i, d in enumerate(regimes) if not d["weekend"]],
        "weekend": [i for i, d in enumerate(regimes) if d["weekend"]],
    }
    out["regimes"] = {}
    for name, days in groups.items():
        days = list(days)
        if not days:
            continue
        out["regimes"][name] = {
            kind: float(
                np.mean(
                    [
                        np.array(r["strategies"][kind]["daily"]["abs"])[days].sum()
                        / np.array(r["strategies"][kind]["daily"]["count"])[days].sum()
                        for r in runs
                    ]
                )
            )
            for kind in STRATEGIES
        }
    out["drift"] = runs[0]["drift"]
    out["chosen"] = [r["chosen"] for r in runs]
    return out


def figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    colors = {
        "static": "#667085",
        "bias_ema": "#16a34a",
        "head_daily": "#087e8b",
        "retrain_weekly": "#dc2626",
    }
    labels = {
        "static": "Static",
        "bias_ema": "Residual average",
        "head_daily": "Head fine-tune (daily)",
        "retrain_weekly": "Full fine-tune (weekly)",
    }
    settings = summary["settings"]

    first = next(iter(settings.values()))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    blocks = first["drift"]
    xs = [b["start"] for b in blocks]
    for ax, key, title in zip(
        axes,
        ("mean_speed", "congested_share", "missing_share"),
        (
            "Mean observed speed (mph)",
            "Share of readings below 40 mph",
            "Share of missing readings",
        ),
    ):
        ax.plot(xs, [b[key] for b in blocks], marker="o", markersize=3, color="#1d4ed8")
        ax.set(title=title)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        train_blocks = sum(b["in_train"] for b in blocks)
        ax.axvline(train_blocks - 0.5, color="#999999", linestyle=":")
    fig.savefig(out / "shift-drift.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=(6.5 * len(settings), 4),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (name, s) in zip(axes[0], settings.items()):
        for kind in STRATEGIES:
            days = s["strategies"][kind]["daily_mae"]
            ax.plot(
                range(1, len(days) + 1),
                days,
                color=colors[kind],
                label=labels[kind],
                marker="o",
                markersize=2.5,
            )
        ax.set(
            title=f"Test MAE per day ({name} model)",
            xlabel="Test day",
            ylabel="MAE (mph)",
        )
    axes[0][0].legend(fontsize=7)
    fig.savefig(out / "shift-daily.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=(6.5 * len(settings), 3.8),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (name, s) in zip(axes[0], settings.items()):
        kinds = [k for k in STRATEGIES if k != "static"]
        diff = [s["strategies"][k]["vs_static"] for k in kinds]
        ax.barh(
            range(len(kinds)),
            [d["estimate"] for d in diff],
            color=[colors[k] for k in kinds],
            xerr=[
                [d["estimate"] - d["ci95"][0] for d in diff],
                [d["ci95"][1] - d["estimate"] for d in diff],
            ],
            capsize=3,
        )
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_yticks(range(len(kinds)), [labels[k] for k in kinds])
        ax.set(
            title=f"MAE change vs static ({name} model)",
            xlabel="mph (negative = better)",
        )
    fig.savefig(out / "shift-gain.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Drift and adaptation experiment")
    parser.add_argument("--config", default="configs/shift.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--topology", default="data/topology")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = run(load_config(args.config), args.data, args.topology, args.out)
    figures(summary, args.out)
    for name, s in summary["settings"].items():
        print(name, {k: round(v["mae"], 4) for k, v in s["strategies"].items()})


if __name__ == "__main__":
    main()
