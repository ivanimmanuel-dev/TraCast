"""Prediction intervals for the traffic forecasts: naive baselines vs conformal methods.

One point forecaster is trained as usual (checkpoints picked on the "val" part) and
every interval method is then built from it without touching test data:

* gaussian_train  forecast +- z * (std of the training residuals)
* gaussian_cal    the same, with the std taken from the calibration part
* conformal_pooled / conformal_sensor   split conformal, one quantile per horizon or
                  per horizon and sensor, calibrated on the "cal" part
* conformal_aci   adaptive conformal inference, updated online with the real delay
* conformal_rolling   recalibrated every day from the residuals already known
* quantile_raw / cqr   a second network trained with the pinball loss for the two
                  quantiles, as is and conformalized on the "cal" part

The cal part sits between validation and test, so it is never used for training or
for choosing a checkpoint.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import json
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import conformal as C
from .benchmark import (
    BenchmarkConfig,
    fit,
    make_loaders,
    model_for,
    prepared,
    split_name,
)
from .experiment import seed_all, write_json
from .provenance import git_state

POINT_METHODS = (
    "gaussian_train",
    "gaussian_cal",
    "conformal_pooled",
    "conformal_sensor",
    "conformal_aci",
    "conformal_rolling",
)
QUANTILE_METHODS = ("quantile_raw", "cqr")


@dataclass
class UncertaintyConfig:
    benchmark: dict
    model: str = "gru_time"
    seeds: list[int] = field(default_factory=lambda: [7, 42, 2026])
    calibration: float = 0.05
    alpha: float = 0.1
    reliability_alphas: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5]
    )
    gamma: float = 0.005
    gamma_sweep: list[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.005, 0.02, 0.05]
    )
    rolling_days: int = 7
    congestion_mph: float = 40.0

    def training(self):
        return BenchmarkConfig(
            **{**self.benchmark, "models": [self.model], "seeds": self.seeds}
        )


def load_config(path):
    return UncertaintyConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def collect(model, dataset, mean, std):
    """Forecasts (mph), targets, masks and last observed speeds for every window."""
    model.eval()
    sensors = len(mean)
    parts = {"pred": [], "target": [], "mask": [], "last": []}
    with torch.no_grad():
        for x, y, mask in DataLoader(dataset, batch_size=256, shuffle=False):
            parts["pred"].append((model(x) * std + mean).numpy())
            parts["target"].append(y.numpy())
            parts["mask"].append(mask.numpy())
            parts["last"].append((x[:, -1, :sensors] * std + mean).numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def fit_quantile(model, loaders, mean, std, config, alpha, label):
    """Train a network that outputs the lower and upper quantile for every target.

    The output has 2 * horizon steps: the first half is the alpha/2 quantile and the
    second half the 1 - alpha/2 quantile. The loss is the pinball loss in normalized
    units, the best epoch on the validation part is kept in memory.
    """
    horizon = config.horizon
    tau = torch.cat(
        [torch.full((horizon,), alpha / 2), torch.full((horizon,), 1 - alpha / 2)]
    ).view(1, 2 * horizon, 1)

    def pinball(prediction, y, mask):
        target = torch.cat([(y - mean) / std] * 2, dim=1)
        gap = target - prediction
        loss = torch.maximum(tau * gap, (tau - 1) * gap)
        return loss[torch.cat([mask, mask], dim=1)].mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config.scheduler_patience
    )
    best, best_state, stale, seconds = float("inf"), None, 0, 0.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        start = time.perf_counter()
        for x, y, mask in loaders["train"]:
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = pinball(model(x), y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        seconds += time.perf_counter() - start
        model.eval()
        with torch.no_grad():
            total, count = 0.0, 0
            for x, y, mask in loaders["val"]:
                total += float(pinball(model(x), y, mask)) * len(x)
                count += len(x)
        validation = total / count
        scheduler.step(validation)
        print(f"{label} epoch={epoch} val_pinball={validation:.5f}", flush=True)
        if validation < best:
            best, stale = validation, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if config.patience is not None and stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return {"epochs_run": epoch, "train_seconds": seconds, "val_pinball": best}


def gaussian_intervals(prediction, sigma, alpha):
    z = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    return prediction - z * sigma[None], prediction + z * sigma[None]


def residual_std(residuals, mask):
    """Per (horizon, sensor) standard deviation of the residuals over observed targets."""
    count = np.maximum(mask.sum(axis=0), 1)
    return np.sqrt((np.where(mask, residuals, 0.0) ** 2).sum(axis=0) / count)


def build_intervals(name, alpha, cfg, test, cal, sigma_train, days, band=None):
    """Lower and upper bounds on the test windows for one method and level."""
    cal_res = C.abs_residuals(cal["pred"], cal["target"])
    if name == "gaussian_train":
        return gaussian_intervals(test["pred"], sigma_train, alpha)
    if name == "gaussian_cal":
        return gaussian_intervals(
            test["pred"], residual_std(cal_res, cal["mask"]), alpha
        )
    if name == "conformal_pooled":
        return C.symmetric(
            test["pred"], C.pooled_quantiles(cal_res, cal["mask"], alpha)
        )
    if name == "conformal_sensor":
        return C.symmetric(
            test["pred"], C.sensor_quantiles(cal_res, cal["mask"], alpha)
        )
    if name == "conformal_aci":
        return C.adaptive_intervals(
            test["pred"],
            test["target"],
            test["mask"],
            cal_res,
            cal["mask"],
            alpha,
            cfg.gamma,
        )
    if name == "conformal_rolling":
        return C.rolling_intervals(
            test["pred"],
            test["target"],
            test["mask"],
            days,
            cal_res,
            cal["mask"],
            alpha,
            span_days=cfg.rolling_days,
        )
    lower_cal, upper_cal, lower, upper = band
    if name == "quantile_raw":
        return lower, upper
    if name == "cqr":
        scores = C.cqr_scores(lower_cal, upper_cal, cal["target"])
        return C.cqr_intervals(
            lower, upper, C.pooled_quantiles(scores, cal["mask"], alpha)
        )
    raise ValueError(name)


def groups_for(test, dataset, cfg):
    """Subsets of the test targets, defined only from what is known at forecast time
    (plus one outcome-conditioned diagnostic, which is labelled as such)."""
    shape = test["target"].shape
    congested = np.broadcast_to((test["last"] < cfg.congestion_mph)[:, None, :], shape)
    times = dataset.times[dataset.history : dataset.history + len(dataset)]
    hour, weekday = times.hour.to_numpy(), times.dayofweek.to_numpy() < 5
    rush = weekday & (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19)))
    return {
        "origin_congested": congested,
        "origin_free_flow": ~congested,
        "rush_hour": np.broadcast_to(rush[:, None, None], shape),
        "off_peak": np.broadcast_to(~rush[:, None, None], shape),
        "target_congested_diagnostic": test["target"] < cfg.congestion_mph,
    }


def run_seed(cfg, data, graph_dir, seed, out):
    bench = cfg.training()
    seed_all(seed, bench.threads)
    timed = split_name(cfg.model)[1]
    datasets, scaler, weights, _ = prepared(
        bench, data, graph_dir, timed, cfg.calibration
    )
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    generator = torch.Generator().manual_seed(seed)
    loaders = make_loaders(datasets, bench, generator)
    out.mkdir(parents=True, exist_ok=True)

    point = model_for(cfg.model, bench, weights)
    trained = fit(point, loaders, mean, std, bench, f"point seed={seed}", out)
    point.load_state_dict(
        torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    )
    cal = collect(point, datasets["cal"], mean, std)
    test = collect(point, datasets["test"], mean, std)
    train = collect(point, datasets["train"], mean, std)
    sigma_train = residual_std(
        C.abs_residuals(train["pred"], train["target"]), train["mask"]
    )
    days, _ = datasets["test"].origin_days()
    groups = groups_for(test, datasets["test"], cfg)

    # a second network for the quantile methods, at the primary level only
    seed_all(seed, bench.threads)
    doubled = BenchmarkConfig(**{**vars(bench), "horizon": 2 * bench.horizon})
    quantile = model_for(cfg.model, doubled, weights)
    q_trained = fit_quantile(
        quantile, loaders, mean, std, bench, cfg.alpha, f"quantile seed={seed}"
    )
    q_cal = collect(quantile, datasets["cal"], mean, std)["pred"]
    q_test = collect(quantile, datasets["test"], mean, std)["pred"]
    horizon = bench.horizon

    def band(raw):
        low, high = raw[:, :horizon], raw[:, horizon:]
        return np.minimum(low, high), np.maximum(low, high)

    cal_low, cal_high = band(q_cal)
    test_low, test_high = band(q_test)
    pack = (cal_low, cal_high, test_low, test_high)

    methods, reliability = {}, {}
    for name in POINT_METHODS + QUANTILE_METHODS:
        for alpha in cfg.reliability_alphas:
            if name in QUANTILE_METHODS and alpha != cfg.alpha:
                continue
            start = time.perf_counter()
            low, high = build_intervals(
                name, alpha, cfg, test, cal, sigma_train, days, pack
            )
            seconds = time.perf_counter() - start
            if alpha == cfg.alpha:
                methods[name] = C.evaluate_intervals(
                    low, high, test["target"], test["mask"], alpha, groups, days
                )
                methods[name]["build_seconds"] = seconds
            keep = test["mask"]
            reliability.setdefault(name, {})[str(alpha)] = {
                "coverage": float(
                    ((test["target"] >= low) & (test["target"] <= high))[keep].mean()
                ),
                "width": float((high - low)[keep].mean()),
            }
        print(
            f"seed {seed} {name}: coverage {methods[name]['overall']['coverage']:.3f} "
            f"width {methods[name]['overall']['width']:.2f} mph",
            flush=True,
        )
    # how much the ACI step size matters; the main table keeps the preset gamma
    sweep = {}
    for gamma in cfg.gamma_sweep:
        low, high = C.adaptive_intervals(
            test["pred"],
            test["target"],
            test["mask"],
            C.abs_residuals(cal["pred"], cal["target"]),
            cal["mask"],
            cfg.alpha,
            gamma,
        )
        scored = C.evaluate_intervals(
            low, high, test["target"], test["mask"], cfg.alpha, groups
        )
        sweep[str(gamma)] = {
            key: scored[key]
            for key in ("overall", "origin_congested", "origin_free_flow")
        }
    keep = test["mask"]
    result = {
        "seed": seed,
        "point_test_mae": float(np.abs(test["pred"] - test["target"])[keep].mean()),
        "point_epochs": len(trained["history"]),
        "point_train_seconds": trained["train_seconds"],
        "quantile": q_trained,
        "calibration_windows": len(cal["pred"]),
        "calibration_targets": int(cal["mask"].sum()),
        "test_targets": int(keep.sum()),
        "methods": methods,
        "reliability": reliability,
        "aci_sweep": sweep,
    }
    write_json(out / "result.json", result)
    return result


def average(trees):
    """Mean and SD over seeds of every number in identical nested dicts and lists."""
    first = trees[0]
    if isinstance(first, dict):
        pairs = {k: average([t[k] for t in trees]) for k in first}
        return {k: v[0] for k, v in pairs.items()}, {k: v[1] for k, v in pairs.items()}
    if isinstance(first, list):
        pairs = [average([t[i] for t in trees]) for i in range(len(first))]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if first is None or isinstance(first, str) or any(t is None for t in trees):
        return first, None
    values = np.asarray(trees, dtype=np.float64)
    return float(values.mean()), (
        float(values.std(ddof=1)) if len(values) > 1 else None
    )


def run(cfg, data, graph_dir, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json",
        {**vars(cfg), **git_state()},
    )
    results = []
    for seed in cfg.seeds:
        folder = out / f"seed-{seed}"
        done = folder / "result.json"
        results.append(
            json.loads(done.read_text())
            if done.exists()
            else run_seed(cfg, data, graph_dir, seed, folder)
        )
    mean, sd = average(
        [
            {
                "methods": r["methods"],
                "reliability": r["reliability"],
                "aci_sweep": r["aci_sweep"],
            }
            for r in results
        ]
    )
    summary = {
        "alpha": cfg.alpha,
        "seeds": cfg.seeds,
        "model": cfg.model,
        "mean": mean,
        "sd": sd,
        "point_test_mae": average([r["point_test_mae"] for r in results]),
        "calibration_targets": results[0]["calibration_targets"],
        "test_targets": results[0]["test_targets"],
        **git_state(),
    }
    write_json(out / "summary.json", summary)
    return summary


LABELS = {
    "gaussian_train": "Gaussian (train residuals)",
    "gaussian_cal": "Gaussian (calibration residuals)",
    "conformal_pooled": "Split conformal",
    "conformal_sensor": "Split conformal, per sensor",
    "conformal_aci": "Adaptive conformal (ACI)",
    "conformal_rolling": "Rolling conformal (7 days)",
    "quantile_raw": "Quantile network",
    "cqr": "Conformalized quantile (CQR)",
}
COLORS = {
    "gaussian_train": "#667085",
    "gaussian_cal": "#a16207",
    "conformal_pooled": "#1d4ed8",
    "conformal_sensor": "#9333ea",
    "conformal_aci": "#087e8b",
    "conformal_rolling": "#16a34a",
    "quantile_raw": "#d97706",
    "cqr": "#dc2626",
}


def figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    # saved summaries have sorted keys, so put the methods back in a fixed order
    methods = {
        n: summary["mean"]["methods"][n]
        for n in LABELS
        if n in summary["mean"]["methods"]
    }
    reliability = {
        n: summary["mean"]["reliability"][n]
        for n in LABELS
        if n in summary["mean"]["reliability"]
    }
    alpha = summary["alpha"]

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax.plot([0.4, 1], [0.4, 1], color="#999999", linewidth=1, linestyle=":")
    for name, by_alpha in reliability.items():
        pairs = sorted((1 - float(a), v["coverage"]) for a, v in by_alpha.items())
        if len(pairs) == 1:
            ax.scatter(
                *pairs[0], color=COLORS[name], label=LABELS[name], marker="D", s=30
            )
        else:
            ax.plot(
                *zip(*pairs),
                marker="o",
                markersize=4,
                color=COLORS[name],
                label=LABELS[name],
            )
    ax.set(
        title="Reliability",
        xlabel="Nominal coverage",
        ylabel="Empirical coverage on test",
    )
    ax.legend(fontsize=7)
    fig.savefig(out / "uncertainty-reliability.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    minutes = [5 * (h + 1) for h in range(len(methods["cqr"]["per_horizon"]))]
    for name, entry in methods.items():
        axes[0].plot(
            minutes,
            [h["coverage"] for h in entry["per_horizon"]],
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[1].plot(
            minutes, [h["width"] for h in entry["per_horizon"]], color=COLORS[name]
        )
    axes[0].axhline(1 - alpha, color="#999999", linestyle=":")
    axes[0].set(
        title=f"Coverage by horizon (target {1 - alpha:.0%})",
        xlabel="Forecast horizon (minutes)",
        ylabel="Empirical coverage",
    )
    axes[1].set(
        title="Interval width by horizon",
        xlabel="Forecast horizon (minutes)",
        ylabel="Width (mph)",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    fig.savefig(out / "uncertainty-horizon.png", dpi=170)
    plt.close(fig)

    names = list(methods)
    groups = [
        ("origin_free_flow", "Free flow\n(last speed >= 40)"),
        ("origin_congested", "Congested\n(last speed < 40)"),
        ("off_peak", "Off peak"),
        ("rush_hour", "Weekday rush hour"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [methods[name][key]["coverage"] for key, _ in groups]
        ax.bar(
            [x + i * width for x in range(len(groups))],
            values,
            width,
            color=COLORS[name],
            label=LABELS[name],
        )
    ax.axhline(1 - alpha, color="#444444", linestyle=":")
    ax.set(
        ylim=(0.5, 1.13),
        yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ylabel="Empirical coverage",
        title="Coverage by traffic condition at the forecast origin",
    )
    ax.set_xticks(
        [x + 0.4 - width / 2 for x in range(len(groups))],
        [label for _, label in groups],
    )
    ax.legend(fontsize=7, ncol=4, loc="upper center")
    fig.savefig(out / "uncertainty-conditions.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    for name in (
        "gaussian_train",
        "conformal_pooled",
        "conformal_aci",
        "conformal_rolling",
    ):
        days = methods[name]["per_day"]
        ax.plot(
            range(1, len(days) + 1),
            [d["coverage"] for d in days],
            marker="o",
            markersize=3,
            color=COLORS[name],
            label=LABELS[name],
        )
    ax.axhline(1 - alpha, color="#999999", linestyle=":")
    ax.set(
        title="Coverage on each test day",
        xlabel="Test day",
        ylabel="Empirical coverage",
        ylim=(0.6, 1.0),
    )
    ax.legend(fontsize=8, ncol=4)
    fig.savefig(out / "uncertainty-daily.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prediction interval experiment")
    parser.add_argument("--config", default="configs/uncertainty.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--topology", default="data/topology")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = run(load_config(args.config), args.data, args.topology, args.out)
    figures(summary, args.out)
    for name, m in summary["mean"]["methods"].items():
        print(
            f"{name:18s} coverage {m['overall']['coverage']:.3f}  width {m['overall']['width']:.2f}"
        )


if __name__ == "__main__":
    main()
