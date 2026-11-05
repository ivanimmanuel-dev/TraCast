"""Run the same benchmark on nested sensor subsets and put the results side by side.

The subsets are prefixes of one graph ordering (topology.order), so the 16 sensors
sit inside the 32, and so on up to all 207. Every model gets the same number of
updates at every size. Two comparisons come out of it:

* own population: each model scored on all the sensors it was trained on. The
  sensor set changes with the size, so these scores are not comparable across sizes.
* common core: every model scored on the first 16 sensors only. The targets are the
  same for every size, so this is the fair way to ask whether more sensors help.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import platform
import sys

import numpy as np
from importlib.metadata import version

from .benchmark import BenchmarkConfig, run as run_benchmark
from .experiment import write_json
from .provenance import git_state
from .stats import day_bootstrap

CORE = 16
# (first, second): the estimate is first minus second, scored on identical windows
PAIRS = [
    ("sgraph_time", "sgru_time"),
    ("sgraph", "sgru"),
    ("slstm_time", "sgru_time"),
    ("sgru_id_time", "sgru_time"),
    ("sgraph_id_time", "sgraph_time"),
    ("sgraph_id_time", "sgru_id_time"),
    ("sgru_time", "gru_time"),
    ("sgraph_time", "graph_gru_time"),
    ("graph_gru_time", "gru_time"),
    ("sgraph_time", "ha"),
    ("gru_time", "ha"),
]


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = raw.pop("sensor_counts")
    if (
        not counts
        or sorted(set(counts)) != counts
        or counts[0] < CORE
        or counts[-1] > 207
    ):
        raise ValueError("sensor_counts must increase from at least 16 up to 207")
    return counts, BenchmarkConfig(**raw)


def run(counts, config, data, graph_dir, out):
    """Run every size in turn; sizes that already finished are skipped."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    git = git_state()
    write_json(
        out / "scaling-config.json",
        {"sensor_counts": counts, "benchmark": asdict(config), **git},
    )
    for count in counts:
        folder = out / f"n{count:03d}"
        status = folder / "status.json"
        if status.exists() and json.loads(status.read_text())["status"] == "complete":
            print(f"{count} sensors: already done", flush=True)
            continue
        if folder.exists():
            folder.rename(
                out / f"n{count:03d}.unfinished-{len(list(out.glob('*.unfinished-*')))}"
            )
        sized = replace(
            config,
            sensors=count,
            core_sizes=[k for k in (CORE, 32, 64, 128) if k < count],
        )
        print(f"=== {count} sensors ===", flush=True)
        run_benchmark(sized, data, graph_dir, folder)
    return summarize(out)


def _stats(values):
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def summarize(out):
    """Collect every size's results into one table: model x sensor count."""
    out = Path(out)
    table, comparisons = {}, {}
    for folder in sorted(out.glob("n[0-9][0-9][0-9]")):
        if json.loads((folder / "status.json").read_text())["status"] != "complete":
            continue
        count = int(folder.name[1:])
        trials = json.loads((folder / "results.json").read_text())
        daily = {
            model: [r["daily"] for r in trials if r["model"] == model]
            for model in dict.fromkeys(r["model"] for r in trials)
            if all("daily" in r for r in trials if r["model"] == model)
        }
        found = {
            f"{a}_vs_{b}": day_bootstrap(daily[a], daily[b])
            for a, b in PAIRS
            if a in daily and b in daily
        }
        if found:
            comparisons[str(count)] = found
        for model in dict.fromkeys(
            r["model"] for r in json.loads((folder / "results.json").read_text())
        ):
            runs = [
                r
                for r in json.loads((folder / "results.json").read_text())
                if r["model"] == model
            ]
            horizons = {
                h["minutes"]: _stats([r["test"]["per_horizon"][i]["mae"] for r in runs])
                for i, h in enumerate(runs[0]["test"]["per_horizon"])
            }
            core = (
                _stats([r["test"]["mae"] for r in runs])
                if count == CORE
                else _stats([r["test"]["core"][str(CORE)]["mae"] for r in runs])
            )
            table.setdefault(model, {})[str(count)] = {
                "mae": _stats([r["test"]["mae"] for r in runs]),
                "rmse": _stats([r["test"]["rmse"] for r in runs]),
                "core16_mae": core,
                "core16_horizons": {
                    str(m): _stats(
                        [
                            (
                                r["test"]
                                if count == CORE
                                else r["test"]["core"][str(CORE)]
                            )["per_horizon"][i]["mae"]
                            for r in runs
                        ]
                    )
                    for i, m in enumerate(horizons)
                },
                "horizon_mae": {str(m): v for m, v in horizons.items()},
                "parameters": runs[0]["parameter_count"],
                "train_seconds": _stats([r["train_seconds"] for r in runs]),
                "seconds_per_epoch": (
                    _stats([r["train_seconds"] / len(r["history"]) for r in runs])
                    if runs[0]["history"]
                    else None
                ),
                "epochs_run": _stats([len(r["history"]) for r in runs])
                if runs[0]["history"]
                else None,
                "inference_ms_per_window": _stats(
                    [r["inference"]["median_ms_per_window"] for r in runs]
                ),
                "peak_memory_mib": _stats(
                    [r["peak_memory"]["bytes"] / 1024**2 for r in runs]
                ),
                "seeds": [r["seed"] for r in runs],
            }
    summary = {
        "sizes": sorted({int(c) for m in table.values() for c in m}),
        "models": table,
        "day_bootstrap": comparisons,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "numpy": version("numpy"),
        },
        "day_bootstrap_note": (
            "Gaps in own-population MAE (first minus second, negative favors the first) "
            "with 95% intervals from resampling test days; training seeds averaged first."
        ),
        "note": (
            "mae is on every sensor a model was trained on, so it is not comparable "
            "across sizes; core16_mae scores every model on the same first 16 sensors."
        ),
        **git_state(),
    }
    write_json(out / "scaling-summary.json", summary)
    return summary


def figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "persistence": "Persistence",
        "ha": "Historical average",
        "linear_time": "Linear + calendar",
        "gru_time": "Dense GRU + calendar",
        "graph_gru_time": "Dense Graph-GRU + calendar",
        "sgru_time": "Shared GRU + calendar",
        "sgraph_time": "Shared Graph-GRU + calendar",
        "slstm_time": "Shared LSTM + calendar",
        "sgru_id_time": "Shared GRU + sensor vector",
        "sgraph_id_time": "Shared Graph-GRU + sensor vector",
    }
    styles = {
        "persistence": ("#667085", ":"),
        "ha": ("#a16207", ":"),
        "linear_time": ("#16a34a", "-"),
        "gru_time": ("#1d4ed8", "-"),
        "graph_gru_time": ("#9333ea", "-"),
        "sgru_time": ("#d97706", "--"),
        "sgraph_time": ("#087e8b", "--"),
        "slstm_time": ("#be123c", "--"),
        "sgru_id_time": ("#a16207", "-."),
        "sgraph_id_time": ("#0f766e", "-."),
    }
    sizes = summary["sizes"]
    out = Path(out)

    def draw(ax, key, field, title, ylabel, log=False):
        for model, by_size in summary["models"].items():
            xs = [s for s in sizes if str(s) in by_size]
            ys = [_pick(by_size[str(s)], key, field) for s in xs]
            if any(y is None for y in ys):
                continue
            color, line = styles.get(model, ("#444444", "-"))
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                color=color,
                linestyle=line,
                label=labels.get(model, model),
            )
        ax.set(
            title=title,
            xlabel="Sensors used to train",
            ylabel=ylabel,
            xscale="log",
            xticks=sizes,
            xticklabels=sizes,
        )
        if log:
            ax.set_yscale("log")
        ax.minorticks_off()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    draw(
        axes[0],
        "mae",
        "mean",
        "Own population (not comparable across sizes)",
        "Test MAE (mph)",
    )
    draw(
        axes[1],
        "core16_mae",
        "mean",
        "Same 16 target sensors for every size",
        "Test MAE on the first 16 sensors (mph)",
    )
    fig.legend(
        *axes[1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    fig.savefig(out / "scaling-accuracy.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
    draw(
        axes[0],
        "seconds_per_epoch",
        "mean",
        "Training time per epoch",
        "seconds",
        log=True,
    )
    draw(axes[1], "peak_memory_mib", "mean", "Peak process memory", "MiB")
    draw(axes[2], "parameters", None, "Parameters", "count", log=True)
    draw(
        axes[3],
        "inference_ms_per_window",
        "mean",
        "Inference",
        "ms per window",
        log=True,
    )
    fig.legend(
        *axes[3].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    fig.savefig(out / "scaling-cost.png", dpi=170)
    plt.close(fig)


def subset_map(graph_dir, out, counts=(16, 32, 64, 128, 207)):
    """Where the nested subsets sit: every sensor coloured by the first size that has it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from . import topology

    graph_dir = Path(graph_dir)
    ids = (graph_dir / "graph_sensor_ids.txt").read_text().strip().split(",")
    distances = pd.read_csv(
        graph_dir / "distances_la_2012.csv", dtype={"from": str, "to": str}
    )
    weights, _ = topology.adjacency(distances, ids)
    ranking = topology.order(weights, ids)
    first = {}
    for rank, node in enumerate(ranking):
        first[node] = next(c for c in counts if rank < c)
    places = pd.read_csv(
        graph_dir / "graph_sensor_locations.csv", dtype={"sensor_id": str}
    )
    places = places.set_index("sensor_id").loc[ids]
    palette = {
        16: "#dc2626",
        32: "#d97706",
        64: "#16a34a",
        128: "#1d4ed8",
        207: "#9ca3af",
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for count in reversed(counts):
        nodes = [n for n in range(len(ids)) if first[n] == count]
        label = f"first {count}" if count == counts[0] else f"added at {count}"
        ax.scatter(
            places["longitude"].iloc[nodes],
            places["latitude"].iloc[nodes],
            s=26 if count == counts[0] else 14,
            color=palette[count],
            label=label,
            zorder=3,
        )
    ax.set(title="Nested sensor subsets", xlabel="Longitude", ylabel="Latitude")
    ax.legend(fontsize=8)
    fig.savefig(Path(out) / "scaling-subsets.png", dpi=170)
    plt.close(fig)


def _pick(entry, key, field):
    value = entry[key]
    if value is None:
        return None
    return value if field is None else value[field]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scaling study over nested sensor subsets"
    )
    parser.add_argument("--config", default="configs/scaling.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--topology", default="data/topology")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    counts, config = load_config(args.config)
    if args.summarize_only:
        summary = summarize(args.out)
    else:
        summary = run(counts, config, args.data, args.topology, args.out)
    figures(summary, args.out)
    subset_map(args.topology, args.out)
    print(json.dumps({m: sorted(v) for m, v in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
