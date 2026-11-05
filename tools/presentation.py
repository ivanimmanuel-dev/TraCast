"""Render the README and report figures from experiment summaries."""

from pathlib import Path
import json
import hashlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "persistence": "Persistence",
    "gru": "GRU",
    "lstm": "LSTM",
    "graph_gru": "Graph-GRU",
    "graph_self": "Graph-GRU (self only)",
}
ORDER = list(LABELS)
COLORS = {
    "persistence": "#667085",
    "gru": "#1d4ed8",
    "lstm": "#d97706",
    "graph_gru": "#087e8b",
    "graph_self": "#9333ea",
}


EXTENDED_LABELS = {
    "persistence": "Persistence",
    "ha": "Historical average",
    "linear": "Linear",
    "linear_time": "Linear + calendar",
    "gru": "GRU",
    "gru_time": "GRU + calendar",
    "lstm": "LSTM",
    "graph_gru": "Graph-GRU",
    "graph_gru_time": "Graph-GRU + calendar",
    "graph_self": "Graph-GRU (self only)",
    "graph_self_time": "Graph-GRU (self only) + calendar",
}
EXTENDED_COLORS = {
    "persistence": "#667085",
    "ha": "#a16207",
    "linear_time": "#16a34a",
    "gru": "#1d4ed8",
    "gru_time": "#0b2a7a",
    "graph_gru_time": "#087e8b",
}
# (key in summary["day_bootstrap"], label); each estimate is first minus second.
COMPARISONS = [
    ("gru_time_vs_gru", "GRU + calendar - GRU"),
    ("linear_time_vs_linear", "Linear + calendar - linear"),
    ("graph_gru_time_vs_graph_gru", "Graph-GRU + calendar - Graph-GRU"),
    ("lstm_vs_gru", "LSTM - GRU"),
    ("graph_gru_vs_graph_self", "Graph-GRU - self only"),
    (
        "graph_gru_time_vs_graph_self_time",
        "Graph-GRU - self only, both with calendar",
    ),
    ("graph_gru_time_vs_gru_time", "Graph-GRU - GRU, both with calendar"),
    ("gru_vs_ha", "GRU - historical average"),
    ("gru_time_vs_ha", "GRU + calendar - historical average"),
    ("ha_vs_persistence", "Historical average - persistence"),
]


def mean_sd(item):
    return f"{item['mean']:.3f}" + (
        f" ({item['sd']:.3f})" if item["sd"] is not None else ""
    )


def phrase(comparison):
    low, high = comparison["ci95"]
    return f"{comparison['estimate']:+.3f} mph (95% CI {low:+.3f} to {high:+.3f})"


def separated(comparison):
    low, high = comparison["ci95"]
    return low > 0 or high < 0


def extended_section(v1_models):
    """Markdown for the longer-training study, or ``None`` before it has run."""
    folder = ROOT / "reports/benchmark-extended"
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    models, found = summary["models"], summary["day_bootstrap"]
    present = [name for name in EXTENDED_LABELS if name in models]

    def at(entry, minutes):
        row = next(r for r in entry["horizons"] if r["minutes"] == minutes)
        return f"{row['mae']['mean']:.3f}"

    rows = "\n".join(
        f"| {EXTENDED_LABELS[name]} | {mean_sd(models[name]['mae'])} "
        f"| {models[name]['test_mae_ci95_days'][0]:.3f}–{models[name]['test_mae_ci95_days'][1]:.3f} "
        f"| {at(models[name], 15)} | {at(models[name], 30)} | {at(models[name], 60)} "
        f"| {mean_sd(models[name]['rmse'])} | {models[name]['parameter_count']:,} "
        f"| {models[name]['epochs_run']['mean']:.0f} |"
        if "epochs_run" in models[name]
        else f"| {EXTENDED_LABELS[name]} | {models[name]['mae']['mean']:.3f} "
        f"| {models[name]['test_mae_ci95_days'][0]:.3f}–{models[name]['test_mae_ci95_days'][1]:.3f} "
        f"| {at(models[name], 15)} | {at(models[name], 30)} | {at(models[name], 60)} "
        f"| {models[name]['rmse']['mean']:.3f} | 0 | – |"
        for name in present
    )
    gaps = "\n".join(
        f"| {label} | {found[key]['estimate']:+.3f} "
        f"| {found[key]['ci95'][0]:+.3f} to {found[key]['ci95'][1]:+.3f} |"
        for key, label in COMPARISONS
        if key in found
    )
    days = next(iter(found.values()))["days"]
    seeds = ", ".join(map(str, config["seeds"]))
    learned = [name for name in present if "epochs_run" in models[name]]
    ranked = sorted(learned, key=lambda name: models[name]["mae"]["mean"])
    best, top = ranked[0], ranked[:3]
    spread = models[top[-1]]["mae"]["mean"] - models[best]["mae"]["mean"]
    names = " and ".join(EXTENDED_LABELS[name] for name in top[1:])
    sentences = [
        f"{EXTENDED_LABELS[best]} is lowest at {models[best]['mae']['mean']:.3f} mph "
        f"(persistence {models['persistence']['mae']['mean']:.3f}, historical average "
        f"{models['ha']['mae']['mean']:.3f}), but {names} are within {spread:.3f} mph "
        f"of it, so I can't rank them."
    ]
    gained = None
    if "gru" in models and "gru" in v1_models:
        gained = v1_models["gru"]["mae"]["mean"] - models["gru"]["mae"]["mean"]
        sentences.append(
            f"Training longer helped: the GRU went from {v1_models['gru']['mae']['mean']:.3f} "
            f"at five epochs to {models['gru']['mae']['mean']:.3f}."
        )
    graph = [
        key
        for key in ("graph_gru_vs_graph_self", "graph_gru_time_vs_graph_self_time")
        if key in found
    ]
    if "gru_time_vs_gru" in found:
        gap = found["gru_time_vs_gru"]
        others = [abs(found[key]["estimate"]) for key in graph]
        others += [gained] if gained else []
        if not separated(gap):
            sentences.append(
                f"Calendar inputs change the GRU by {phrase(gap)}, which the test days "
                f"can't tell apart from zero."
            )
        elif gap["estimate"] > 0:
            sentences.append(f"Calendar inputs hurt the GRU by {phrase(gap)}.")
        else:
            biggest = all(abs(gap["estimate"]) > other for other in others)
            lead = "gave the biggest jump" if biggest else "helped"
            sentences.append(
                f"Calendar inputs {lead}: the GRU changes by {phrase(gap)}."
            )
    if graph:
        parts = [
            f"{phrase(found[key])} {'with' if key.endswith('_time') else 'without'} "
            f"calendar inputs"
            for key in graph
        ]
        apart = [key for key in graph if separated(found[key])]
        opposite = len({found[key]["estimate"] > 0 for key in graph}) == 2
        if not apart:
            verdict = "Neither gap is clearly different from zero."
        elif len(apart) == len(graph) and opposite:
            verdict = "The sign flips, so the graph doesn't give a reliable gain."
        else:
            verdict = "At least one of the two is clearly different from zero."
        sentences.append(
            "Graph-GRU minus its self-only version (the same model with the neighbor "
            "averages zeroed) is " + " and ".join(parts) + f". {verdict}"
        )
    narrative = " ".join(sentences)
    return rf"""### Longer training, calendar inputs, baselines

The five-epoch run below stops while the models are still improving, so here each one trains for up to {config["epochs"]} epochs, stops after {config["patience"]} epochs with no validation gain, and halves the learning rate after {config["scheduler_patience"]}. I also added calendar inputs (time of day as sine and cosine plus a weekend flag, marked "+ calendar"), a historical-average baseline (the training mean for each sensor, weekday, and 5-minute slot), and a linear model. {len(config["seeds"])} seeds: {seeds}.

| Model | MAE (SD) | MAE range over test days | 15 min | 30 min | 60 min | RMSE (SD) | Parameters | Epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

MAE and RMSE are test means in mph, and the 15/30/60 min columns are MAE at those horizons. The range column resamples the {days} test days, so it shows how much a score depends on which days ended up in the test set. Persistence and the historical average aren't trained, so they run once.

{narrative}

| Difference in MAE (first - second) | Estimate (mph) | 95% CI over test days |
|---|---:|---:|
{gaps}

Negative means the first model is better. Both models in a row are scored on the same windows and the same resampled days. These intervals are about which days were held out; the seed SDs in the table above cover random initialization.

![Test error by forecast horizon, longer-training study; shading is seed SD](reports/figures/extended-horizons.png)

[Per-seed metrics and timings](reports/benchmark-extended/results.json) · [Protocol and comparisons](reports/benchmark-extended/report.md)
"""


EXTENDED_REPRODUCE = r"""
## Reproduce

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark-extended.json --out runs/benchmark-extended
```

This writes one folder per model and seed, then `results.json`, `summary.json` and the plots. `summary.json` has the seed stats plus a `day_bootstrap` section with the intervals above (every trial saves per-day error sums, which is all the bootstrap needs).

Timings are rough: the machine was doing other things during part of the run, so don't compare them with the five-epoch study. The errors aren't affected. Repeating the GRU seed-7 trial under a different load gave exactly the same validation errors.

The manifest says `git_dirty` because the run used code that wasn't committed yet. After the 47 trials finished, the convergence plot crashed on early-stopped runs of different lengths, so I fixed `benchmark.plots` and redrew the plots from `results.json`. No trial or metric changed.

Model names ending in `_time` get the calendar inputs. `ha` is fit on the training split only, and `linear` is one linear layer over the flattened window.
"""


def write_extended_report(extended):
    folder = ROOT / "reports/benchmark-extended"
    report = extended.replace(
        "### Longer training, calendar inputs, baselines",
        "# Extended benchmark: longer training, calendar inputs, baselines",
        1,
    )
    report = report.replace(
        "The five-epoch run below",
        "The [five-epoch run](../benchmark/report.md)",
        1,
    )
    report = report.replace("(reports/figures/", "(../figures/")
    report = report.replace("(reports/benchmark-extended/", "(")
    (folder / "report.md").write_bytes((report + EXTENDED_REPRODUCE).encode())


def extended_figure(out):
    folder = ROOT / "reports/benchmark-extended"
    if not (folder / "summary.json").exists():
        return
    models = json.loads((folder / "summary.json").read_text())["models"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, color in EXTENDED_COLORS.items():
        if name not in models:
            continue
        entry = models[name]
        for metric_name, ax in zip(("mae", "rmse"), axes):
            x = [r["minutes"] for r in entry["horizons"]]
            y = np.array([r[metric_name]["mean"] for r in entry["horizons"]])
            sd = np.array([r[metric_name]["sd"] or 0 for r in entry["horizons"]])
            ax.plot(
                x,
                y,
                label=EXTENDED_LABELS[name],
                color=color,
                linestyle=":" if name in ("persistence", "ha") else "-",
                linewidth=1.7,
            )
            if entry["n"] > 1:
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10)
            ax.set(
                title=f"Test {metric_name.upper()}",
                xlabel="Forecast horizon (minutes)",
                ylabel="mph",
            )
    axes[0].legend(loc="upper left", framealpha=0.95)
    fig.savefig(out / "extended-horizons.png", dpi=180)
    plt.close(fig)


def main():
    source = ROOT / "reports/benchmark/summary.json"
    summary = json.loads(source.read_text())
    models = summary["models"]
    out = ROOT / "reports/figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name in ORDER:
        entry = models[name]
        for metric, ax in zip(("mae", "rmse"), axes):
            x = [r["minutes"] for r in entry["horizons"]]
            y = np.array([r[metric]["mean"] for r in entry["horizons"]])
            sd = np.array([r[metric]["sd"] or 0 for r in entry["horizons"]])
            ax.plot(
                x,
                y,
                label=LABELS[name],
                color=COLORS[name],
                linestyle=(
                    "--"
                    if name == "graph_self"
                    else ":"
                    if name == "persistence"
                    else "-"
                ),
                linewidth=1.7,
            )
            if entry["n"] > 1:
                ax.fill_between(x, y - sd, y + sd, color=COLORS[name], alpha=0.10)
            ax.set(
                title=f"Test {metric.upper()}",
                xlabel="Forecast horizon (minutes)",
                ylabel="mph",
            )
    axes[0].legend(loc="upper left", framealpha=0.95)
    fig.savefig(out / "forecast-horizons.png", dpi=180)
    plt.close(fig)
    records = json.loads((ROOT / "reports/benchmark/results.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    for name in ORDER[1:]:
        values = np.array(
            [
                [h["validation"]["mae"] for h in r["history"]]
                for r in records
                if r["model"] == name
            ]
        )
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.plot(
            range(1, values.shape[1] + 1),
            mean,
            label=LABELS[name],
            color=COLORS[name],
            linestyle="--" if name == "graph_self" else "-",
        )
        ax.fill_between(
            range(1, values.shape[1] + 1),
            mean - sd,
            mean + sd,
            color=COLORS[name],
            alpha=0.10,
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation MAE (mph)",
        title="Validation convergence: mean and seed SD",
        xticks=range(1, values.shape[1] + 1),
    )
    ax.legend()
    fig.savefig(out / "validation-convergence.png", dpi=170)
    plt.close(fig)
    provenance = {
        "source": "reports/benchmark/summary.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "uncertainty": "Sample SD across three training seeds",
    }
    (out / "provenance.json").write_bytes(
        (json.dumps(provenance, indent=2) + "\n").encode()
    )

    extended_figure(out)
    table = "\n".join(
        f"| {LABELS[name]} | {mean_sd(models[name]['mae'])} | {mean_sd(models[name]['rmse'])} | {models[name]['parameter_count']:,} |"
        for name in ORDER
    )
    conditions = json.loads(
        (ROOT / "reports/federated-study/summary.json").read_text()
    )["conditions"]
    local = all("local_only_mae" in c for c in conditions.values())
    federated_rows = "\n".join(
        f"| {'Balanced' if c['partition_rule'] == 'balanced' else 'Congestion-grouped'} | {c['round_budget']} | {c['fedavg_mae']['mean']:.3f} | {c['centralized_mae']['mean']:.3f} | "
        + (f"{c['local_only_mae']['mean']:.3f} | " if local else "")
        + f"{c['parameter_bytes']:,} B |"
        for c in conditions.values()
    )
    federated_header = (
        "| Sensor grouping | Rounds | Federated MAE | Centralized MAE | "
        + ("Local-only MAE | " if local else "")
        + "Model weights transferred |\n|---|---:|---:|---:|"
        + ("---:|" if local else "")
        + "---:|"
    )
    balanced = sorted(
        (c for c in conditions.values() if c["partition_rule"] == "balanced"),
        key=lambda c: c["round_budget"],
    )
    shortest, longest = balanced[0], balanced[-1]
    drop = shortest["fedavg_mae"]["mean"] - longest["fedavg_mae"]["mean"]
    ratio = longest["parameter_bytes"] / shortest["parameter_bytes"]
    grouped = [c for c in conditions.values() if c["partition_rule"] == "congestion"]
    grouped_note = (
        f" With congestion-grouped clients it is {grouped[-1]['fedavg_mae']['mean']:.3f} against {grouped[-1]['local_only_mae']['mean']:.3f}."
        if local and grouped
        else ""
    )
    local_note = (
        f"\n\nAt {longest['round_budget']} rounds with balanced groups, federated averaging gets {longest['fedavg_mae']['mean']:.3f} mph, local-only {longest['local_only_mae']['mean']:.3f}, and centralized {longest['centralized_mae']['mean']:.3f}.{grouped_note} The gaps are about as big as the seed spread, so here training alone is roughly as good as federating."
        if local
        else ""
    )
    best_epochs = [
        (r["best_epoch"], max(h["epoch"] for h in r["history"]))
        for r in records
        if r["history"]
    ]
    at_last = sum(best == last for best, last in best_epochs)
    extended = extended_section(models)
    if extended:
        write_extended_report(extended)
    pointer = " The longer-training study above fixes that." if extended else ""
    extended_block = f"{extended}\n" if extended else ""
    text = rf"""# TraCast

TraCast predicts traffic speed 5 to 60 minutes ahead from the last hour of sensor readings. The data is METR-LA: 207 road sensors in Los Angeles, March to June 2012. Everything runs on a CPU laptop.

I use it to compare a few things: GRU vs LSTM vs a plain linear model, whether the road graph helps, whether knowing the time of day helps, and federated training vs centralized vs training each client alone.

[Research brief](reports/research-brief.pdf) · [Technical report](reports/research-report.md) · [Run guide](docs/reproducibility.md) · [Data sources](DATA.md) · [Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml)

## Results

Every model here forecasts 12 steps (60 minutes) from the previous 12, using 16 sensors picked from the DCRNN road-distance graph and the full March to June timeline. The first 70% is train, the next 10% validation, the last 20% test. MAE is mean absolute error and RMSE punishes big misses more; both are in mph.

{extended_block}### First pass: five epochs

My first run trained each model for five epochs with three seeds (7, 42, 2026). Persistence repeats the last reading, so it runs once. Numbers in parentheses are the standard deviation across seeds.

| Model | MAE (SD) | RMSE (SD) | Parameters |
|---|---:|---:|---:|
{table}

LSTM and Graph-GRU end up with about the same MAE, but the LSTM has fewer parameters. "Self only" is Graph-GRU with the neighbor averages set to zero (same architecture, same parameter count). Taking the averages out changes MAE by only {abs(models["graph_gru"]["mae"]["mean"] - models["graph_self"]["mae"]["mean"]):.4f} mph (it helped in one seed and hurt in two), and most of Graph-GRU's lead over the GRU stays. One catch: every model, the plain GRU too, already gets all 16 sensors through its input layer, so this tests the graph averages and not whether neighboring sensors matter.

Five epochs turned out to be too short. In {at_last} of {len(best_epochs)} runs the best validation epoch was the last one, so the models were still improving when training stopped.{pointer}

Training took {models["gru"]["train_seconds"]["mean"]:.1f} s for the GRU, {models["lstm"]["train_seconds"]["mean"]:.1f} s for the LSTM and {models["graph_gru"]["train_seconds"]["mean"]:.1f} s for Graph-GRU on a Ryzen 7 5825U, with about 320 MiB peak memory.

![Test error by forecast horizon; shading is seed SD](reports/figures/forecast-horizons.png)

[Per-seed metrics and timings](reports/benchmark/results.json) · [Benchmark protocol](reports/benchmark/report.md)

## Federated learning

The Flower simulation gives each of three clients its own set of sensors. Every round, each client trains its copy of the same GRU on its own readings, and the server averages the weights, weighted by how many training examples each client has (one example is one sensor's readings).

I compare it with two other setups. Centralized trains the same model on all the clients' data together, from the same starting weights, with the same normalization, masks and number of examples. Local-only trains each client by itself with the same amount of local work and no communication. All three clients run on one machine.

Twelve sensors, fourteen days, three seeds, one local epoch per round. Sensors are ranked by how often their training speeds fall below 40 mph. Balanced groups spread those ranks across the clients, and congestion-grouped clients get consecutive blocks.

{federated_header}
{federated_rows}{local_note}

Going from {shortest["round_budget"]} to {longest["round_budget"]} rounds cuts balanced MAE by {drop:.3f} mph and sends {ratio:g} times as many model-weight bytes. The byte counts cover weights sent for training, validation and the final test; message overhead isn't counted.

[Round convergence, RMSE, client counts, and partition controls](reports/federated-study/findings.md)

## Run

The recorded experiments ran on Python 3.11 and CPU PyTorch on Windows x64. CI runs the tests on Windows and Ubuntu. From the repo folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m tracast.cli fetch
.\.venv\Scripts\python.exe -m pytest -q -W error

# Seven-day GRU smoke run
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/gru-smoke

# Three-client Flower simulation: three rounds, one local epoch
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-smoke
```

The full model comparison, the longer-training study and the federated study:

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark-extended.json --out runs/benchmark-extended
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study
```

On Linux or WSL use `python3.11` and `.venv/bin/python`. The installed `tracast` command runs everything too, e.g. `tracast benchmark --config configs/benchmark-extended.json --out runs/benchmark-extended`, and `tracast --help` lists the subcommands.

The benchmark downloads the topology files itself. Use a fresh output folder for each run. More in the [run guide](docs/reproducibility.md).

## Pipeline

```mermaid
flowchart TD
    D[METR-LA speeds] --> V[Checksum and schema validation]
    T[DCRNN distance graph] --> G[Directed incoming/outgoing means]
    V --> S[Chronological 70/10/20 split]
    S --> N[Training-only scalers and causal fill]
    N --> W[12 input steps → 12 forecast steps]
    W --> R[Persistence / historical average / linear / GRU / LSTM]
    W --> G
    G --> H[Shared node projection → GRU]
    R --> E[Validation selection → test MAE/RMSE]
    H --> E
```

Windows never cross a split boundary. Normalization uses training data only, and zero or missing targets are left out of the loss and the metrics. The graph has 16 sensors and 85 directed edges from the published distance data; the sensors and edge weights come from distances, not from speeds. The saved model is the one with the best validation MAE, and test scores come after that.

## Reports

The [report index](reports/README.md) explains the folders, result files and provenance records.

| Experiment | Results |
|---|---|
| Recurrent encoders and graph ablation | [Methods](reports/benchmark/report.md), [summary](reports/benchmark/summary.json), [exact repeats](reports/benchmark/repeatability.json) |
| Longer training, calendar inputs, baselines | [Protocol](reports/benchmark-extended/report.md), [summary](reports/benchmark-extended/summary.json) |
| Federated budgets and client distributions | [Findings](reports/federated-study/findings.md), [summary](reports/federated-study/summary.json) |
| Centralized GRU baseline | [Results](reports/centralized-baseline.md) |
| Three-client Flower baseline | [Results](reports/federated-smoke/report.md), [exact repeats](reports/federated-smoke/determinism.json) |

Everything uses one sensor subset and one fixed test period, so the seed SDs only show how much training varies on those dates. Each result has its config, file hashes, normalization stats, training histories and metrics in the repo. Raw data and model checkpoints stay local.
"""
    (ROOT / "README.md").write_bytes(text.encode())


if __name__ == "__main__":
    main()
