"""Build the README and the figures from the saved experiment summaries."""

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
# rows of the README table
COMPACT = [
    "persistence",
    "ha",
    "linear",
    "gru",
    "lstm",
    "graph_gru",
    "gru_time",
    "graph_gru_time",
]
# (key in summary["day_bootstrap"], label); every estimate is first minus second
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


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


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


def at(entry, minutes):
    row = next(r for r in entry["horizons"] if r["minutes"] == minutes)
    return row["mae"]["mean"]


def extended():
    """Everything the text needs from the longer-training study, or None."""
    folder = ROOT / "reports/benchmark-extended"
    if not (folder / "summary.json").exists():
        return None
    summary = load("reports/benchmark-extended/summary.json")
    return {
        "models": summary["models"],
        "found": summary["day_bootstrap"],
        "config": load("reports/benchmark-extended/config.json"),
    }


def extended_rows(ext, names):
    """Full table rows: MAE with SD, test-day range, 15/30/60 min, RMSE, size, epochs."""
    rows = []
    for name in names:
        m = ext["models"][name]
        low, high = m["test_mae_ci95_days"]
        if "epochs_run" in m:
            rows.append(
                f"| {EXTENDED_LABELS[name]} | {mean_sd(m['mae'])} | {low:.3f}-{high:.3f} "
                f"| {at(m, 15):.3f} | {at(m, 30):.3f} | {at(m, 60):.3f} "
                f"| {mean_sd(m['rmse'])} | {m['parameter_count']:,} "
                f"| {m['epochs_run']['mean']:.0f} |"
            )
        else:
            rows.append(
                f"| {EXTENDED_LABELS[name]} | {m['mae']['mean']:.3f} | {low:.3f}-{high:.3f} "
                f"| {at(m, 15):.3f} | {at(m, 30):.3f} | {at(m, 60):.3f} "
                f"| {m['rmse']['mean']:.3f} | 0 | - |"
            )
    return "\n".join(rows)


def gap_rows(ext):
    found = ext["found"]
    return "\n".join(
        f"| {label} | {found[key]['estimate']:+.3f} "
        f"| {found[key]['ci95'][0]:+.3f} to {found[key]['ci95'][1]:+.3f} |"
        for key, label in COMPARISONS
        if key in found
    )


def top_three(models):
    """Best learned model, the next two, and how far apart the three are."""
    learned = [n for n in models if "epochs_run" in models[n]]
    ranked = sorted(learned, key=lambda n: models[n]["mae"]["mean"])
    spread = models[ranked[2]]["mae"]["mean"] - models[ranked[0]]["mae"]["mean"]
    return ranked[0], ranked[1:3], spread


def graph_sentence(ext):
    """What removing the neighbor averages does, with and without the calendar."""
    found = ext["found"]
    keys = [
        key
        for key in ("graph_gru_vs_graph_self", "graph_gru_time_vs_graph_self_time")
        if key in found
    ]
    parts = [
        f"{phrase(found[key])} {'with' if key.endswith('_time') else 'without'} "
        f"calendar inputs"
        for key in keys
    ]
    apart = [key for key in keys if separated(found[key])]
    opposite = len({found[key]["estimate"] > 0 for key in keys}) == 2
    if not apart:
        verdict = "Neither gap is clearly different from zero."
    elif len(apart) == len(keys) and opposite:
        verdict = "The sign flips, so the graph doesn't give a reliable gain."
    else:
        verdict = "At least one of the two is clearly different from zero."
    return (
        "Graph-GRU minus its self-only version (the same model with the neighbor "
        "averages zeroed) is " + " and ".join(parts) + f". {verdict}"
    )


def extended_narrative(ext, first_pass):
    models, found = ext["models"], ext["found"]
    best, others, spread = top_three(models)
    names = " and ".join(EXTENDED_LABELS[n] for n in others)
    sentences = [
        f"{EXTENDED_LABELS[best]} is lowest at {models[best]['mae']['mean']:.3f} mph "
        f"(persistence {models['persistence']['mae']['mean']:.3f}, historical average "
        f"{models['ha']['mae']['mean']:.3f}), but {names} are within {spread:.3f} mph "
        f"of it, so I can't rank them."
    ]
    gained = first_pass["gru"]["mae"]["mean"] - models["gru"]["mae"]["mean"]
    sentences.append(
        f"Training longer helped: the GRU went from {first_pass['gru']['mae']['mean']:.3f} "
        f"at five epochs to {models['gru']['mae']['mean']:.3f}."
    )
    gap = found["gru_time_vs_gru"]
    graph = [
        abs(found[k]["estimate"])
        for k in ("graph_gru_vs_graph_self", "graph_gru_time_vs_graph_self_time")
        if k in found
    ]
    if not separated(gap):
        sentences.append(
            f"Calendar inputs change the GRU by {phrase(gap)}, which the test days "
            f"can't tell apart from zero."
        )
    elif gap["estimate"] > 0:
        sentences.append(f"Calendar inputs hurt the GRU by {phrase(gap)}.")
    else:
        biggest = all(abs(gap["estimate"]) > other for other in graph + [gained])
        lead = "gave the biggest jump" if biggest else "helped"
        sentences.append(f"Calendar inputs {lead}: the GRU changes by {phrase(gap)}.")
    sentences.append(graph_sentence(ext))
    return " ".join(sentences)


def extended_intro(config):
    return (
        f"Each model trains for up to {config['epochs']} epochs, stops after "
        f"{config['patience']} epochs with no validation gain, and halves the learning "
        f'rate after {config["scheduler_patience"]}. Models marked "+ calendar" also get '
        f"the time of day (sine and cosine) and a weekend flag as inputs. I added two "
        f"simple baselines: a historical average (the training mean for each sensor, "
        f"weekday and 5-minute slot) and a linear model over the whole input window. "
        f"{len(config['seeds'])} seeds: {', '.join(map(str, config['seeds']))}."
    )


def federated(conditions):
    """Pieces of the federated text: table, comparison sentence, round-budget cost."""
    local = all("local_only_mae" in c for c in conditions.values())
    balanced = sorted(
        (c for c in conditions.values() if c["partition_rule"] == "balanced"),
        key=lambda c: c["round_budget"],
    )
    shortest, longest = balanced[0], balanced[-1]
    grouped = [c for c in conditions.values() if c["partition_rule"] == "congestion"]
    header = (
        "| Sensor grouping | Rounds | Federated MAE | Centralized MAE | "
        + ("Local-only MAE | " if local else "")
        + "Model weights sent |\n|---|---:|---:|---:|"
        + ("---:|" if local else "")
        + "---:|"
    )
    rows = "\n".join(
        f"| {'Balanced' if c['partition_rule'] == 'balanced' else 'Congestion-grouped'} "
        f"| {c['round_budget']} | {c['fedavg_mae']['mean']:.3f} "
        f"| {c['centralized_mae']['mean']:.3f} | "
        + (f"{c['local_only_mae']['mean']:.3f} | " if local else "")
        + f"{c['parameter_bytes']:,} B |"
        for c in conditions.values()
    )
    compare = ""
    if local:
        compare = (
            f"At {longest['round_budget']} rounds with balanced groups, federated "
            f"averaging gets {longest['fedavg_mae']['mean']:.3f} mph, local-only "
            f"{longest['local_only_mae']['mean']:.3f} and centralized "
            f"{longest['centralized_mae']['mean']:.3f}."
        )
        if grouped:
            g = grouped[-1]
            compare += (
                f" With congestion-grouped clients it is {g['fedavg_mae']['mean']:.3f} "
                f"against {g['local_only_mae']['mean']:.3f}."
            )
    verdict = (
        "The gaps are about the size of the seed spread, so in this setup training "
        "alone is about as good as federating."
        if local
        else ""
    )
    cost = (
        f"Going from {shortest['round_budget']} to {longest['round_budget']} rounds "
        f"cuts balanced MAE by "
        f"{shortest['fedavg_mae']['mean'] - longest['fedavg_mae']['mean']:.3f} mph and "
        f"sends {longest['parameter_bytes'] / shortest['parameter_bytes']:g} times as "
        f"many model-weight bytes. The byte counts cover weights sent for training, "
        f"validation and the final test; message overhead isn't counted."
    )
    return {
        "header": header,
        "rows": rows,
        "compare": compare,
        "verdict": verdict,
        "cost": cost,
    }


def forecast_figure(out, models):
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


def convergence_figure(out, records):
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


def extended_figure(out, ext):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, color in EXTENDED_COLORS.items():
        if name not in ext["models"]:
            continue
        entry = ext["models"][name]
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


EXTENDED_REPORT = """# Longer training, calendar inputs, baselines

The [first benchmark](../benchmark/report.md) trained every model for five epochs and stopped while they were still improving. This one fixes that and adds a few things. Same 16 sensors, same split, same masked loss.

{intro}

| Model | MAE (SD) | MAE range over test days | 15 min | 30 min | 60 min | RMSE (SD) | Parameters | Epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

MAE and RMSE are test means in mph; 15/30/60 min are MAE at those horizons. The range column resamples the {days} test days to show how much a score depends on which days ended up in the test set. Persistence and the historical average aren't trained, so they run once.

{narrative}

| Difference in MAE (first - second) | Estimate (mph) | 95% CI over test days |
|---|---:|---:|
{gaps}

Negative means the first model is better. Both models in a row are scored on the same windows and the same resampled days. The intervals are about which days were held out; the seed SDs in the table above cover random initialization.

![Test error by forecast horizon; shading is seed SD](../figures/extended-horizons.png)

![Validation convergence](convergence.png)

## Running it

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.benchmark --config configs/benchmark-extended.json --out runs/benchmark-extended
```

This writes one folder per model and seed, then `results.json`, `summary.json` and the plots. `summary.json` has the seed stats plus a `day_bootstrap` section with the intervals above (every trial saves per-day error sums, which is all the bootstrap needs).

Timings are rough: the machine was doing other things during part of the run, so don't compare them with the first benchmark. The errors aren't affected. Repeating the GRU seed-7 trial under a different load gave exactly the same validation errors.

The manifest says `git_dirty` because the run used code that wasn't committed yet. After the 47 trials finished, the convergence plot crashed on early-stopped runs of different lengths, so I fixed `benchmark.plots` and redrew the plots from `results.json`. No trial or metric changed.

Model names ending in `_time` get the calendar inputs. `ha` is fit on the training split only, and `linear` is one linear layer over the flattened window.

Files: [summary](summary.json), [per-seed results](results.json), [protocol](protocol.json), [manifest](manifest.json), [config](config.json).
"""

README = """# TraCast

**Traffic-speed forecasting on METR-LA, from recurrent baselines to federated graph models, on a CPU.**

[![Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml/badge.svg)](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml) ![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![MIT license](https://img.shields.io/badge/license-MIT-green)

TraCast forecasts the speed on 207 Los Angeles road sensors 5 to 60 minutes ahead from the last hour of readings (METR-LA, March to June 2012). It started with four questions: do recurrent models beat simple baselines, does the road graph help, does the time of day help, and does federated training lose anything against pooling the data? It grew into scaling to all 207 sensors, prediction intervals, drift, forecasting for sensors a model never saw, a DCRNN-style diffusion model, and federated learning on the whole network.

Every number in the reports is generated from saved results, a comparison is only called better or worse when an interval over test days says so, and the tests include leakage checks.

**Start with** the [technical report](reports/research-report.md) or the [two-page brief](reports/research-brief.pdf). **Studies:** [scaling](reports/scaling/report.md) · [shared variants](reports/variants/report.md) · [intervals](reports/uncertainty/report.md) · [drift](reports/shift/report.md) · [federated graph](reports/fedgraph/report.md) · [DCRNN](reports/diffusion/report.md) · [FedProx](reports/fedopt/report.md) · [new sensors](reports/holdout/report.md). **Also:** [audit](docs/audit.md) · [run guide](docs/reproducibility.md) · [data sources](DATA.md).

## At a glance

{findings}

## Quick start

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe -m pip install -e . --no-deps
.\\.venv\\Scripts\\python.exe -m pytest -q -W error
.\\.venv\\Scripts\\python.exe -m tracast.cli fetch
.\\.venv\\Scripts\\python.exe -m tracast.cli run --config configs/smoke.json --out runs/gru-smoke
```

The recorded experiments ran on Python 3.11 and CPU PyTorch on Windows x64, and CI runs the tests on Windows and Ubuntu (on Linux use `python3.11` and `.venv/bin/python`). The last command trains one GRU on seven days of data as a smoke test. `tracast --help` lists every experiment, and the [run guide](docs/reproducibility.md) has the command for each, from the five-epoch benchmark to the multi-hour federated runs. Use a fresh output folder for each run.

## How the experiments are set up

The first 70% of the timeline is training, the next 10% validation and the last 20% (24 days) test, and windows never cross a split boundary. Zero and missing readings are left out of the loss and the scores, inputs are forward-filled within a split, and normalization uses training data only. The saved model is the one with the best validation MAE, and the test set is scored after that. Gaps between models are measured on identical windows, with 95% intervals from resampling test days.

The early studies use 16 sensors picked by walking the published DCRNN distance graph from its first sensor, which gives 85 directed edges. Sensors and edge weights come from distances only, never from speeds. Graph-GRU gives each sensor its own speed plus the average of its upstream and downstream neighbors, passes that through a small shared layer and feeds all 16 sensors into a GRU; "self only" is the same model with the neighbor averages set to zero. Every model in those studies, the plain GRU included, reads all 16 sensors, so they test the graph-weighted averages and not whether neighboring sensors matter. The shared-weight models further down take that confound away.

## Baselines and the clock (16 sensors)

Test MAE in mph, forecasting 12 steps (60 minutes) from the previous 12. Learned models are averaged over {seeds} seeds, with the standard deviation in parentheses. Persistence and the historical average aren't trained, so they run once. The [full table](reports/benchmark-extended/report.md) has RMSE, parameter counts and how much each score moves with the choice of test days.

| Model | MAE (SD) | 15 min | 60 min |
|---|---:|---:|---:|
{rows}

![Test error by forecast horizon; shading is seed SD](reports/figures/extended-horizons.png)

My first run trained every model for five epochs. In {at_last} of {n_runs} runs the best validation epoch was the last one, so the models were still improving when training stopped. The numbers above come from a rerun with early stopping (up to {max_epochs} epochs) and {seeds} seeds. The five-epoch results are still in [reports/benchmark](reports/benchmark/report.md).

## A first federated study (12 sensors)

The Flower simulation gives each of three clients its own set of sensors. Every round each client trains its copy of the same GRU on its own readings, and the server averages the weights, weighted by how many training examples each client has (one example is one sensor's readings). Centralized trains the same model on all clients' data together, from the same starting weights, normalization, masks and number of examples. Local-only trains each client by itself with the same amount of local work and no communication.

Twelve sensors, fourteen days, three seeds, one local epoch per round. Sensors are ranked by how often their training speeds fall below 40 mph; balanced groups spread those ranks across the clients and congestion-grouped clients get consecutive blocks.

{fed_header}
{fed_rows}

{fed_compare} {fed_verdict}

{fed_cost} More in the [federated findings](reports/federated-study/findings.md).

{scaling_section}{uncertainty_section}{shift_section}{federated_graph_section}## Limits

- **Data.** One city, one four-month period and one 24-day test window. The early studies use 16 sensors on a single highway corridor, and the test period has more missing readings than training (the [audit](docs/audit.md) has the numbers).
- **Training.** No hyperparameter search, and models get equal update budgets rather than training to convergence, so a model that needs longer is undersold. Most federated studies use one or two seeds and one geographic partition.
- **Federation.** Every federated run is a simulation on one machine: no real network, stragglers or dropped clients.
- **Comparisons.** Nothing here is compared with published METR-LA numbers, because splits, horizons and budgets differ. The DCRNN-style model predicts all horizons at once instead of using the paper's decoder.

Each report lists its own limits in more detail.

{next_section}## Repository map

- `src/tracast/` holds the package:
  - data and protocol: `data.py`, `topology.py` (the pinned DCRNN road graph), `experiment.py`, `stats.py` (test-day bootstrap), `provenance.py`
  - models: `model.py` (GRU), `graph_model.py`, `recurrent.py` (LSTM), `shared.py` (shared-weight models), `diffusion.py` (DCRNN-style), `baselines.py`
  - experiments: `benchmark.py`, `scaling.py`, `uncertainty.py` and `conformal.py`, `shift.py`, `inductive.py` (held-out sensors)
  - federated learning: `federated*.py` and `flower_*.py` (the first 12-sensor study), `fedgraph*.py` (the federated graph models: partitions, FedAvg and FedProx, the Flower apps, and the arms x seeds runner)
- `tests/` has unit tests on small constructed arrays, so no real data is needed, plus leakage tests.
- `configs/` has one JSON file per experiment.
- `reports/` has the saved results for every experiment, the technical report and the brief.
- `docs/` has the run guide and the audit of the code and data.
- `tools/` has the scripts that rebuild this README, the figures and the reports from the saved results.

## Credits and license

The code is MIT licensed. METR-LA and the road graph come from Li, Yu, Shahabi and Liu (ICLR 2018) and have their own terms, see [DATA.md](DATA.md). Federated runs use [Flower](https://flower.ai). The [technical report](reports/research-report.md#references) lists the papers behind the methods. To cite TraCast, use [CITATION.cff](CITATION.cff).
"""

NL = chr(10)


def findings(ext, first_pass, fed):
    """The headline results as markdown bullets, numbers read from the summaries."""
    m, found = ext["models"], ext["found"]
    gru, gru_t = m["gru"], m["gru_time"]
    spread = top_three(m)[2]
    return [
        f"- **The clock helps more than anything else I tried.** Adding the time of day "
        f"and a weekend flag took the GRU from {gru['mae']['mean']:.3f} to "
        f"{gru_t['mae']['mean']:.3f} mph MAE, a change of {phrase(found['gru_time_vs_gru'])}, "
        f"and at the 60-minute horizon from {at(gru, 60):.3f} to {at(gru_t, 60):.3f}. "
        f"Training longer helped too, but less: five epochs gave "
        f"{first_pass['gru']['mae']['mean']:.3f}.",
        f"- **The road graph doesn't give a reliable gain.** The neighbor averages help a "
        f"little without the clock (MAE {found['graph_gru_vs_graph_self']['estimate']:+.3f} "
        f"mph) and hurt a little with it "
        f"({found['graph_gru_time_vs_graph_self_time']['estimate']:+.3f}). The best three "
        f"models are within {spread:.3f} mph of each other, so I can't rank them.",
        f"- **Everything beats persistence ({m['persistence']['mae']['mean']:.3f}).** "
        f"The historical average ({m['ha']['mae']['mean']:.3f}) and a linear model "
        f"({m['linear']['mae']['mean']:.3f}) land in between, and the GRUs are well "
        f"ahead of both. (At 15 minutes persistence is still ahead of the historical "
        f"average.)",
        f"- **In the first, small federated study (12 sensors, three clients), federated averaging was about as good as training each client alone.** "
        f"{fed['compare']}",
    ]


def build_readme(first_pass, records):
    import glance
    from pages import readme_blocks

    blocks = readme_blocks()
    ext = extended()
    fed = federated(load("reports/federated-study/summary.json")["conditions"])
    best_epochs = [
        (r["best_epoch"], max(h["epoch"] for h in r["history"]))
        for r in records
        if r["history"]
    ]
    m = ext["models"]
    return README.format(
        findings=NL.join(glance.bullets(ext)),
        seeds=len(ext["config"]["seeds"]),
        rows=NL.join(
            f"| {EXTENDED_LABELS[n]} | {mean_sd(m[n]['mae'])} | {at(m[n], 15):.3f} "
            f"| {at(m[n], 60):.3f} |"
            for n in COMPACT
        ),
        at_last=sum(best == last for best, last in best_epochs),
        n_runs=len(best_epochs),
        max_epochs=ext["config"]["epochs"],
        fed_header=fed["header"],
        fed_rows=fed["rows"],
        fed_compare=fed["compare"],
        fed_verdict=fed["verdict"],
        fed_cost=fed["cost"],
        scaling_section=blocks["scaling"] + "\n" if blocks["scaling"] else "",
        uncertainty_section=blocks["uncertainty"] + "\n"
        if blocks["uncertainty"]
        else "",
        shift_section=blocks["shift"] + "\n" if blocks["shift"] else "",
        federated_graph_section=blocks["federated_graph"] + "\n"
        if blocks["federated_graph"]
        else "",
        next_section=blocks["next"] + "\n",
    )


def main():
    source = ROOT / "reports/benchmark/summary.json"
    first_pass = json.loads(source.read_text())["models"]
    records = load("reports/benchmark/results.json")
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
    forecast_figure(out, first_pass)
    convergence_figure(out, records)
    provenance = {
        "source": "reports/benchmark/summary.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "uncertainty": "Sample SD across three training seeds",
    }
    (out / "provenance.json").write_bytes(
        (json.dumps(provenance, indent=2) + "\n").encode()
    )
    ext = extended()
    extended_figure(out, ext)
    names = [n for n in EXTENDED_LABELS if n in ext["models"]]
    report = EXTENDED_REPORT.format(
        intro=extended_intro(ext["config"]),
        rows=extended_rows(ext, names),
        days=next(iter(ext["found"].values()))["days"],
        narrative=extended_narrative(ext, first_pass),
        gaps=gap_rows(ext),
    )
    (ROOT / "reports/benchmark-extended/report.md").write_bytes(report.encode())
    (ROOT / "README.md").write_bytes(build_readme(first_pass, records).encode())


if __name__ == "__main__":
    main()
