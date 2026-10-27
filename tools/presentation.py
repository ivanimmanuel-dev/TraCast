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
                    else ":" if name == "persistence" else "-"
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
            range(1, 6),
            mean,
            label=LABELS[name],
            color=COLORS[name],
            linestyle="--" if name == "graph_self" else "-",
        )
        ax.fill_between(
            range(1, 6), mean - sd, mean + sd, color=COLORS[name], alpha=0.10
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Validation MAE (mph)",
        title="Validation convergence: mean and seed SD",
        xticks=range(1, 6),
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

    def metric(item):
        return f"{item['mean']:.3f}" + (
            f" ({item['sd']:.3f})" if item["sd"] is not None else ""
        )

    table = "\n".join(
        f"| {LABELS[name]} | {metric(models[name]['mae'])} | {metric(models[name]['rmse'])} | {models[name]['parameter_count']:,} |"
        for name in ORDER
    )
    study = ROOT / "reports/federated-study/summary.json"
    federated = (
        json.loads(study.read_text())["brief_findings"]
        if study.exists()
        else "The three-client Flower baseline completed twice with identical checkpoint tensors and metrics. Its three-round FedAvg MAE was 4.430 mph against 4.071 for the matched centralized reference."
    )
    federated_bullet = ""
    ci_line = ""
    if study.exists():
        conditions = json.loads(study.read_text())["conditions"]
        first, last = conditions["balanced-1"], conditions["balanced-5"]
        federated_bullet = (f"- In the separate 12-sensor Flower study, mean MAE fell from "
            f"{first['fedavg_mae']['mean']:.3f} to {last['fedavg_mae']['mean']:.3f} mph "
            "over 1 to 5 rounds, at four times the estimated parameter payload.\n")
    receipt = ROOT / "reports/ci.json"
    if receipt.exists():
        ci = json.loads(receipt.read_text())
        if ci["conclusion"] == "success":
            ci_line = f"[{ci['tests'].split()[0]} tests passing on a clean Windows runner]({ci['url']}).\n"
    text = f"""# TraCast

Traffic-speed forecasting on METR-LA with PyTorch and Flower. Recurrent encoders, directed road-distance features, and federated model averaging under chronological evaluation.

[Research brief (PDF)](reports/research-brief.pdf) \u00b7 [Technical report](reports/research-report.md) \u00b7 [Reproduce the experiments](docs/reproducibility.md)

{ci_line}
## Findings

The centralized benchmark uses 16 sensors, the full available March-June 2012 timeline, and five epochs per learned model. Values are mean (sample SD) over seeds 7, 42, and 2026; errors are mph. Persistence is deterministic and evaluated once.

| Model | MAE (SD) | RMSE (SD) | Parameters |
|---|---:|---:|---:|
{table}

- Graph-GRU minus GRU mean MAE: {models['graph_gru']['mae']['mean']-models['gru']['mae']['mean']:+.4f} mph, with {100*(models['graph_gru']['parameter_count']/models['gru']['parameter_count']-1):.0f}% more parameters.
- Removing graph neighbors changed mean MAE by {abs(models['graph_gru']['mae']['mean']-models['graph_self']['mae']['mean']):.4f} mph, with mixed seed-level effects. Most of the gain remained in the self-only projection.
- LSTM reached nearly the same mean error as Graph-GRU with fewer parameters and lower seed dispersion in this configuration.
- Five training epochs averaged {models['gru']['train_seconds']['mean']:.1f} s for GRU, {models['lstm']['train_seconds']['mean']:.1f} s for LSTM, and {models['graph_gru']['train_seconds']['mean']:.1f} s for Graph-GRU on a Ryzen 7 5825U. Fresh-process peak memory was about 320 MiB, including data loading.

{federated_bullet}
![Forecast horizon errors](reports/figures/forecast-horizons.png)

[Per-seed results, timings, and counts](reports/benchmark/results.json) \u00b7 [Graph ablation and protocol](reports/benchmark/report.md)

## Architecture

```mermaid
flowchart TD
    D["METR-LA HDF5<br/>34,272 timestamps x 207 sensors"] --> V["Checksum + schema validation"]
    T["Published DCRNN distances<br/>pinned revision + sensor IDs"] --> A["Directed Gaussian weights<br/>connected 16-sensor subset"]
    V --> S["Chronological 70/10/20 split"]
    A --> S
    S --> N["Per-sensor training statistics<br/>causal fill + masked targets"]
    N --> W["12 input steps -> 12 forecast steps"]
    W --> R["Persistence / GRU / LSTM"]
    W --> G["Own speed + incoming/outgoing means<br/>shared Linear(3,4) + ReLU"]
    A --> G
    G --> H["GRU encoder + forecast head"]
    R --> E["Validation checkpoint selection<br/>test MAE/RMSE by horizon"]
    H --> E
```

The graph contains 85 directed non-self distance relationships. Its weights and subset selection use topology only. The Graph-GRU spatial layer is trained jointly with the recurrent encoder. The self-only ablation removes neighbor inputs and retains the same architecture and parameter count. [Graph source and hashes](DATA.md#sensor-distance-graph).

## Federated learning

{federated}

The Flower `ClientApp`, `ServerApp`, and official `FedAvg` strategy run through the local CLI/SuperLink and Ray runtime. Clients hold disjoint sensor subsets. A shared univariate GRU keeps parameter meanings consistent across clients. Per-sensor normalization uses local training values; global scores pool error sums and valid target counts.

The centralized reference uses the same initial weights, normalized examples, masks, and exposure. Client partitions represent simulated data ownership. They do not represent cities or independent organizations, and the simulation applies neither differential privacy nor secure aggregation.

[Round-budget and heterogeneity study](reports/federated-study/findings.md) \u00b7 [Original Flower baseline](reports/federated-smoke/report.md)

## Run

Python 3.11, Windows x64, CPU PyTorch. From the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m tracast.cli fetch
.\.venv\Scripts\python.exe -m pytest -q -W error

# Three-seed GRU/LSTM/Graph-GRU benchmark and graph ablation
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark-new

# Flower round budgets and client-distribution study
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/study-new
```

The benchmark downloads and validates topology files automatically. Use unused output paths. The measured budget for the model benchmark is a few minutes; the sequential Flower study takes roughly ten minutes on this machine. [Environment, timing definitions, smoke commands, and report regeneration](docs/reproducibility.md).

## Evaluation contract

- Split rows before making windows. Inputs and targets remain within one chronological partition.
- Fit means/scales only on observed training readings; forward-fill within a split and use the training mean for leading gaps.
- Treat zero/NaN targets as missing. Compute MAE/RMSE from valid forecast/sensor pairs, with counts at every horizon.
- Select checkpoints by validation MAE. Report test results after selection.
- Preserve configurations, source/data hashes, sensor IDs, scalers, graph weights, convergence, timing definitions, and model hashes.

[Dataset provenance and citation](DATA.md) \u00b7 [Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml)

## Reports

| Study | Records |
|---|---|
| Recurrent encoders and graph ablation | [Report](reports/benchmark/report.md), [summary](reports/benchmark/summary.json), [repeatability](reports/benchmark/repeatability.json) |
| Federated budgets and heterogeneity | [Findings](reports/federated-study/findings.md), [summary](reports/federated-study/summary.json) |
| Original centralized baseline | [Phase 1 report](reports/verification.md) |
| Original Flower baseline | [Report](reports/federated-smoke/report.md), [exact repeats](reports/federated-smoke/determinism.json) |

The original implementations and results remain under `phase1-verified` and `flower-verified`. Current reports keep those historical results separate from the topology-selected benchmark. Raw data, topology CSVs, checkpoints, client partitions, and local Flower state stay out of Git. Sensor-failure experiments have not been added.
"""
    (ROOT / "README.md").write_bytes(text.encode())


if __name__ == "__main__":
    main()
