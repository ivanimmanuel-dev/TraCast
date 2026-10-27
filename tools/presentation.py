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

    def metric(item):
        return f"{item['mean']:.3f}" + (
            f" ({item['sd']:.3f})" if item["sd"] is not None else ""
        )

    table = "\n".join(
        f"| {LABELS[name]} | {metric(models[name]['mae'])} | {metric(models[name]['rmse'])} | {models[name]['parameter_count']:,} |"
        for name in ORDER
    )
    conditions = json.loads(
        (ROOT / "reports/federated-study/summary.json").read_text()
    )["conditions"]
    federated_rows = "\n".join(
        f"| {'Balanced' if c['partition_rule'] == 'balanced' else 'Congestion-grouped'} | {c['round_budget']} | {c['fedavg_mae']['mean']:.3f} | {c['centralized_mae']['mean']:.3f} | {c['parameter_bytes']:,} B |"
        for c in conditions.values()
    )
    text = rf"""# TraCast

Traffic-speed forecasting on METR-LA. PyTorch recurrent models, directed road-distance features, and Flower federated learning, sized for a CPU laptop.

[Research brief](reports/research-brief.pdf) · [Technical report](reports/research-report.md) · [Run guide](docs/reproducibility.md) · [Data sources](DATA.md) · [Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml)

## Results

Sixteen topology-selected sensors, March–June 2012, five training epochs. Errors are mph, reported as mean (sample SD) over seeds 7, 42, and 2026. Persistence is evaluated once.

| Model | MAE (SD) | RMSE (SD) | Parameters |
|---|---:|---:|---:|
{table}

LSTM nearly matches Graph-GRU's mean MAE with fewer parameters. Removing graph neighbors changes mean MAE by {abs(models['graph_gru']['mae']['mean']-models['graph_self']['mae']['mean']):.4f} mph, with mixed effects across seeds. The self-only projection retains most of the improvement over GRU.

Five epochs averaged {models['gru']['train_seconds']['mean']:.1f} s for GRU, {models['lstm']['train_seconds']['mean']:.1f} s for LSTM, and {models['graph_gru']['train_seconds']['mean']:.1f} s for Graph-GRU on a Ryzen 7 5825U. Whole-process peak memory was about 320 MiB.

![Test error by forecast horizon; shading shows seed SD](reports/figures/forecast-horizons.png)

[Per-seed metrics and timings](reports/benchmark/results.json) · [Benchmark protocol](reports/benchmark/report.md)

## Federated learning

Three simulated clients hold disjoint sensor subsets and share a univariate GRU. Flower's `ClientApp`, `ServerApp`, and `FedAvg` run through a local SuperLink and Ray. Each sensor uses its own training-only normalization; evaluation pools error sums and valid target counts across clients.

The twelve-sensor, fourteen-day study uses three seeds and one local epoch per round:

| Partition | Rounds | FedAvg MAE | Pooled GRU MAE | Parameter payload |
|---|---:|---:|---:|---:|
{federated_rows}

One to five rounds reduces balanced-partition MAE by 0.273 mph at four times the parameter payload. The pooled reference shares initial weights, examples, scalers, masks, and total exposure. Partial batches yield different optimizer-step counts. Payload estimates count model tensors; client ownership is simulated on one host.

[Round convergence, RMSE, client counts, and partition controls](reports/federated-study/findings.md)

## Run

Python 3.11, Windows x64, CPU PyTorch. From the repository directory:

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

For the full comparisons:

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study
```

The benchmark fetches topology files automatically. Use a fresh output directory for each run. [Configuration, inspection, repeats, and measurement definitions](docs/reproducibility.md).

## Pipeline

```mermaid
flowchart TD
    D[METR-LA speeds] --> V[Checksum and schema validation]
    T[DCRNN distance graph] --> G[Directed incoming/outgoing means]
    V --> S[Chronological 70/10/20 split]
    S --> N[Training-only scalers and causal fill]
    N --> W[12 input steps → 12 forecast steps]
    W --> R[Persistence / GRU / LSTM]
    W --> G
    G --> H[Shared node projection → GRU]
    R --> E[Validation selection → test MAE/RMSE]
    H --> E
```

Windows stay within their chronological split. Zero/NaN targets are masked. The graph's 16 sensors and 85 directed relationships are selected from topology alone. The self-only ablation keeps the same architecture and parameter count while removing neighbor inputs.

## Reports

| Experiment | Results |
|---|---|
| Recurrent encoders and graph ablation | [Methods](reports/benchmark/report.md), [summary](reports/benchmark/summary.json), [exact repeats](reports/benchmark/repeatability.json) |
| Federated budgets and client distributions | [Findings](reports/federated-study/findings.md), [summary](reports/federated-study/summary.json) |
| Centralized GRU baseline | [Results](reports/centralized-baseline.md) |
| Three-client Flower baseline | [Results](reports/federated-smoke/report.md), [exact repeats](reports/federated-smoke/determinism.json) |

The studies use sensor subsets and one fixed test period. Seed SD measures training variation on those dates. Raw data and checkpoints remain local; committed records include configurations, source/data hashes, scalers, histories, and horizon counts.
"""
    (ROOT / "README.md").write_bytes(text.encode())


if __name__ == "__main__":
    main()
