# Running TraCast

## Environment

Python 3.11 on Windows x64. Recorded experiments used Python 3.11.9, PyTorch 2.5.1+cpu, Flower 1.42.0, and Ray 2.55.1. The runtime and test dependencies are pinned in `requirements-lock.txt`.

Run from the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -W error
.\.venv\Scripts\python.exe -m tracast.cli fetch
```

Downloads are checked against pinned SHA-256 hashes. The graph benchmark also fetches the DCRNN distances, sensor IDs, and coordinates. [DATA.md](../DATA.md) lists sources, processing, and attribution. Raw data, client partitions, checkpoints, and local Flower state are ignored by Git. Each run requires a new output directory.

## Inspect the data

```powershell
.\.venv\Scripts\python.exe -m tracast.cli inspect --out runs/inspection
```

This validates the full traffic frame, reports missing readings per sensor, and plots training traces, the speed distribution, and missing fractions. Zeros and NaNs count as missing. Timestamps retain the source's recorded clock.

## Centralized models

Start with a seven-day GRU run or a one-epoch graph comparison:

```powershell
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/gru-smoke
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/graph-pilot.json --out runs/graph-smoke
```

The full benchmark uses 16 topology-selected sensors, all 34,272 timestamps, seeds 7/42/2026, and five epochs. Persistence, GRU, LSTM, Graph-GRU, and the self-only ablation share sensor order, split ranges, scalers, masks, batch size, and seeded data order. Each model/seed runs sequentially in a fresh process. Parameter counts differ.

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark
```

Validation MAE selects checkpoints; test scoring follows selection. Each trial saves results, epoch history, protocol, status, and its best checkpoint. The parent directory contains resolved config, source/data/graph hashes, versions, aggregate metrics, paired differences, and plots. Means and sample SD use the same test block across seeds.

To repeat one trial:

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --trial graph_gru --seed 42 --out runs/graph-repeat
```

Compare test metrics, validation histories, and `state_sha256`. The recorded Graph-GRU and LSTM seed-42 repeats matched exactly; see [repeatability.json](../reports/benchmark/repeatability.json). Timing and memory vary between executions.

## Flower simulation

Three clients, two sensors each, seven days, three rounds, one local epoch per round:

```powershell
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-a
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-b
.\.venv\Scripts\python.exe -m tracast.compare_runs runs/flower-a runs/flower-b
```

The runner launches Flower's `ClientApp`, `ServerApp`, and `FedAvg` through `flwr run --stream`, a local SuperLink, and Ray. It uses the installed environment, one active client actor, two reserved Ray CPUs, one PyTorch thread, and a 128 MiB object store. Ray support on Windows is experimental. Use a dedicated environment so an existing SuperLink does not retain another project's dependency settings.

Client partitions hold disjoint sensors on the same host. Each sensor uses its own observed-training scaler. The pooled reference uses the same initial weights, examples, masks, exposure, and optimizer-reset schedule. Partial final batches produce different optimizer-step counts. The shared model forecasts one sensor per example.

## Round budgets and client distributions

```powershell
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study
```

The preset runs twelve sequential experiments on twelve sensors and fourteen days:

| Partition | Round budgets | Seeds |
|---|---|---|
| Balanced congestion ranks | 1, 3, 5 | 7, 42, 2026 |
| Congestion-grouped ranks | 5 | 7, 42, 2026 |

Sensors rank by the fraction of observed training speeds below 40 mph. Balanced groups interleave ranks; congestion groups take consecutive rank blocks. Each client owns four sensors. All cases share temporal boundaries. Ranking, scaling, and distribution histograms use training rows only.

The pooled dataset retains canonical sensor order across partition rules. The study checks identical pooled results and model hashes for equal budgets, plus identical federated model prefixes across round budgets. Per-case records include client sample counts, training and validation convergence, horizon metrics, selected test scores, and communication estimates.

The recorded full benchmark takes a few minutes; the Flower study took 11.45 minutes on a Ryzen 7 5825U. [Results and controls](../reports/federated-study/findings.md).

## Measurement definitions

| Measurement | Definition |
|---|---|
| MAE / RMSE | Error sums pooled over valid forecast/sensor pairs, in mph; also reported by horizon |
| Training time | Update loops including batch retrieval, excluding validation |
| Evaluation time | Complete scoring loop including inverse normalization and metrics |
| Forward inference | Median of 30 normalized forward passes after 5 warmups on the first test batch, per window |
| Peak memory | Whole fresh-process Windows PeakWorkingSetSize, including loading and preprocessing |
| Flower wall time | Launch, simulation, selected test scoring, pooled reference, and plots |
| Communication | Raw model tensor bytes for training downloads/uploads, validation downloads, and selected-model test downloads |

Communication excludes serialization, metrics, configuration, and transport overhead. Workstation timings include normal background activity. Repeated-seed SD measures initialization/shuffle variation on fixed dates.

## Regenerate reports

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe tools/presentation.py
.\.venv\Scripts\python.exe tools/build_research.py
```

These commands read the committed experiment records and rebuild the README, figures, technical report, and two-page PDF. They do not train models. `tools/export_records.py` replaces local paths with `$REPO` and `$PYTHON_ENV` when exporting records; the export receipt retains before/after hashes. Historical manifests identify the source content and dependency lock used by each measured run.
