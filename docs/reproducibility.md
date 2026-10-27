# Reproduce the experiments

## Environment

Python 3.11 on Windows x64. The measured runs used Python 3.11.9, PyTorch 2.5.1+cpu, Flower 1.42.0, and Ray 2.55.1. `requirements-lock.txt` pins the numerical runtime and test dependencies. Report generation has its own `requirements-report.txt`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The development environment is `..\..\work\flwr-venv\Scripts\python.exe` relative to this checkout. It is outside the repository. Replace the interpreter path in the commands below to use it.

## Inputs

```powershell
.\.venv\Scripts\python.exe -m tracast.cli fetch
```

The HDF5 download is pinned to an immutable mirror revision and checked against its published SHA-256 before loading. The benchmark fetches the DCRNN distance, sensor-ID, and location files automatically. Its three graph-file checksums are pinned in `src/tracast/topology.py`. [DATA.md](../DATA.md) records exact source URLs, sizes, hashes, processing choices, attribution, and license declarations.

The original data files remain unchanged. Raw HDF5 files, topology CSVs, generated client partitions, and checkpoints are excluded from Git.

## Centralized benchmark

Pilot: 16 connected sensors, seven days, one epoch, seed 42.

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/graph-pilot.json --out runs/graph-pilot-new
```

Full comparison: the same 16 sensors, all 34,272 timestamps, seeds 7/42/2026, and five epochs per learned model. Models: persistence, GRU, LSTM, Graph-GRU, and a self-only graph ablation.

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark-new
```

Every model/seed runs in a fresh process, sequentially. All share the same sensor order, split ranges, training statistics, target masks, batch size, epoch count, and seed-specific data order. Model parameter counts differ. Validation MAE selects each checkpoint; test data are scored after selection. `summary.json` reports means and sample SD across seeds. The fixed test period is shared across seeds.

Each trial saves `result.json`, epoch history, protocol, status, and a local best checkpoint. The parent run saves config, source/data/graph hashes, versions, aggregate results, paired differences, and figures. Use a new output directory for each run.

For a single repeat:

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --trial graph_gru --seed 42 --out runs/graph-repeat-new
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --trial lstm --seed 42 --out runs/lstm-repeat-new
```

Compare test dictionaries, validation histories, and `state_sha256` values. Timing and memory fields are measured and can vary. The executed seed-42 repeats matched exactly; [repeatability.json](../reports/benchmark/repeatability.json) records those checks.

## Flower smoke run

```powershell
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-new-a
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-new-b
.\.venv\Scripts\python.exe -m tracast.compare_runs runs/flower-new-a runs/flower-new-b
```

This preserves the original six-sensor, seven-day, three-round setup. The original `contiguous` partition rule remains the default. The current record adds training-distribution metadata, so its config/source identity differs from the historical manifest even when numerical results match.

## Federated budget and heterogeneity study

```powershell
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study-new
```

The preset runs twelve Flower experiments sequentially:

| Rule | Round budgets | Seeds |
|---|---|---|
| Balanced congestion ranks | 1, 3, 5 | 7, 42, 2026 |
| Congestion-grouped ranks | 5 | 7, 42, 2026 |

The first twelve sensors in canonical traffic-file order provide two weeks of data. Each client owns four sensors. Sensor rankings use only each sensor's observed training fraction below 40 mph. Balanced groups interleave the rank order; congestion groups take consecutive rank blocks. Validation/test values never enter the ranking. Recorded training-speed histograms quantify the resulting distribution difference.

A shared univariate GRU keeps parameter semantics aligned across clients. Each sensor retains its local observed-training mean/scale. The pooled reference orders examples by the original canonical sensor order, regardless of client assignment; its parameters and metrics are checked for equality across partition rules. The study also checks exact global-model prefix hashes across 1/3/5-round runs at each seed.

A local Flower SuperLink launches the official ClientApp/ServerApp/FedAvg runtime through `flwr run --stream`. Runtime dependency installation is disabled so workers use the preinstalled lock. One client actor is active at a time, with two reserved Ray CPUs, one PyTorch thread, and a 128 MiB object-store cap. Ray's Windows support is experimental. The launcher sets UTF-8 output and disables update checks and usage telemetry for the local run.

The managed SuperLink can remain available between runs. A previously running service from another environment can retain different dependency settings; use a dedicated environment/session for this project. The completed study's local test service was stopped after verification. Client ownership is logical on the shared simulation host; it is not an operating-system privacy boundary.

Each case stores all round metrics, client counts, additive horizon errors, test results, runtime logs, and parameter-payload estimates. Communication counts model tensor values for training downloads/uploads, validation downloads, and selected-model test downloads. It excludes packaging, messages, metrics, configuration, transport framing, latency, and energy.

## Timing and memory definitions

- Training seconds: cumulative update-loop time, including batch retrieval, excluding validation.
- Validation and test-evaluation seconds: complete scoring loops, including inverse normalization and metrics.
- Forward inference: median of thirty normalized forward passes after five warmups on the first test batch; reported per window. No loader or scoring cost is included.
- Peak memory: whole fresh process PeakWorkingSetSize on Windows, including loading, preprocessing, training, and scoring. It is not model-only allocation.
- Flower case wall time: startup, simulation, selected test scoring, pooled reference, and plots. This includes substantial runtime overhead.

The model benchmark ran sequentially on a Ryzen 7 5825U with four PyTorch threads. Other documentation work continued on the workstation. These timings are measurements from that run, not isolated hardware benchmarks.

## Reports and figures

The report builder reads committed metric records rather than training new models.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe tools/presentation.py
.\.venv\Scripts\python.exe tools/build_research.py
.\.venv\Scripts\python.exe tools/export_records.py
```

`tools/presentation.py` renders figures and the README. `tools/build_research.py` writes the technical report and two-page research brief. The PDF was rendered with Poppler and visually checked. Intermediate page images are kept outside the repository.

## Regression records

The original centralized and Flower implementations remain under `phase1-verified` and `flower-verified`. Their result JSONs and figures are preserved. The original centralized smoke metrics were reproduced exactly. New Graph-GRU and LSTM seed-42 repeats also matched their model hashes, test scores, and validation histories.

The first clean GitHub installation exposed an obsolete `trafficmind==0.1.0` editable-package entry in the lock. Its generated source metadata was removed and the lock corrected. Numerical library versions did not change. Historical manifests retain the source hashes of the files used at execution time; current setup commands use the corrected lock.

Exported report records use `$REPO` and `$PYTHON_ENV` in place of workstation paths. `reports/export-provenance.json` retains hashes of the original and exported files. Numerical values, model/data hashes, and deterministic run IDs remain unchanged; complete local run records remain in ignored run directories.
