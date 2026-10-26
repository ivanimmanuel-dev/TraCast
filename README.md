# TraCast

Traffic-speed forecasting on real METR-LA with PyTorch and Flower. CPU-first research code: persistence, a GRU baseline, and a three-client FedAvg simulation.

## Setup

Python 3.11 on Windows x64. From the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m tracast.cli fetch
.\.venv\Scripts\python.exe -m pytest -q -W error
```

The lock includes the CPU build of PyTorch, Flower 1.42.0, Ray 2.55.1, and the test dependencies. It preserves the Phase 1 NumPy, pandas, PyTables, matplotlib, and PyTorch versions. The original environment lock is retained in `reports/phase1-requirements-lock.txt`.

For the already-installed development environment, use `..\..\work\flwr-venv\Scripts\python.exe` in place of `.\.venv\Scripts\python.exe`.

## Three-round Flower smoke run

```powershell
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/fl-new-a
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/fl-new-b
.\.venv\Scripts\python.exe -m tracast.compare_runs runs/fl-new-a runs/fl-new-b
```

Each run executes the genuine Flower `ClientApp`, `ServerApp`, and `flwr.serverapp.strategy.FedAvg` through `flwr run --stream`, using a managed local SuperLink and Ray. It then trains and evaluates the matched centralized GRU. No handwritten replacement for Flower's aggregation is used.

The smoke config selects the first six sensors and seven days of observations, assigns two disjoint sensors to each of three simulated clients, and runs three rounds with one local epoch per client. All clients participate every round. One Ray actor slot and one PyTorch thread per client keep memory and CPU use small; Ray's object-store limit is 128 MiB. The client IDs describe sensor partitions only.

The launcher disables runtime dependency installation so workers use the locked environment. It stores Flower's local files inside the run directory. The managed local SuperLink stays available between runs. Run this project in a dedicated environment/session; a previously running local SuperLink may have different dependency settings. Ray's Windows support is experimental; this Windows configuration was executed successfully.

Use a new output directory for every run. A failed Flower process or missing federated output marks the run failed and stops the comparison.

## Model and comparison

The federated task reuses the existing GRU as `GRU(sensors=1)`, with one sensor per example and shared weights across sensors. This avoids averaging input/output channel weights that refer to different sensor IDs on different clients. It models each sensor's history without cross-sensor features.

The centralized reference uses the same six sensors, per-sensor train-only scalers, windows, masks, model shape, initial tensors, batch size, optimizer settings, and total example exposure. Its three pooled epoch blocks match three federated rounds. Adam resets at each local fit and centralized block. FedAvg weights client models by usable training-window counts. Exact optimizer-step counts differ because each client has its own final partial batch.

These results are separate from the Phase 1 multivariate GRU results. See [the federated protocol and report](reports/federated-smoke/report.md) for architectural decisions, executed results, resource/communication definitions, and limitations.

## Temporal evaluation

Rows split chronologically into 70% training, 10% validation, and 20% test before making sequences. All client partitions share the same time grid and boundaries. Windows stay within their split. Inputs use 12 five-minute readings; predictions cover the next 12 steps.

Each sensor's mean and population standard deviation are fitted using observed training values only and retained locally. Inputs forward-fill within a split, then use the training mean for leading gaps. Zero/NaN targets are masked. Training windows with no observed future targets are excluded; validation and test preserve every window and score valid targets only.

Every round evaluates validation data. The lowest validation MAE selects a checkpoint; clients evaluate its test predictions only after selection. Global and per-horizon MAE/RMSE pool absolute errors, squared errors, and valid target counts. Client RMSEs are never averaged.

## Outputs

A completed federated run contains:

- `manifest.json`, `config.json`, `clients.json`: source/data/partition hashes, versions, partitions, time boundaries, normalization, and sample counts.
- `rounds.json`: client training statistics, validation metrics at every horizon, model hashes, and parameter communication estimates for every round.
- `federated.json`: selected round, global/client test metrics, and communication totals.
- `centralized-history.json`, `centralized.json`: matched reference training and selected test results, plus persistence.
- `initial.pt`, `round-*.pt`, `best-fedavg.pt`, `best-centralized.pt`: local checkpoints.
- `convergence.png`, `horizon-metrics.png`: validation convergence and selected-model test curves.
- `flower.log`, `launcher.json`, `status.json`: actual runtime logs, launch settings, and completion/failure state.

The run ID hashes config, raw-data checksum, source/tests/config files, and the dependency lock. Repeats compare numerical records, run IDs, partition hashes, and every checkpoint tensor. Framework run IDs and timestamps are expected to differ.

## Phase 1

```powershell
.\.venv\Scripts\python.exe -m tracast.cli inspect --out runs/inspect-new
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/phase1-smoke-new
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/subset.json --out runs/phase1-subset-new
```

The original implementation and reports are retained under the `phase1-verified` Git tag. The current package retains its behavior and tests. The original real-data smoke metrics were rerun and matched exactly. Historical JSON manifests remain verbatim execution records.

- [Phase 1 protocol](docs/phase1-protocol.md)
- [Phase 1 verification and results](reports/verification.md)
- [Data provenance, citation, and license notes](DATA.md)

Raw HDF5 files, client partitions, run directories, Flower bundles, and checkpoints are ignored by Git. Reports contain derived metrics and plots. GNN and sensor-failure experiments are outside this phase.
