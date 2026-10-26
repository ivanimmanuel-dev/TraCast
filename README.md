# TraCast

Traffic-speed forecasting on real METR-LA data. Phase one: persistence and a small PyTorch GRU on CPU.

## Setup

Python 3.11, Windows PowerShell, Git. Run from this repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The lock includes the tested CPU PyTorch build and all installed dependencies. It targets Python 3.11 on Windows x64. No CUDA setup is needed. The main dependency versions also live in `pyproject.toml`.

## Run

```powershell
.\.venv\Scripts\python.exe -m tracast.cli fetch
.\.venv\Scripts\python.exe -m tracast.cli inspect --out runs/inspect-new
.\.venv\Scripts\python.exe -m pytest -q -W error
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/smoke-new
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/subset.json --out runs/subset-new
```

Use a new output directory for each run. Existing runs are never overwritten. `fetch` reuses an existing file only after checking its SHA-256. The dataset stays in `data/raw/` and is ignored by Git.

The environment created during development is outside the repository. To use that existing environment from this directory, replace `.\.venv\Scripts\python.exe` with `..\..\work\venv\Scripts\python.exe`.

## Data

The pinned HDF5 file contains 34,272 rows and 207 sensor columns, March 1, 2012 00:00 through June 27, 2012 23:55, at five-minute intervals. Values are traffic speeds in mph. Timestamps have no timezone metadata; the loader preserves the recorded clock without guessing a timezone.

Download provenance, checksum, citation, and license details are in [DATA.md](DATA.md). The loader checks the checksum before reading HDF5, validates the complete frame, then selects the first N sensor columns in file order. Sensor IDs are logged. It rejects duplicate or unordered timestamps, gaps, duplicate IDs, nonnumeric values, negative speeds, and infinities. The old HDF5 frequency attribute is rebuilt from the actual timestamps for current pandas compatibility.

## Experiment protocol

- Split rows chronologically: 70% train, 10% validation, 20% test. Split before making windows. Every input and target window stays within its own split; no timestamps are shared across splits. The first 12 rows of each split provide history.
- Use 12 input steps and predict the next 12 steps directly: 60 minutes of history, forecasts every five minutes through 60 minutes. Windows have stride one and are sliced on demand rather than stored as expanded arrays.
- Treat zero and NaN as missing. Reports count each separately for every sensor and split. Forward-fill inputs using earlier readings within the same split. Fill remaining leading gaps with that sensor's training mean. No backward fill or interpolation.
- Fit one mean and population standard deviation per sensor using observed training values only. Constant sensors get a scale of one. A sensor with no observed training values stops the run.
- Missing targets stay masked in training and evaluation. Both models use the same target mask. The persistence baseline repeats the last filled speed across all forecast steps.
- GRU: one recurrent layer, 32 hidden units, linear head for all sensors and forecast steps. Adam, learning rate 0.001, gradient clipping at 5. Training loss is masked MAE in normalized units; selection uses validation MAE in mph.
- Save the checkpoint with the lowest validation MAE. Evaluate test data after checkpoint selection. Report MAE and RMSE in mph, pooled over valid forecast/sensor pairs and separately at every horizon. Accumulate absolute and squared errors and counts across batches; do not average batch RMSEs. Overlapping forecasts within a split are intentional rolling-origin evaluations.

This split protocol is stricter at boundaries than implementations that create windows before splitting. The sensor subset and protocol differ from published all-sensor benchmarks; compare the two local models under this protocol.

## CPU budget

The smoke config uses eight sensors, the first 2,016 timestamps (seven days), and two epochs. The subset config uses 16 sensors, the full date range, and five epochs. Both use batch size 64, four CPU threads, and zero loader workers. The GRU has about 11k parameters at 16 sensors. Reading the 57 MB HDF5 and keeping the selected arrays fits comfortably within a 16 GB machine; no expanded window cache is created. Peak process memory has not been measured.

## Reproduction

Python, NumPy, and PyTorch seeds are fixed. PyTorch deterministic algorithms are enabled; MKLDNN is disabled. Training shuffle uses a seeded generator. Checkpoints are selected only using validation data.

Each run contains:

```text
config.json       resolved settings
manifest.json     run ID, input/code hashes, versions, sensor IDs, split ranges, command
missing.json      selected-data and per-split missing-data reports
scaler.json       training means and scales
history.json      every epoch's training loss and validation metrics
best.pt           selected model state
metrics.json      persistence and GRU validation/test MAE, RMSE, counts, horizons
training_eda.png   training traces, speed histogram, missing fractions
status.json       completion state
```

The run ID hashes config, data checksum, source/tests/config files, and dependency lock. Manifest commands can differ because output paths differ. Exact reproducibility is checked by repeating the smoke run and comparing metrics, epoch histories, scaler values, run IDs, and model tensors. Other CPUs, operating systems, and PyTorch versions have not been checked for identical floating-point results.

## Layout

```text
configs/                 smoke and initial subset configs
src/tracast/data.py  download, checks, splits, scaling, windows
src/tracast/model.py GRU and metrics
src/tracast/experiment.py training, evaluation, logs, plots
src/tracast/cli.py   commands
tests/                  mechanics and regression tests
reports/                verified results and development notes
```

Tests use small constructed arrays to check mechanics. Research results in `reports/` come from the checksum-verified METR-LA file. Read [reports/verification.md](reports/verification.md) for executed checks, measured results, and remaining work.

Flower and GNNs are outside this phase.
