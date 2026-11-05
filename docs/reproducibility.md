# Run guide

## Setup

The recorded experiments used Python 3.11.9, CPU PyTorch 2.5.1, Flower 1.42.0 and Ray 2.55.1 on Windows x64, with everything pinned in `requirements-lock.txt`. CI runs the tests on Windows and Ubuntu. On Linux use `python3.11` and `.venv/bin/python`.

From the repo folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -W error
.\.venv\Scripts\python.exe -m tracast.cli fetch
```

Downloads are checked against pinned SHA-256 hashes, and the graph benchmark also fetches the DCRNN distance files. [DATA.md](../DATA.md) has the sources. Raw data, client partitions, checkpoints and Flower state are gitignored. Every run needs a new output folder.

`tracast --help` lists the subcommands. `tracast benchmark ...` is the same as `python -m tracast.benchmark ...`, and likewise for `federated`, `study` and `compare`.

## Looking at the data

```powershell
.\.venv\Scripts\python.exe -m tracast.cli inspect --out runs/inspection
```

This validates the whole traffic table, counts missing readings per sensor and plots training traces, the speed distribution and the missing fractions. Zeros and NaNs count as missing, and timestamps keep the clock the source recorded.

## Centralized experiments

### Smoke runs

A seven-day GRU run and a one-epoch graph comparison, to check that everything works:

```powershell
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/gru-smoke
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/graph-pilot.json --out runs/graph-smoke
```

### Five-epoch benchmark

16 sensors picked from the road graph, all 34,272 timestamps, seeds 7, 42 and 2026, five epochs. Persistence, GRU, LSTM, Graph-GRU and the self-only version share the sensor order, split, scalers, masks, batch size and data order. Each model and seed runs in a fresh process, one after another.

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark
```

The best-validation checkpoint is scored on the test set. Each trial saves its results, epoch history, protocol, status and checkpoint; the parent folder has the resolved config, source/data/graph hashes, versions, summary, paired differences and plots.

To repeat one trial:

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --trial graph_gru --seed 42 --out runs/graph-repeat
```

Compare the test metrics, validation histories and `state_sha256`. The seed-42 Graph-GRU and LSTM repeats matched exactly ([repeatability.json](../reports/benchmark/repeatability.json)); timing and memory vary from run to run.

### Longer training, calendar inputs, baselines

Five epochs is too short (validation error is still falling), so [benchmark-extended.json](../configs/benchmark-extended.json) trains longer. Same sensors, split, windows and loss.

```powershell
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark-extended.json --out runs/benchmark-extended
```

| Setting | What it does |
|---|---|
| `epochs`, `patience` | Stop after `patience` epochs with no validation-MAE gain, or at `epochs`. The best-validation checkpoint is the one scored. `patience: null` (the default) runs every epoch. |
| `scheduler`, `scheduler_patience` | `plateau` halves the learning rate after `scheduler_patience` epochs with no gain. `none` (the default) keeps it fixed. |
| `_time` suffix on a model name | Adds time of day (sine, cosine) and a weekend flag to every input step. Without the suffix the batch is cut down to speeds. |
| `ha` | Historical average: training mean per sensor, weekday and 5-minute slot. Slots never seen fall back to the time-of-day mean, then the sensor mean. Uses no recent readings. |
| `linear` | One linear layer from the flattened input window to all forecasts, same loss. |

With these left at their defaults the original protocol is unchanged: the seed-42 GRU and Graph-GRU runs from `configs/benchmark.json` give the recorded test MAE and `state_sha256`.

Each trial also saves per-day error sums for the test period. `summary.json` uses them to resample test days (training seeds averaged first) for MAE intervals and paired gaps between models. That covers which days were held out; the seed SD covers initialization and shuffling.

## Bigger networks, uncertainty and drift

### Scaling to all 207 sensors

The same benchmark at 16, 32, 64, 128 and 207 sensors. The sensors are prefixes of one graph ordering, so each size contains the one before it, and every model gets the same number of updates at every size.

```powershell
.\.venv\Scripts\python.exe -m tracast.cli scaling --config configs/scaling.json --out runs/scaling
```

Sizes that already finished are skipped when you rerun into the same folder, so it can be stopped and resumed. `scaling-summary.json` has every model and size in one table. It scores each model twice: on all the sensors it was trained on (these sets differ by size, so they can't be compared across sizes) and on the same first 16 sensors (the same targets at every size, so this is the comparison that says whether more sensors help). The `s*` model names (`sgru`, `slstm`, `sgraph`) share one recurrent net across sensors; see [shared.py](../src/tracast/shared.py). `train_stride` makes an epoch a random 1/k of the training windows. A `_id` suffix (`sgru_id`, `sgraph_id`) adds a learned vector per sensor to a shared model.

The cost numbers in the scaling report come from a separate short run with nothing else on the machine:

```powershell
.\.venv\Scripts\python.exe -m tracast.cli scaling --config configs/scaling-cost.json --out runs/scaling-cost
```

### Shared model variants

The LSTM cell and the per-sensor vectors, tested one at a time against the shared GRU and Graph-GRU at 16 and 64 sensors. The two reference models are trained again in the same run, so every gap is measured on identical windows.

```powershell
.\.venv\Scripts\python.exe -m tracast.cli scaling --config configs/variants.json --out runs/variants
```

### Prediction intervals

```powershell
.\.venv\Scripts\python.exe -m tracast.cli uncertainty --config configs/uncertainty.json --out runs/uncertainty
```

One point forecaster per seed, then every interval method built from it. The split is train 70%, validation 5% (picks the checkpoint), calibration 5% (sits just before the test period, never used for training or picking a checkpoint) and test 20%. Methods are described at the top of [uncertainty.py](../src/tracast/uncertainty.py), and the guarantees and what breaks them are in [conformal.py](../src/tracast/conformal.py). `gamma_sweep` in the config lists the ACI step sizes that are also evaluated, as a sensitivity check next to the preset `gamma` used in the main table.

### Drift and adaptation

```powershell
.\.venv\Scripts\python.exe -m tracast.cli shift --config configs/shift.json --out runs/shift
```

Replays the test period one day at a time. A strategy may only use windows whose last target has already been observed when the day starts (`known_windows` in [shift.py](../src/tracast/shift.py)), and the tests corrupt the not-yet-arrived targets to check that. The cheap strategies' settings are picked by replaying validation, never test.

## Federated experiments

### Flower smoke run

Two sensors on each of three clients, seven days, three rounds with one local epoch each. Running it twice and comparing checks that it repeats exactly:

```powershell
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-a
.\.venv\Scripts\python.exe -m tracast.federated --config configs/federated-smoke.json --out runs/flower-b
.\.venv\Scripts\python.exe -m tracast.compare_runs runs/flower-a runs/flower-b
```

The runner starts Flower's `ClientApp`, `ServerApp` and `FedAvg` through `flwr run --stream`, a local SuperLink and Ray, using the installed environment, one active client actor, two reserved Ray CPUs, one PyTorch thread and a 128 MiB object store. Ray on Windows is experimental. Use a dedicated environment so an existing SuperLink doesn't keep another project's dependency settings.

Clients hold disjoint sensors on the same host, and each sensor is normalized with its own training statistics. The pooled reference uses the same initial weights, examples, masks, exposure and optimizer-reset schedule. Partial last batches give different optimizer-step counts. The shared model forecasts one sensor per example.

### Round budgets and client distributions

```powershell
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study
```

Twelve cases, one after another, on twelve sensors and fourteen days:

| Partition | Round budgets | Seeds |
|---|---|---|
| Balanced congestion ranks | 1, 3, 5 | 7, 42, 2026 |
| Congestion-grouped ranks | 5 | 7, 42, 2026 |

Sensors are ranked by the fraction of observed training speeds below 40 mph. Balanced groups interleave the ranks and congestion groups take consecutive blocks. Each client owns four sensors, all cases share the same time boundaries, and ranking, scaling and the speed histograms use training rows only.

Each case also trains a local-only arm. Every client starts from the shared initial weights, does the same local work as in federated training (same seeds, fresh optimizer each round) and never communicates. It keeps the round with its own best validation MAE, and the clients' test statistics are pooled so the score compares with the federated and pooled ones.

The pooled data keeps the canonical sensor order for both partition rules. The study checks that the pooled results and model hashes match for equal budgets, and that the models after one and three rounds match the first rounds of the five-round runs. The whole study takes about 14 minutes on a Ryzen 7 5825U. [Results](../reports/federated-study/findings.md).

## What each measurement means

| Measurement | Definition |
|---|---|
| MAE / RMSE | Error sums pooled over valid forecast/sensor pairs, in mph; also given per horizon |
| Training time | Update loops including batch loading, excluding validation |
| Evaluation time | The whole scoring loop, including inverse normalization and metrics |
| Forward inference | Median of 30 normalized forward passes after 5 warmups on the first test batch, per window |
| Peak memory | Windows PeakWorkingSetSize of the whole fresh process, including loading and preprocessing |
| Flower wall time | Launch, simulation, test scoring, the pooled and local-only models, and plots |
| Communication | Raw model bytes for training downloads and uploads, validation downloads and the final test download |

Communication leaves out serialization, metrics, config and transport overhead. Timings include whatever else the machine was doing. The seed SD shows initialization and shuffling variation on fixed dates.

## Rebuilding the reports

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe tools/presentation.py
.\.venv\Scripts\python.exe tools/pages.py
.\.venv\Scripts\python.exe tools/architecture.py
.\.venv\Scripts\python.exe tools/data_audit.py
.\.venv\Scripts\python.exe tools/federated_page.py
.\.venv\Scripts\python.exe tools/build_research.py
```

`pages.py` writes the scaling, shared-variants, intervals, drift and audit pages, `architecture.py` draws the shared-model schematic and `data_audit.py` needs the raw data to recompute the data facts. Run `pages.py` before `presentation.py`, because the README reads what it writes. These read the saved results and rebuild the README, the figures, the federated findings page, the technical report and the two-page PDF. They don't train anything. `tools/export_records.py` swaps local paths for `$REPO` and `$PYTHON_ENV` in saved records, and the export receipt keeps the before and after hashes. A manifest describes the code at the time of its run, so its source hashes can differ from the current checkout.

## Configs

| Config | Experiment |
|---|---|
| [smoke.json](../configs/smoke.json) | Eight-sensor GRU, two epochs, seven days |
| [subset.json](../configs/subset.json) | Sixteen-sensor GRU, five epochs, full timeline |
| [graph-pilot.json](../configs/graph-pilot.json) | GRU vs Graph-GRU after one epoch on seven days |
| [benchmark.json](../configs/benchmark.json) | Recurrent models and the graph ablation, three seeds |
| [benchmark-extended.json](../configs/benchmark-extended.json) | Longer training with early stopping, calendar inputs, a historical average and a linear model, five seeds |
| [scaling.json](../configs/scaling.json) | 16 to 207 sensors, dense and shared models, two seeds |
| [scaling-cost.json](../configs/scaling-cost.json) | The same sizes and models for 3 epochs and one seed, run alone to measure time and memory |
| [variants.json](../configs/variants.json) | Shared LSTM and shared models with a vector per sensor, at 16 and 64 sensors, two seeds |
| [uncertainty.json](../configs/uncertainty.json) | Prediction intervals for the dense GRU with calendar inputs, three seeds |
| [shift.json](../configs/shift.json) | Static model vs three adaptation strategies, standard and stale settings, three seeds |
| [federated-smoke.json](../configs/federated-smoke.json) | Three Flower rounds and a matched centralized GRU |
| [federated-study.json](../configs/federated-study.json) | Round budgets and client groupings, three seeds |

## Source map

| Files | What they do |
|---|---|
| [data.py](../src/tracast/data.py) | Fetch and validate METR-LA, split by time, fit scalers, build windows |
| [model.py](../src/tracast/model.py), [recurrent.py](../src/tracast/recurrent.py) | GRU, LSTM, linear model and the error accumulators |
| [baselines.py](../src/tracast/baselines.py), [stats.py](../src/tracast/stats.py) | Historical average; test-day resampling for intervals |
| [provenance.py](../src/tracast/provenance.py) | Git commit and whether the working tree had changes |
| [cli.py](../src/tracast/cli.py) | The `tracast` command |
| [shared.py](../src/tracast/shared.py) | Recurrent forecasters that share weights across sensors, with and without graph neighbors |
| [scaling.py](../src/tracast/scaling.py) | Runs the benchmark at several sensor counts and builds the comparison tables and figures |
| [conformal.py](../src/tracast/conformal.py), [uncertainty.py](../src/tracast/uncertainty.py) | Conformal interval methods and the experiment that compares them with baselines |
| [shift.py](../src/tracast/shift.py) | Day-by-day replay with adaptation strategies, plus drift diagnostics |
| [topology.py](../src/tracast/topology.py), [graph_model.py](../src/tracast/graph_model.py) | Build the distance graph and mix neighbor readings into the GRU |
| [experiment.py](../src/tracast/experiment.py), [benchmark.py](../src/tracast/benchmark.py) | Train and score the centralized models and save the results |
| [federated_task.py](../src/tracast/federated_task.py) | Sensor ownership, local datasets, shared training, pooled metrics |
| [flower_client.py](../src/tracast/flower_client.py), [flower_server.py](../src/tracast/flower_server.py) | Flower messages, aggregation and per-round records |
| [federated.py](../src/tracast/federated.py), [federated_study.py](../src/tracast/federated_study.py) | Launch simulations, compare round budgets and groupings |
| [compare_runs.py](../src/tracast/compare_runs.py) | Compare two federated runs, including saved checkpoint tensors |

The [report index](../reports/README.md) explains what's in `reports/`.
