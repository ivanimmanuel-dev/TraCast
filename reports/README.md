# Reports

Saved results for every experiment, plus the technical report and a two-page brief. Start with the [technical report](research-report.md) or the [brief](research-brief.pdf). The [run guide](../docs/reproducibility.md) has the commands to rerun things.

## Experiments

Each experiment has its own sensors and training budget, so compare models within a row, not across rows.

| Experiment | Setup | Start here |
|---|---|---|
| Five-epoch benchmark | 16 graph-picked sensors, full timeline, 5 epochs, 3 seeds | [Benchmark](benchmark/report.md) |
| Longer training, calendar inputs, baselines | Same sensors and timeline, up to 60 epochs with early stopping, 5 seeds | [Extended benchmark](benchmark-extended/report.md) |
| Scaling to all 207 sensors | Nested subsets of 16 to 207 sensors, dense and shared-weight models, 2 seeds | [Scaling report](scaling/report.md) |
| Shared model variants | LSTM cell and a learned vector per sensor in the shared models, 16 and 64 sensors, 2 seeds | [Variants report](variants/report.md) |
| Federated shared Graph-GRU on all 207 sensors | 8 geographic clients with different numbers of sensors and only their own part of the road graph, six arms (federated, pooled, local-only), 2 seeds | [Federated graph report](fedgraph/report.md) |
| DCRNN-style diffusion GRU | Centralized against the shared and dense models at 16 and 64 sensors, depth and direction ablation, federated and pooled at 64 sensors | [DCRNN report](diffusion/report.md) |
| FedAvg, FedProx, client distributions, local fine-tuning | 64 sensors, six clients cut by area, by congestion level or balanced, one seed (fine-tuning: two) | [FedProx report](fedopt/report.md) |
| Sensors the model never trained on | One held-out geographic area, zero-shot models and sensor-vector fixes, 2 seeds | [Held-out sensors report](holdout/report.md) |
| Prediction intervals | 90% intervals on the 16-sensor GRU, six interval methods plus a quantile network, 3 seeds | [Intervals report](uncertainty/report.md) |
| Drift and adaptation | Day-by-day replay of the test period with four adaptation strategies, two settings, 3 seeds | [Drift report](shift/report.md) |
| Federated study | 12 sensors, 14 days, 3 clients, 1/3/5 rounds, 3 seeds, with a local-only arm | [Findings](federated-study/findings.md) |
| Centralized GRU baseline | First 8 sensors for 7 days and 2 epochs, or first 16 sensors for the full timeline and 5 epochs; seed 42 | [Results](centralized-baseline.md) |
| Flower smoke run | First 6 sensors, 7 days, 3 rounds, 1 local epoch, seed 42 | [Results](federated-smoke/report.md) |
| Graph pilot | The 16 graph-picked sensors, 7 days, 1 epoch, seed 42 | [Results](graph-pilot/report.md) |

## What's in the folders

The benchmark's [results.json](benchmark/results.json) has every trial and its [summary.json](benchmark/summary.json) averages them by model. The [extended benchmark](benchmark-extended/summary.json) has the same layout plus `day_bootstrap` (test-day intervals and paired gaps). The federated study's [cases.json](federated-study/cases.json) has all twelve cases and its [summary.json](federated-study/summary.json) averages them by grouping and round budget. Each [case folder](federated-study/cases) keeps one run's files and its log. `b3-s42` means balanced groups, three rounds, seed 42; `c5-s7` means congestion groups, five rounds, seed 7.

| File | What it holds |
|---|---|
| `config.json` | The settings used |
| `manifest.json` | Source, dependency versions, data hashes and run ID |
| `protocol.json` | Sensor order, split boundaries, normalization and graph details |
| `history.json`, `centralized-history.json`, `rounds.json` | Training loss and validation error by epoch or round |
| `metrics.json`, `federated.json`, `centralized.json`, `local-only.json` | Scores of the selected models, by horizon, with target counts |
| `clients.json` | Each client's sensors, sample counts, normalization and speed distribution |
| `repeatability.json`, `determinism.json` | Comparisons of repeated runs |
| `status.json`, `flower.log` | Whether the run finished, and Flower's output |

The scaling folder has `scaling-summary.json` (the accuracy run), `cost-summary.json` (a separate short run with nothing else on the machine, which is where the cost numbers come from) and each size's manifest. The intervals and drift folders have `summary.json`, `config.json` and one result file per seed. [data-audit.json](data-audit.json) holds the facts about the data that the experiments rely on, and [docs/audit.md](../docs/audit.md) explains what was checked.

The federated folders (`fedgraph/`, `fedopt/`, `diffusion/federated/`) and `holdout/` hold each study's `summary.json`, `settings.json` and `launch.json` (which lists the files that weren't committed when the runs started), plus one folder per case under `cases/` with its result, manifest and setup. The raw run folders, client partitions and Flower logs stay in the ignored `runs/` folder. `diffusion/` also has the centralized and ablation summaries, and the 64-sensor part of the centralized comparison was assembled from the trials that finished before the run was stopped (its `status.json` says so).

A manifest describes the code at the time of its run, so its source hashes can differ from the current checkout. `$REPO` and `$PYTHON_ENV` stand in for local paths in saved records, and the [export receipts](export-provenance.json) keep the hashes from before the swap.

[Baseline preservation](baseline-preservation.json) checks the saved records against their source commit and maps later filename changes. The [baseline dependency lock](baseline-requirements-lock.txt) is the older centralized environment; new installs use the root [requirements-lock.txt](../requirements-lock.txt). The [CI receipt](ci.json) marks a completed clean-install test run, and [Actions](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml) shows the latest status. The `tests.txt` files inside experiment folders belong to those experiments.

Raw data, generated client arrays, checkpoints and Flower state stay in the ignored `data/` and `runs/` folders. What's committed here is the measurements.
