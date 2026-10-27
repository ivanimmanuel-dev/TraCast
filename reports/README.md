# Experiment reports

The [technical report](research-report.md) explains the methods and results. The [two-page brief](research-brief.pdf) summarizes the main comparisons. To reproduce an experiment, use the [run guide](../docs/reproducibility.md).

## Studies

Each study has its own sensor selection and training budget. Compare models within the same row of this table.

| Study | Data and training | Read first |
|---|---|---|
| Recurrent models and graph features | 16 sensors selected from the distance graph; full March–June timeline; 5 epochs; 3 seeds | [Benchmark](benchmark/report.md) |
| Federated rounds and sensor grouping | 12 sensors in file order; 14 days; 3 clients; 1/3/5 rounds; 3 seeds | [Findings](federated-study/findings.md) |
| Centralized GRU baseline | First 8 sensors for 7 days and 2 epochs, or first 16 sensors for the full timeline and 5 epochs; seed 42 | [Results](centralized-baseline.md) |
| Three-client Flower baseline | First 6 sensors; 7 days; 3 rounds; 1 local epoch; seed 42 | [Results](federated-smoke/report.md) |
| One-epoch graph comparison | Same 16 graph-selected sensors; 7 days; seed 42 | [Results](graph-pilot/report.md) |

## Reading the records

The benchmark's [results.json](benchmark/results.json) contains individual trials; its [summary.json](benchmark/summary.json) aggregates them by model. The federated study's [cases.json](federated-study/cases.json) collects all twelve experiments; its [summary.json](federated-study/summary.json) aggregates them by sensor grouping and round budget. The [case folders](federated-study/cases) retain each run's individual files and runtime log. For example, `b3-s42` means balanced groups, three rounds, and seed 42; `c5-s7` means congestion-grouped sensors, five rounds, and seed 7.

| Record | Purpose |
|---|---|
| `config.json` | Stores the settings used for the run. |
| `manifest.json` | Identifies the executed source, dependency versions, data hashes, and run ID. |
| `protocol.json` | Records sensor order, chronological boundaries, normalization statistics, and graph construction where applicable. |
| `history.json`, `centralized-history.json`, `rounds.json` | Records training loss and validation errors by epoch or round. |
| `metrics.json`, `federated.json`, `centralized.json` | Stores selected-model scores, forecast horizons, and valid target counts. |
| `clients.json` | Records each client's sensors, sample counts, normalization, and training distribution where available. |
| `repeatability.json`, `determinism.json` | Records comparisons of repeated runs. |
| `status.json`, `flower.log` | Records completion status and the Flower runtime output. |

Record names vary between runners. A manifest describes the code used when that experiment ran, so its source hashes can differ from the current checkout. `$REPO` and `$PYTHON_ENV` replace local filesystem paths in exported records. [Export receipts](export-provenance.json) retain the corresponding content hashes.

[Baseline preservation](baseline-preservation.json) checks saved records against their source commit and maps any later filename changes. The [baseline dependency lock](baseline-requirements-lock.txt) records the older centralized environment; new installations use the root [requirements-lock.txt](../requirements-lock.txt). The [CI receipt](ci.json) identifies a completed clean-install test run, while [current Actions runs](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml) show the latest status. Test transcripts inside study folders belong to those experiments.

Raw datasets, generated client arrays, checkpoints, and Flower service state are stored locally under ignored data and run directories. The committed records contain the measurements used in the reports.
