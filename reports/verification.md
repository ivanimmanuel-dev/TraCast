# Verification: October 26, 2026

Phase one runs on the real checksum-verified METR-LA HDF5 file.

## Environment

- Windows x64; Python 3.11.9; PyTorch 2.5.1+cpu.
- Exact installed dependency versions: `requirements-lock.txt`.
- Packages installed into the workspace's `work/venv` environment; TraCast installed editable.
- `python -m pip check`: no broken requirements.
- CPU-only runs with four threads and no DataLoader workers.
- Hardware model and installed RAM were not independently read: Windows denied the hardware query. The initial sizing target is the supplied Ryzen 7 / 16 GB laptop.

## Executed checks

- Downloaded 57,038,056 bytes from the immutable mirror revision; SHA-256 matches the published file hash.
- Validated all 34,272 timestamps and 207 numeric sensor columns before selecting sensors.
- Continuous five-minute index, March 1 through June 27, 2012; no NaNs, 575,302 zeros, 8.109351% zero-or-NaN values.
- `python -m pytest -q -W error`: **26 passed**, no warnings.
- Generated training-only plots and visually inspected the smoke plot.
- Completed the real-data smoke run twice and the full-date-range 16-sensor run once.
- Repeated smoke run: config, missing report, scaler, epoch histories, metrics, completion record, run ID, and every saved model tensor agree exactly. Checkpoint serialization bytes were not used as the comparison.
- Raw data, download sidecar, run directories, and checkpoints are excluded by `.gitignore`. Small JSON reports and plots are retained here.

## Measured test results

Metrics below are pooled across all 12 forecast horizons, weighted by observed target counts. Units: mph. These are local subset experiments, with no hyperparameter search.

| Experiment | Model | MAE | RMSE |
| --- | --- | ---: | ---: |
| Smoke: 8 sensors, 7 days, 2 epochs | persistence | 4.207292 | 7.577871 |
| Smoke: 8 sensors, 7 days, 2 epochs | gru | 5.647629 | 9.870540 |
| Subset: 16 sensors, full dates, 5 epochs | persistence | 4.154315 | 7.842400 |
| Subset: 16 sensors, full dates, 5 epochs | gru | 3.515679 | 6.719764 |

The smoke GRU performs worse than persistence after two epochs. The full-date-range subset GRU improves pooled MAE and RMSE after five epochs. At the five-minute horizon its RMSE is slightly worse than persistence (4.590 vs 4.463 mph). Full per-horizon values and validation histories are saved with each report. The selected subset checkpoint is epoch 5.

## Full-date-range boundaries

| Split | First timestamp | Last timestamp | Rows | Windows |
| --- | --- | --- | ---: | ---: |
| Train | 2012-03-01 00:00 | 2012-05-23 07:05 | 23,990 | 23,967 |
| Validation | 2012-05-23 07:10 | 2012-06-04 04:40 | 3,427 | 3,404 |
| Test | 2012-06-04 04:45 | 2012-06-27 23:55 | 6,855 | 6,832 |

The 16-sensor GRU contains 11,136 trainable parameters. Sensor IDs are listed in the manifests and scaler reports.

## Commands executed

From the TraCast repository directory, using the development environment:

```powershell
..\..\work\venv\Scripts\python.exe -m pytest -q -W error
..\..\work\venv\Scripts\python.exe -m tracast.cli inspect --out runs/inspection-verified
..\..\work\venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/smoke-verified
..\..\work\venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/smoke-repeat
..\..\work\venv\Scripts\python.exe -m tracast.cli run --config configs/subset.json --out runs/subset
```

The initial pre-fix smoke and inspection attempts failed while plotting legacy HDF5 frequency metadata. Rebuilding the index from its timestamp values fixed the issue; a regression test covers it. The initial smoke directory is marked failed. Verified reports come only from completed post-fix runs. The raw file was not changed.

## Remaining checks

- No 207-sensor training run, seed sweep, hyperparameter search, uncertainty intervals, or external benchmark comparison.
- Peak memory and standalone runtime were not measured. Smoke repeat and subset training overlapped, so their wall times are not recorded as benchmarks.
- Exact floating-point reproducibility across other machines and operating systems is unverified.
- A complete clean-environment reinstall from the lock has not been tested; dependency consistency and the installed editable package were checked.
- The original Google Drive file was not independently compared to the mirror. Original-provider dataset-specific licensing and the mirror's relicensing authority remain unverified; see `DATA.md`.
- Flower and GNNs are not implemented.

Next iteration: keep this baseline fixed, repeat the 16-sensor experiment over several seeds, then inspect validation errors by horizon and sensor before changing the model.
