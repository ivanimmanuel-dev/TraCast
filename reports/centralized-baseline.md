# Centralized GRU baseline

The first experiment: persistence against a GRU that reads several sensors at once. It takes the first N sensor columns in file order, not the graph-picked sensors used later, with seed 42, 12 input and 12 forecast steps, batch size 64 and four CPU threads.

## Results

Test errors in mph over all valid sensor and forecast pairs, across all twelve horizons.

| Setup | Model | MAE | RMSE |
|---|---|---:|---:|
| 8 sensors, 7 days, 2 epochs | Persistence | 4.207292 | 7.577871 |
| 8 sensors, 7 days, 2 epochs | GRU | 5.647629 | 9.870540 |
| 16 sensors, full dates, 5 epochs | Persistence | 4.154315 | 7.842400 |
| 16 sensors, full dates, 5 epochs | GRU | 3.515679 | 6.719764 |

Two epochs on seven days isn't enough: the GRU is worse than persistence. With 16 sensors, the full timeline and five epochs it wins on both measures, though persistence is still slightly better at five minutes (RMSE 4.463 vs 4.590 mph). Epoch 5 had the best validation MAE. The smoke run repeated exactly, with the same metrics, histories, scalers, run ID and checkpoint tensors.

## Protocol

Rows are split in time before windows are built, so every input and target stays inside its split. Per-sensor means and standard deviations come from observed training readings, inputs are forward-filled within a split (training means for leading gaps) and zero/NaN targets are masked. Training minimizes normalized MAE with Adam at 0.001 and gradient clipping at 5.

| Split | First timestamp | Last timestamp | Rows | Windows |
|---|---|---|---:|---:|
| Train | 2012-03-01 00:00 | 2012-05-23 07:05 | 23,990 | 23,967 |
| Validation | 2012-05-23 07:10 | 2012-06-04 04:40 | 3,427 | 3,404 |
| Test | 2012-06-04 04:45 | 2012-06-27 23:55 | 6,855 | 6,832 |

The 16-sensor GRU has 11,136 parameters. These single-seed runs didn't record timing or peak memory; the [multi-seed benchmark](benchmark/report.md) does.

## Reproduce

After the [setup](../docs/reproducibility.md):

```powershell
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/smoke-new
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/subset.json --out runs/subset-new
```

[Smoke records](smoke) · [Full-date records](subset) · [Data sources](../DATA.md)
