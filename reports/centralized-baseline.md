# Centralized GRU baseline

Persistence and a multivariate GRU on the first sensor columns in METR-LA file order. Both runs use seed 42, 12 input steps, 12 forecast steps, batch size 64, and four CPU threads. These sensor subsets differ from the topology-selected benchmark.

## Results

Test errors in mph, pooled over valid sensor/forecast pairs across all twelve horizons.

| Configuration | Model | MAE | RMSE |
|---|---|---:|---:|
| 8 sensors, 7 days, 2 epochs | Persistence | 4.207292 | 7.577871 |
| 8 sensors, 7 days, 2 epochs | GRU | 5.647629 | 9.870540 |
| 16 sensors, full dates, 5 epochs | Persistence | 4.154315 | 7.842400 |
| 16 sensors, full dates, 5 epochs | GRU | 3.515679 | 6.719764 |

The two-epoch GRU trails persistence. The five-epoch GRU improves pooled MAE and RMSE; its five-minute RMSE is 4.590 mph versus 4.463 for persistence. Epoch 5 was selected by validation MAE. The smoke run was repeated with identical metrics, histories, scalers, run ID, and checkpoint tensors.

## Protocol

Rows split chronologically before window construction. Every input and target remains within its split. Observed training readings determine per-sensor means and population standard deviations. Inputs forward-fill within a split, with training means for leading gaps. Zero/NaN targets are masked. Training minimizes normalized MAE with Adam at 0.001 and gradient clipping at 5.

| Split | First timestamp | Last timestamp | Rows | Windows |
|---|---|---|---:|---:|
| Train | 2012-03-01 00:00 | 2012-05-23 07:05 | 23,990 | 23,967 |
| Validation | 2012-05-23 07:10 | 2012-06-04 04:40 | 3,427 | 3,404 |
| Test | 2012-06-04 04:45 | 2012-06-27 23:55 | 6,855 | 6,832 |

The 16-sensor GRU has 11,136 parameters. These single-seed runs did not record isolated timing or peak memory. The [multi-seed benchmark](benchmark/report.md) records both for the topology-selected subset.

## Reproduce

After [environment setup](../docs/reproducibility.md):

```powershell
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/smoke-new
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/subset.json --out runs/subset-new
```

[Smoke records](smoke) · [Full-date records](subset) · [Dataset provenance](../DATA.md)
