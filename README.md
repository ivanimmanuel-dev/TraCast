# TraCast

TraCast forecasts traffic speed 5 to 60 minutes ahead from the last hour of sensor readings. The data is METR-LA: 207 road sensors in Los Angeles, March to June 2012. Everything runs on a CPU laptop.

I wanted to know four things. Do recurrent models beat simple baselines? Does the road graph help? Does knowing the time of day help? And does federated training lose anything compared with training on all the data in one place? Then I scaled it up to all 207 sensors, put prediction intervals on the forecasts and checked what happens when traffic drifts.

[Technical report](reports/research-report.md) · [Two-page brief](reports/research-brief.pdf) · [Scaling](reports/scaling/report.md) · [Intervals](reports/uncertainty/report.md) · [Drift](reports/shift/report.md) · [Audit](docs/audit.md) · [Run guide](docs/reproducibility.md) · [Data sources](DATA.md) · [Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml)

## What I found

- **The clock helps more than anything else I tried.** Adding the time of day and a weekend flag took the GRU from 3.255 to 3.007 mph MAE, a change of -0.249 mph (95% CI -0.324 to -0.181), and at the 60-minute horizon from 3.889 to 3.391. Training longer helped too, but less: five epochs gave 3.409.
- **The road graph doesn't give a reliable gain.** The neighbor averages help a little without the clock (MAE -0.038 mph) and hurt a little with it (+0.031). The best three models are within 0.031 mph of each other, so I can't rank them.
- **Everything beats persistence (4.100).** The historical average (3.751) and a linear model (3.622) land in between, and the GRUs are well ahead of both. (At 15 minutes persistence is still ahead of the historical average.)
- **Federated averaging is about as good as training each client alone.** At 5 rounds with balanced groups, federated averaging gets 3.774 mph, local-only 3.796 and centralized 3.750. With congestion-grouped clients it is 3.836 against 3.769.
- **In the shared models the graph helps, at 5 of 5 sizes.** Adding the neighbor averages changes MAE by -0.057 to -0.029 mph. Without them a shared model sees nothing about its neighbors, so this is a clean test of the graph, which the dense models can't give.
- **Sharing the weights wins once the network is big.** At 128 and 207 sensors the shared GRU beats the dense GRU by 0.157 to 0.242 mph (the interval excludes zero), and at 16, 32 and 64 they are level. It does that with 4,044 parameters instead of 105,396.
- **Prediction intervals that are right on average can fail when it matters.** At a nominal 90%, split conformal covers 90.5% of the test speeds, but only 63% when the last speed was below 40 mph (against 93% in free flow). A quantile network with a conformal correction holds 88% there at the same average width.
- **Adapting a deployed model helps a little, and cheap adaptation can hurt.** Fine-tuning the output layer daily improves MAE by 0.031 mph and fine-tuning the whole network weekly by 0.042 (for a model trained on only the first 40% of the timeline: 0.072 and 0.188), at 3 s and 27 s for the 24 test days. Averaging past errors into the forecast was worse than doing nothing (+0.094 mph at best).

## Results

Test MAE in mph on 16 sensors, forecasting 12 steps (60 minutes) from the previous 12. The last 20% of the timeline (24 days) is the test set. Learned models are averaged over 5 seeds, with the standard deviation in parentheses. Persistence and the historical average aren't trained, so they run once. The [full table](reports/benchmark-extended/report.md) has RMSE, parameter counts and how much each score moves with the choice of test days.

| Model | MAE (SD) | 15 min | 60 min |
|---|---:|---:|---:|
| Persistence | 4.100 | 3.316 | 5.290 |
| Historical average | 3.751 | 3.751 | 3.752 |
| Linear | 3.622 (0.001) | 2.967 | 4.572 |
| GRU | 3.255 (0.026) | 2.805 | 3.889 |
| LSTM | 3.271 (0.013) | 2.845 | 3.885 |
| Graph-GRU | 3.193 (0.038) | 2.779 | 3.765 |
| GRU + calendar | 3.007 (0.025) | 2.730 | 3.391 |
| Graph-GRU + calendar | 3.016 (0.031) | 2.727 | 3.422 |

![Test error by forecast horizon; shading is seed SD](reports/figures/extended-horizons.png)

My first run trained every model for five epochs. In 12 of 12 runs the best validation epoch was the last one, so the models were still improving when training stopped. The numbers above come from a rerun with early stopping (up to 60 epochs) and 5 seeds. The five-epoch results are still in [reports/benchmark](reports/benchmark/report.md).

## Scaling to all 207 sensors

Everything above uses 16 sensors from one highway. I ran the same comparison on nested sets of 16, 32, 64, 128 and 207 sensors (each set contains the one before it), plus two new models that share one small GRU across every sensor, so their size doesn't depend on how many sensors there are. Test MAE in mph on the same first 16 sensors, whatever size the model was trained on:

| Trained on this many sensors | 16 | 32 | 64 | 128 | 207 |
|---|---:|---:|---:|---:|---:|
| Historical average | 3.751 | 3.751 | 3.751 | 3.751 | 3.751 |
| Dense GRU | 3.094 (0.012) | 3.199 (0.035) | 3.233 (0.013) | 3.453 (0.036) | 3.564 (0.017) |
| Dense Graph-GRU | 3.091 (0.036) | 3.147 (0.016) | 3.227 (0.019) | 3.509 (0.076) | 3.585 (0.038) |
| Shared GRU | 3.109 (0.010) | 3.195 (0.008) | 3.270 (0.011) | 3.324 (0.014) | 3.329 (0.005) |
| Shared Graph-GRU | 3.054 (0.013) | 3.089 (0.007) | 3.181 (0.019) | 3.252 (0.017) | 3.246 (0.006) |

![Accuracy by sensor count](reports/scaling/scaling-accuracy.png)

The graph helps the shared models at every size (-0.057 to -0.029 mph), which the dense models can't show because they already read every sensor. Sharing the weights also wins once the network is big: at 128 and 207 sensors the shared GRU beats the dense one by 0.157 to 0.242 mph with 4,044 parameters instead of 105,396. What it costs is compute: at 207 sensors the shared Graph-GRU needs 23.9 s per epoch against 2.1 s for the dense one. And more sensors did not make the first 16 better: trained on all 207, the dense models are +0.471 to +0.494 mph worse on those 16 targets, the shared ones +0.192 to +0.220.

The sizes' own-population scores, the cost in time and memory, and the limits are in the [scaling report](reports/scaling/report.md).

## Prediction intervals

A point forecast doesn't say how sure the model is, so I built 90% prediction intervals on the 16-sensor GRU and checked them on the held-out test days. There is a separate calibration part between validation and test that never touches training or the choice of checkpoint. Methods: a Gaussian baseline, split conformal, adaptive conformal (ACI), and a quantile network with a conformal correction (CQR). Averages over 3 seeds:

| Method | Coverage | Width (mph) | Coverage when last speed < 40 mph | Interval score |
|---|---:|---:|---:|---:|
| Gaussian, training residuals | 0.938 | 18.7 | 0.718 | 32.5 |
| Split conformal | 0.905 | 13.9 | 0.627 | 32.0 |
| Adaptive conformal (ACI) | 0.900 | 13.9 | 0.659 | 32.1 |
| Conformalized quantile (CQR) | 0.904 | 13.8 | 0.882 | 19.7 |

![Coverage by traffic condition](reports/uncertainty/uncertainty-conditions.png)

At a nominal 90%, split conformal covers 90.5% of the test speeds, calibrated across all twelve horizons, while the Gaussian baselines overshoot it (93.8%) and drift away from the diagonal at other levels. But when the last speed was below 40 mph (8% of the targets), split conformal covers 63% and ACI 66%, against 93% in free flow: one quantile for everything is too small exactly when traffic is bad. The quantile network with a conformal correction learns widths that depend on the traffic state and holds 88% there, at the same average width (13.8 against 13.9 mph) and with an interval score of 19.7 against 32.0.

The other methods, the reliability plot, coverage by horizon and day, and the limits are in the [intervals report](reports/uncertainty/report.md).

## Drift and adaptation

A model is trained once and used for months, so I replayed the test period one day at a time and let it adapt using only data that had already arrived (the tests check that nothing from the future gets in). Test MAE in mph, with the change against the static model in brackets, for the usual model and for a stale one trained on only the first 40% of the timeline:

| Strategy | Standard model | Stale model | Adaptation seconds (standard) |
|---|---:|---:|---:|
| Static model | 3.006 | 3.250 | 0 |
| Residual average | 3.176 (+0.169) | 3.407 (+0.157) | 1 |
| Residual average by time of day | 3.100 (+0.094) | 3.290 (+0.040) | 0 |
| Output layer fine-tuned daily | 2.976 (-0.031) | 3.179 (-0.072) | 3 |
| Whole network fine-tuned weekly | 2.965 (-0.042) | 3.062 (-0.188) | 27 |

![MAE change against the static model](reports/shift/shift-gain.png)

Fine-tuning the output layer daily improves the usual model's MAE by 0.031 mph and fine-tuning the whole network weekly by 0.042, and both intervals over test days exclude zero. For a stale model the gains are 0.072 and 0.188, so weekly fine-tuning wins back 77% of the 0.244 mph the stale model lost. The output-layer update costs 3 s over the 24 test days against 27 s for the weekly one. Averaging past errors into the forecast made things worse. The drift in these four months is mild, and the static model's error didn't grow with time since training, so the gains are small.

The data drift statistics, the regime breakdown and the limits are in the [drift report](reports/shift/report.md).

## How it's set up

The first 70% of the timeline is train, the next 10% validation and the last 20% test, and windows never cross a split boundary. Zero and missing readings are left out of the loss and the metrics, inputs are forward-filled within a split, and normalization uses training data only. The saved model is the one with the best validation MAE; test scores come after that.

The 16 sensors come from walking the published DCRNN distance graph from its first sensor, which gives 85 directed edges. Sensors and edge weights come from distances only, never from speeds. Graph-GRU gives each sensor its own speed plus the average of its upstream and downstream neighbors, passes that through a small shared layer, and feeds all 16 sensors into a GRU. "Self only" is the same model with the neighbor averages set to zero. One thing to keep in mind: every model here, the plain GRU too, reads all 16 sensors, so this tests the graph-weighted averages and not whether neighboring sensors matter.

What this doesn't cover: one 16-sensor subset, one 24-day test window, no hyperparameter search, and a federated part that is a simulation on one machine with a small univariate GRU.

## Federated learning

The Flower simulation gives each of three clients its own set of sensors. Every round each client trains its copy of the same GRU on its own readings, and the server averages the weights, weighted by how many training examples each client has (one example is one sensor's readings). Centralized trains the same model on all clients' data together, from the same starting weights, normalization, masks and number of examples. Local-only trains each client by itself with the same amount of local work and no communication.

Twelve sensors, fourteen days, three seeds, one local epoch per round. Sensors are ranked by how often their training speeds fall below 40 mph; balanced groups spread those ranks across the clients and congestion-grouped clients get consecutive blocks.

| Sensor grouping | Rounds | Federated MAE | Centralized MAE | Local-only MAE | Model weights sent |
|---|---:|---:|---:|---:|---:|
| Balanced | 1 | 4.047 | 3.792 | 4.003 | 180,288 B |
| Balanced | 3 | 3.809 | 3.765 | 3.826 | 450,720 B |
| Balanced | 5 | 3.774 | 3.750 | 3.796 | 721,152 B |
| Congestion-grouped | 5 | 3.836 | 3.750 | 3.769 | 721,152 B |

At 5 rounds with balanced groups, federated averaging gets 3.774 mph, local-only 3.796 and centralized 3.750. With congestion-grouped clients it is 3.836 against 3.769. The gaps are about the size of the seed spread, so in this setup training alone is about as good as federating.

Going from 1 to 5 rounds cuts balanced MAE by 0.273 mph and sends 4 times as many model-weight bytes. The byte counts cover weights sent for training, validation and the final test; message overhead isn't counted. More in the [federated findings](reports/federated-study/findings.md).

## Run

The recorded experiments ran on Python 3.11 and CPU PyTorch on Windows x64, and CI runs the tests on Windows and Ubuntu. From the repo folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m tracast.cli fetch
.\.venv\Scripts\python.exe -m pytest -q -W error
```

Then the experiments:

```powershell
# five-epoch benchmark and the longer-training study
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark.json --out runs/benchmark
.\.venv\Scripts\python.exe -m tracast.benchmark --config configs/benchmark-extended.json --out runs/benchmark-extended

# federated study
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study

# all 207 sensors, prediction intervals, drift (the scaling run takes a few hours)
.\.venv\Scripts\python.exe -m tracast.cli scaling --config configs/scaling.json --out runs/scaling
.\.venv\Scripts\python.exe -m tracast.cli uncertainty --config configs/uncertainty.json --out runs/uncertainty
.\.venv\Scripts\python.exe -m tracast.cli shift --config configs/shift.json --out runs/shift
```

On Linux or WSL use `python3.11` and `.venv/bin/python`. The installed `tracast` command runs everything too (`tracast --help` lists the subcommands). Use a fresh output folder for each run. The [run guide](docs/reproducibility.md) has the rest.

## What's next

- Give the bigger networks more capacity (a larger hidden size, a learned vector per sensor) and train them to convergence. The same-16 scores got worse as the population grew, and I held the capacity and the update budget fixed, so this is the first thing to try.
- Hold out sensors: train the shared model on some sensors and test it on ones it has never seen. Sharing weights makes that possible, and the dense models can't do it at all.
- Multi-hop or learned graph features in the shared model, since one-hop averages were enough to help but are the simplest choice.
- Intervals that are joint across horizons and sensors, and conformal methods with guarantees conditional on the traffic state instead of only on average.
- Federated learning with the shared model. Its size doesn't depend on the sensor count, which suits clients that own different sets of sensors.
- More data: another year or another city, and drift with real change points (roadworks, a sensor replaced) and a change detector that decides when to adapt.

## What's where

- `src/tracast/`: data loading, models (dense and shared-weight), the benchmark, scaling, conformal intervals, drift replay and the federated code
- `tests/`: unit tests on small constructed arrays, no real data needed
- `configs/`: one JSON file per experiment
- `reports/`: saved results for every experiment, the technical report and the brief
- `docs/`: the run guide and the audit of the code and data
- `tools/`: scripts that rebuild the README, figures and reports from the saved results

The code is MIT licensed. The data and the DCRNN graph have their own terms, see [DATA.md](DATA.md).
