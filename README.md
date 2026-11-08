# TraCast

**Traffic-speed forecasting on METR-LA, from recurrent baselines to federated graph models, on a CPU.**

[![Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml/badge.svg)](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml) ![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![MIT license](https://img.shields.io/badge/license-MIT-green)

TraCast forecasts the speed on 207 Los Angeles road sensors 5 to 60 minutes ahead from the last hour of readings (METR-LA, March to June 2012). It started with four questions: do recurrent models beat simple baselines, does the road graph help, does the time of day help, and does federated training lose anything against pooling the data? It grew into scaling to all 207 sensors, prediction intervals, drift, forecasting for sensors a model never saw, a DCRNN-style diffusion model, and federated learning on the whole network.

Every number in the reports is generated from saved results, a comparison is only called better or worse when an interval over test days says so, and the tests include leakage checks.

**Start with** the [technical report](reports/research-report.md) or the [two-page brief](reports/research-brief.pdf). **Studies:** [scaling](reports/scaling/report.md) · [shared variants](reports/variants/report.md) · [intervals](reports/uncertainty/report.md) · [drift](reports/shift/report.md) · [federated graph](reports/fedgraph/report.md) · [DCRNN](reports/diffusion/report.md) · [FedProx](reports/fedopt/report.md) · [new sensors](reports/holdout/report.md). **Also:** [audit](docs/audit.md) · [run guide](docs/reproducibility.md) · [data sources](DATA.md).

## At a glance

- **Does the time of day help?** More than anything else I tried. Adding it, with a weekend flag, takes the GRU from 3.255 to 3.007 mph MAE (-0.249 mph, 95% CI -0.324 to -0.181), and every model beats persistence (4.100). ([benchmark](reports/benchmark-extended/report.md))
- **Does the road graph help?** Hardly in the dense models: the three best 16-sensor models are within 0.031 mph of each other. In a shared-weight model, where the graph is the only view of the neighbors, it helps at all 5 sizes (-0.057 to -0.029 mph). ([scaling](reports/scaling/report.md))
- **What changes at 207 sensors?** Sharing one small GRU across all sensors takes over: at 128 and 207 sensors it beats the dense GRU by 0.157 to 0.242 mph with 4,044 parameters instead of 105,396. Giving each sensor a learned 4-number vector helps more than the graph does (MAE -0.208 to -0.107 mph, against -0.055 to -0.050 for the graph). ([scaling](reports/scaling/report.md), [variants](reports/variants/report.md))
- **Can the uncertainty be trusted?** On average, yes: 90% conformal intervals cover 90.5% of the test speeds. In congestion, no: 63% when the last speed was under 40 mph, against 93% in free flow. A quantile network with a conformal correction holds 88% there at the same average width. ([intervals](reports/uncertainty/report.md))
- **What happens when traffic drifts?** Not much over these four months, so adapting helps a little: fine-tuning the output layer daily gains 0.031 mph and the whole network weekly 0.042 (0.072 and 0.188 for a model trained on only the first 40% of the timeline). Averaging past errors into the forecast makes it worse. ([drift](reports/shift/report.md))
- **Does federated learning cost accuracy?** On all 207 sensors, split into 8 geographic clients that each see only their own part of the road graph, the federated shared Graph-GRU scores 3.672 mph (the historical average gets 4.193). Federating costs 0.076 mph against pooling the data, and every client training alone does 0.161 mph better. In a 64-sensor study, fine-tuning the federated model on each client closed that gap. ([federated graph](reports/fedgraph/report.md), [FedProx](reports/fedopt/report.md))
- **Is a diffusion model worth it?** Centrally, yes. A DCRNN-style diffusion GRU beats the shared Graph-GRU by 0.119 mph at 16 sensors and 0.174 at 64 (one seed there), and a shuffled graph costs it 0.090, so it does use the road structure. Federated, the advantage reverses: it is 0.026 mph worse than the federated Graph-GRU. ([DCRNN](reports/diffusion/report.md))
- **Can it forecast a sensor it never trained on?** Yes. On 14 held-out sensors with only their first week of data, the shared Graph-GRU scores 3.867 mph against 3.759 when it trained on them. The sensor-vector model has no vector for a new sensor; fitting one on that first week gives 3.688 mph, 0.179 better than the zero-shot Graph-GRU, while predicting the vector from sensor descriptors didn't work. ([held-out sensors](reports/holdout/report.md))
- **Does FedProx help?** Not here: it was worse than FedAvg in all 6 settings I ran, and the penalty grew with mu. Five local epochs per round, though, cut communication about 5 times for 0.030 mph (64 sensors, six clients, one seed). ([FedProx](reports/fedopt/report.md))

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q -W error
.\.venv\Scripts\python.exe -m tracast.cli fetch
.\.venv\Scripts\python.exe -m tracast.cli run --config configs/smoke.json --out runs/gru-smoke
```

The recorded experiments ran on Python 3.11 and CPU PyTorch on Windows x64, and CI runs the tests on Windows and Ubuntu (on Linux use `python3.11` and `.venv/bin/python`). The last command trains one GRU on seven days of data as a smoke test. `tracast --help` lists every experiment, and the [run guide](docs/reproducibility.md) has the command for each, from the five-epoch benchmark to the multi-hour federated runs. Use a fresh output folder for each run.

## How the experiments are set up

The first 70% of the timeline is training, the next 10% validation and the last 20% (24 days) test, and windows never cross a split boundary. Zero and missing readings are left out of the loss and the scores, inputs are forward-filled within a split, and normalization uses training data only. The saved model is the one with the best validation MAE, and the test set is scored after that. Gaps between models are measured on identical windows, with 95% intervals from resampling test days.

The early studies use 16 sensors picked by walking the published DCRNN distance graph from its first sensor, which gives 85 directed edges. Sensors and edge weights come from distances only, never from speeds. Graph-GRU gives each sensor its own speed plus the average of its upstream and downstream neighbors, passes that through a small shared layer and feeds all 16 sensors into a GRU; "self only" is the same model with the neighbor averages set to zero. Every model in those studies, the plain GRU included, reads all 16 sensors, so they test the graph-weighted averages and not whether neighboring sensors matter. The shared-weight models further down take that confound away.

## Baselines and the clock (16 sensors)

Test MAE in mph, forecasting 12 steps (60 minutes) from the previous 12. Learned models are averaged over 5 seeds, with the standard deviation in parentheses. Persistence and the historical average aren't trained, so they run once. The [full table](reports/benchmark-extended/report.md) has RMSE, parameter counts and how much each score moves with the choice of test days.

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

## A first federated study (12 sensors)

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

## Scaling to all 207 sensors

The studies above use 12 or 16 of the 207 sensors. I ran the same comparison on nested sets of 16, 32, 64, 128 and 207 sensors (each set contains the one before it), plus two new models that share one small GRU across every sensor, so their size doesn't depend on how many sensors there are. Test MAE in mph on the same first 16 sensors, whatever size the model was trained on:

| Trained on this many sensors | 16 | 32 | 64 | 128 | 207 |
|---|---:|---:|---:|---:|---:|
| Historical average | 3.751 | 3.751 | 3.751 | 3.751 | 3.751 |
| Dense GRU | 3.094 (0.012) | 3.199 (0.035) | 3.233 (0.013) | 3.453 (0.036) | 3.564 (0.017) |
| Dense Graph-GRU | 3.091 (0.036) | 3.147 (0.016) | 3.227 (0.019) | 3.509 (0.076) | 3.585 (0.038) |
| Shared GRU | 3.109 (0.010) | 3.195 (0.008) | 3.270 (0.011) | 3.324 (0.014) | 3.329 (0.005) |
| Shared Graph-GRU | 3.054 (0.013) | 3.089 (0.007) | 3.181 (0.019) | 3.252 (0.017) | 3.246 (0.006) |

![Accuracy by sensor count](reports/scaling/scaling-accuracy.png)

The graph helps the shared models at every size (-0.057 to -0.029 mph), which the dense models can't show because they already read every sensor. Sharing the weights wins once the network is big: at 128 and 207 sensors the shared GRU beats the dense one by 0.157 to 0.242 mph with 4,044 parameters instead of 105,396.

Sharing costs compute, though: at 207 sensors the shared Graph-GRU needs 23.9 s per epoch against 2.1 s for the dense one. And more sensors did not make the first 16 better: trained on all 207, the dense models are +0.471 to +0.494 mph worse on those 16 targets, the shared ones +0.192 to +0.220.

The [variants report](reports/variants/report.md) tests two changes to the shared models at 16 and 64 sensors. A learned vector of 4 numbers per sensor, which lets the network tell sensors apart, changes test MAE by -0.208 to -0.107 mph (negative is better), a bigger effect than the graph has. With it the shared GRU scores 3.045 on the first 16 sensors when trained on 64, against 3.233 for the dense GRU. The catch is that a model with those vectors can't forecast a sensor it wasn't trained on (more on that below). Swapping the GRU for an LSTM changes it by only -0.030 to -0.020.

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

At a nominal 90%, split conformal covers 90.5% of the test speeds, calibrated across all twelve horizons, while the Gaussian baselines overshoot it (93.8%) and drift away from the diagonal at other levels. But when the last speed was below 40 mph (8% of the targets), split conformal covers 63% and ACI 66%, against 93% in free flow: one quantile for everything is too small exactly when traffic is bad. The quantile network with a conformal correction learns widths that depend on the traffic state and holds 88% there, at the same average width (13.8 against 13.9 mph) and with an interval score of 19.7 against 32.0. Letting ACI adapt faster does lift congestion coverage to 88%, but only by widening every interval (the average goes from 13.9 to 19.6 mph).

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

## Federated graph learning

The first federated experiment was small: 12 sensors and a GRU that looked at one sensor at a time. The shared Graph-GRU can be federated properly, because its 4,236 weights don't depend on how many sensors a client has. I cut all 207 sensors into 8 geographic clients (10, 21, 25, 26, 28, 31, 32 and 34 sensors) with k-means on their coordinates. A client sees only its own sensors' readings and the road graph among them, so 16% of the road edges (the ones crossing into another client) are gone for everyone. Flower's FedAvg averages only the learned weights. Test MAE in mph, 2 seeds, with a pooled model (same data and training examples, one place) and local-only training (no communication) as references:

| Arm | MAE (SD) | RMSE |
|---|---:|---:|
| Federated shared Graph-GRU | 3.672 (0.014) | 7.202 |
| Federated shared GRU (no graph) | 3.705 (0.006) | 7.257 |
| Pooled shared Graph-GRU, full graph | 3.580 (0.007) | 7.092 |
| Pooled shared Graph-GRU, local graphs only | 3.596 (0.005) | 7.134 |
| Local-only shared Graph-GRU (no communication) | 3.511 (0.000) | 6.979 |
| Persistence | 4.245 | 8.196 |
| Historical average | 4.193 | 7.862 |

![Clients and the road graph](reports/fedgraph/partition-map.png)

Federated training was 0.076 mph worse than a pooled model with the same cut graph (interval +0.061 to +0.090), and the federated Graph-GRU was 0.034 mph better than the federated GRU with no graph (interval -0.046 to -0.022), so the local graph still helps. Training alone won on every client; the [headline report](reports/fedgraph/report.md) has the numbers by client and by forecast horizon. The weights are tiny next to the data (11.8 MiB exchanged over the whole run), though this is a single-machine simulation, not a deployment.

## A DCRNN-style diffusion GRU

The shared Graph-GRU only averages direct neighbors. DCRNN (Li et al., ICLR 2018) puts diffusion convolution inside the GRU gates: random walks along the road and against it, several hops, so information can travel along the network and the direction of the road matters. I wrote it from the paper's equations as a direct multi-horizon model, with the number of hops, the directions and the isolated-sensor handling configurable, and it works with the pooled and the federated code. Test MAE in mph at 16 sensors, same data, splits and update budget for every model:

| Model | MAE (SD) | Parameters |
|---|---:|---:|
| Dense GRU | 3.082 (0.021) | 11,424 |
| Shared GRU (no graph) | 3.117 (0.016) | 4,044 |
| Shared Graph-GRU (one-hop neighbor averages) | 3.061 (0.015) | 4,236 |
| DCRNN with no diffusion (0 hops) | 3.116 (0.012) | 3,948 |
| DCRNN, shuffled graph | 3.032 (0.016) | 17,772 |
| DCRNN (2 hops, forward and reverse) | 2.942 (0.019) | 17,772 |

![The model](reports/diffusion/dcrnn-architecture.png)

Diffusion itself is worth 0.174 mph over the same model with no hops. Two hops are enough, since three are not clearly better. Treating the road as undirected costs 0.102 mph against using both directions, so the direction matters. Federated, with six clients on 64 sensors, that advantage reverses: federated DCRNN is 0.026 mph worse than the federated Graph-GRU (interval +0.005 to +0.051), although pooled it is 0.164 mph better. Federating costs DCRNN 0.259 mph and the Graph-GRU 0.072 against pooled models with the same cut graph. The [report](reports/diffusion/report.md) has the ablations, the 64-sensor results and the federated runs.

## Sensors the model never saw

A sensor-vector model has a row per training sensor, so a new sensor has no vector, while the plain shared models can take any sensor. I held out one geographic area and gave each new sensor only its first week of readings. Test MAE in mph on the held-out sensors, 2 seeds:

| Method | MAE (SD) |
|---|---:|
| Persistence | 4.474 (0.000) |
| Historical average, first days only | 5.631 (0.000) |
| Shared GRU, zero-shot | 3.851 (0.002) |
| Shared Graph-GRU, zero-shot | 3.867 (0.014) |
| Sensor-vector GRU, zero vector | 4.385 (0.081) |
| Sensor-vector GRU, mean vector | 4.451 (0.002) |
| Sensor-vector GRU, vector predicted from descriptors | 4.663 (0.009) |
| Sensor-vector GRU, vector fitted on first days | 3.688 (0.006) |
| Shared Graph-GRU trained on all sensors | 3.759 (0.003) |
| Sensor-vector GRU trained on all sensors | 3.563 (0.024) |

On a sensor it never saw, the shared Graph-GRU was 0.108 mph worse than the same model trained on it. The sensor-vector model has no vector to use (4.385 mph with a zero vector). Fitting just that vector on the first week was 0.179 mph better than the zero-shot Graph-GRU. A vector predicted from sensor descriptors was 0.212 mph worse than the average vector. The [report](reports/holdout/report.md) has the comparisons with intervals and the limits.

## FedAvg, FedProx and different kinds of clients

The [FedProx study](reports/fedopt/report.md) uses 64 sensors and six clients, cut three ways (by area, balanced across congestion levels, or grouped by congestion level). It compares FedAvg and FedProx at several strengths, five local epochs against one, pooled and local-only references, and local fine-tuning of the federated model. Test MAE in mph, seed 7:

| Clients | FedAvg | FedProx, mu 0.001 | FedProx, mu 0.01 | FedProx, mu 0.1 |
|---|---:|---:|---:|---:|
| Geographic areas | 3.476 | 3.498 | 3.559 | 3.794 |
| Balanced | 3.452 | - | - | 3.802 |
| Congestion-grouped | 3.526 | - | 3.591 | 3.812 |

FedAvg's error followed how different the clients' speed distributions were rather than how much of the road graph each cut kept (one seed, so a hint rather than a result), and five local epochs a round cut the communication by about a factor of five at a small cost in accuracy. Letting each client keep training the federated model on its own data took geographic clients from 3.473 to 3.312 mph. Every client training alone scores 3.306, and the fine-tuned model was not clearly different from that (0.006 mph).

## Limits

- **Data.** One city, one four-month period and one 24-day test window. The early studies use 16 sensors on a single highway corridor, and the test period has more missing readings than training (the [audit](docs/audit.md) has the numbers).
- **Training.** No hyperparameter search, and models get equal update budgets rather than training to convergence, so a model that needs longer is undersold. Most federated studies use one or two seeds and one geographic partition.
- **Federation.** Every federated run is a simulation on one machine: no real network, stragglers or dropped clients.
- **Comparisons.** Nothing here is compared with published METR-LA numbers, because splits, horizons and budgets differ. The DCRNN-style model predicts all horizons at once instead of using the paper's decoder.

Each report lists its own limits in more detail.

## What's next

- Bigger shared models and longer training. The scores on the first 16 sensors got worse as the training population grew, every federated arm's best round was its last, and training alone beat one set of shared weights for every area. A bigger shared model, or a few specialized ones, is the first thing to try, with the fine-tuning result (which already closes the gap) as the one to compare against.
- A closer DCRNN: the paper's encoder-decoder with scheduled sampling, a second layer, and training for accuracy on all 207 sensors (mine predicts every horizon at once from the last hidden state, and is trained up to 64 sensors and only timed at 207).
- Real deployment effects (partial participation, stragglers, dropped clients, a network) that a single-machine simulation can't show, and more rounds and learning-rate schedules for FedAvg and FedProx.
- More than one geographic cut, and more seeds. The partitions are fixed k-means solutions and the federated studies use one or two seeds, so a different cut or another seed could change the smaller gaps.
- Held-out sensors beyond one area: a random set of sensors, the graph model with a vector, and a better way to start a new sensor than the average vector (predicting it from descriptors didn't work).
- Intervals that are joint across horizons and sensors, and conformal methods with guarantees conditional on the traffic state instead of only on average.
- More data: another year or another city, and drift with real change points (roadworks, a sensor replaced) and a change detector that decides when to adapt.

## Repository map

- `src/tracast/` holds the package:
  - data and protocol: `data.py`, `topology.py` (the pinned DCRNN road graph), `experiment.py`, `stats.py` (test-day bootstrap), `provenance.py`
  - models: `model.py` (GRU), `graph_model.py`, `recurrent.py` (LSTM), `shared.py` (shared-weight models), `diffusion.py` (DCRNN-style), `baselines.py`
  - experiments: `benchmark.py`, `scaling.py`, `uncertainty.py` and `conformal.py`, `shift.py`, `inductive.py` (held-out sensors)
  - federated learning: `federated*.py` and `flower_*.py` (the first 12-sensor study), `fedgraph*.py` (the federated graph models: partitions, FedAvg and FedProx, the Flower apps, and the arms x seeds runner)
- `tests/` has unit tests on small constructed arrays, so no real data is needed, plus leakage tests.
- `configs/` has one JSON file per experiment.
- `reports/` has the saved results for every experiment, the technical report and the brief.
- `docs/` has the run guide and the audit of the code and data.
- `tools/` has the scripts that rebuild this README, the figures and the reports from the saved results.

## Credits and license

The code is MIT licensed. METR-LA and the road graph come from Li, Yu, Shahabi and Liu (ICLR 2018) and have their own terms, see [DATA.md](DATA.md). Federated runs use [Flower](https://flower.ai). The [technical report](reports/research-report.md#references) lists the papers behind the methods. To cite TraCast, use [CITATION.cff](CITATION.cff).
