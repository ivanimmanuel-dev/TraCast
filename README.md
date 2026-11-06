# TraCast

TraCast forecasts traffic speed 5 to 60 minutes ahead from the last hour of sensor readings. The data is METR-LA: 207 road sensors in Los Angeles, March to June 2012. Everything runs on a CPU laptop.

I wanted to know four things. Do recurrent models beat simple baselines? Does the road graph help? Does knowing the time of day help? And does federated training lose anything compared with training on all the data in one place? Then I scaled it up to all 207 sensors, put prediction intervals on the forecasts and checked what happens when traffic drifts. Last, I federated the shared Graph-GRU across all 207 sensors, wrote a DCRNN-style diffusion GRU, and tested forecasting for sensors a model never trained on.

[Technical report](reports/research-report.md) · [Two-page brief](reports/research-brief.pdf) · [Scaling](reports/scaling/report.md) · [Shared variants](reports/variants/report.md) · [Federated graph](reports/fedgraph/report.md) · [DCRNN](reports/diffusion/report.md) · [FedProx](reports/fedopt/report.md) · [New sensors](reports/holdout/report.md) · [Intervals](reports/uncertainty/report.md) · [Drift](reports/shift/report.md) · [Audit](docs/audit.md) · [Run guide](docs/reproducibility.md) · [Data sources](DATA.md) · [Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml)

## What I found

- **The clock helps more than anything else I tried.** Adding the time of day and a weekend flag took the GRU from 3.255 to 3.007 mph MAE, a change of -0.249 mph (95% CI -0.324 to -0.181), and at the 60-minute horizon from 3.889 to 3.391. Training longer helped too, but less: five epochs gave 3.409.
- **The road graph doesn't give a reliable gain.** The neighbor averages help a little without the clock (MAE -0.038 mph) and hurt a little with it (+0.031). The best three models are within 0.031 mph of each other, so I can't rank them.
- **Everything beats persistence (4.100).** The historical average (3.751) and a linear model (3.622) land in between, and the GRUs are well ahead of both. (At 15 minutes persistence is still ahead of the historical average.)
- **In the first, small federated study (12 sensors, three clients), federated averaging was about as good as training each client alone.** At 5 rounds with balanced groups, federated averaging gets 3.774 mph, local-only 3.796 and centralized 3.750. With congestion-grouped clients it is 3.836 against 3.769.
- **In the shared models the graph helps, at 5 of 5 sizes.** Adding the neighbor averages changes MAE by -0.057 to -0.029 mph. Without them a shared model sees nothing about its neighbors, so this is a clean test of the graph, which the dense models can't give.
- **Sharing the weights wins once the network is big.** At 128 and 207 sensors the shared GRU beats the dense GRU by 0.157 to 0.242 mph (the interval excludes zero), and at 16, 32 and 64 they are level. It does that with 4,044 parameters instead of 105,396.
- **In a shared model, telling the sensors apart matters more than the graph.** A learned 4-number vector per sensor changes test MAE by -0.208 to -0.107 mph at 16 and 64 sensors, the graph's neighbor averages by -0.055 to -0.050 and an LSTM instead of a GRU by -0.030 to -0.020 (negative is better).
- **Prediction intervals that are right on average can fail when it matters.** At a nominal 90%, split conformal covers 90.5% of the test speeds, but only 63% when the last speed was below 40 mph (against 93% in free flow). A quantile network with a conformal correction holds 88% there at the same average width.
- **Adapting a deployed model helps a little, and cheap adaptation can hurt.** Fine-tuning the output layer daily improves MAE by 0.031 mph and fine-tuning the whole network weekly by 0.042 (for a model trained on only the first 40% of the timeline: 0.072 and 0.188), at 3 s and 27 s for the 24 test days. Averaging past errors into the forecast was worse than doing nothing (+0.094 mph at best).
- **Federated graph forecasting works at full scale, but training alone was better.** A federated shared Graph-GRU on all 207 sensors (8 geographic clients with 10 to 34 sensors, each seeing only its own part of the road graph) scores 3.672 (0.014) mph, about 0.52 mph better than the historical average. Every client training by itself did 0.161 mph better (interval +0.137 to +0.184), and also beat one pooled model with the same graph, which points to a single set of weights for every area as the limit.
- **A DCRNN-style diffusion GRU is the best centralized model I tried.** At 16 sensors it beats the shared Graph-GRU by 0.119 mph (interval -0.153 to -0.083) and by 0.174 mph at 64 sensors (one seed there). A randomly relabelled graph is 0.090 mph worse, so the real road structure is what it uses. Federated, with six clients on 64 sensors, that advantage reverses: federated DCRNN is 0.026 mph worse than the federated Graph-GRU (interval +0.005 to +0.051), although pooled it is 0.164 mph better.
- **A model can forecast sensors it never trained on, and the sensor-vector models can be rescued.** On 14 held-out sensors (a whole area, with only their first week of data available) the shared Graph-GRU scores 3.867 (0.014) mph zero-shot against 3.759 (0.003) when it trained on them. The vector model breaks without a vector (4.385 (0.081) with a zero vector), but fitting just that vector on the first week gives 3.688 (0.006) mph, 0.179 mph better than the plain zero-shot model. Predicting the vector from sensor descriptors did not work (4.663 (0.009) against 4.451 (0.002) for the average vector).
- **FedProx did not help on 64 sensors with six clients.** FedProx was worse than FedAvg in all 6 settings I ran, and the penalty grows with mu (geographic clients: +0.022 at mu 0.001, +0.082 at mu 0.01, +0.318 at mu 0.1); with five local epochs a round, where there is more drift to correct, it was still 0.462 mph worse. A round here is short and the model is still improving at the end, so the proximal pull mostly slows learning.
- **Fine-tuning the federated model on each client.** On geographic clients, letting every client keep training the final federated model on its own data improves it by 0.161 mph (interval -0.197 to -0.123) and ends level with every client training alone (no clear difference).

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

Two variants of the shared models are in the [variants report](reports/variants/report.md), tested at 16 and 64 sensors. A learned vector of 4 numbers per sensor, which lets the shared network tell sensors apart, changes the test MAE on the sensors the models train on by -0.208 to -0.107 mph (negative is better), a bigger effect than the graph has, and with it the shared GRU scores 3.045 on the first 16 sensors when trained on 64, against 3.233 for the dense GRU. The catch is that a model with those vectors can't forecast a sensor it wasn't trained on. Swapping the GRU for an LSTM changes it by -0.030 to -0.020.

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

Federated training cost +0.076 (+0.061 to +0.090) mph against a pooled model with the same cut graph, and the local graph still helped (-0.034 (-0.046 to -0.022) mph against the federated model with no graph). Training alone won on every client; the [headline report](reports/fedgraph/report.md) has the numbers by client and by forecast horizon. The weights are tiny next to the data (11.8 MiB exchanged over the whole run), though this is a single-machine simulation, not a deployment.

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

Three hops are not clearly better than two. Diffusion itself is worth 0.174 mph over the same model with no hops. Treating the road as undirected is 0.102 mph worse than using both directions. Federated, with six clients on 64 sensors, that advantage reverses: federated DCRNN is 0.026 mph worse than the federated Graph-GRU (interval +0.005 to +0.051), although pooled it is 0.164 mph better. Federating costs DCRNN 0.259 mph and the Graph-GRU 0.072 against pooled models with the same cut graph. The [report](reports/diffusion/report.md) has the ablations, the 64-sensor results and the federated runs.

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

The [report](reports/holdout/report.md) has the comparisons with intervals and the limits.

## FedAvg, FedProx and different kinds of clients

The [FedProx study](reports/fedopt/report.md) uses 64 sensors and six clients, cut three ways (by area, balanced across congestion levels, or grouped by congestion level), and compares FedAvg and FedProx at several strengths, five local epochs against one, pooled and local-only references, and local fine-tuning of the federated model. In that one seed FedAvg's error followed how different the clients' speed distributions were rather than how much of the road graph each cut kept, and five local epochs a round cut the communication by about a factor of five at a small cost in accuracy. The FedProx grid used seed 7 and the fine-tuning runs seeds 7 and 42.

## How it's set up

The first 70% of the timeline is train, the next 10% validation and the last 20% test, and windows never cross a split boundary. Zero and missing readings are left out of the loss and the metrics, inputs are forward-filled within a split, and normalization uses training data only. The saved model is the one with the best validation MAE; test scores come after that.

The 16 sensors come from walking the published DCRNN distance graph from its first sensor, which gives 85 directed edges. Sensors and edge weights come from distances only, never from speeds. Graph-GRU gives each sensor its own speed plus the average of its upstream and downstream neighbors, passes that through a small shared layer, and feeds all 16 sensors into a GRU. "Self only" is the same model with the neighbor averages set to zero. One thing to keep in mind: every model here, the plain GRU too, reads all 16 sensors, so this tests the graph-weighted averages and not whether neighboring sensors matter.

What this doesn't cover: one 16-sensor subset, one 24-day test window, no hyperparameter search, and federated parts that are simulations on one machine.

## The first federated study (12 sensors)

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

# all 207 sensors, shared-model variants, prediction intervals, drift (the scaling run takes a few hours)
.\.venv\Scripts\python.exe -m tracast.cli scaling --config configs/scaling.json --out runs/scaling
.\.venv\Scripts\python.exe -m tracast.cli scaling --config configs/variants.json --out runs/variants
.\.venv\Scripts\python.exe -m tracast.cli uncertainty --config configs/uncertainty.json --out runs/uncertainty
.\.venv\Scripts\python.exe -m tracast.cli shift --config configs/shift.json --out runs/shift

# federated graph models: all 207 sensors, DCRNN, FedProx, held-out sensors (each takes hours)
.\.venv\Scripts\python.exe -m tracast.cli fedgraph --config configs/fedgraph.json --out runs/fedgraph
.\.venv\Scripts\python.exe -m tracast.cli fedgraph --config configs/fedgraph-diffusion.json --out runs/fedgraph-diffusion
.\.venv\Scripts\python.exe -m tracast.cli fedgraph --config configs/fedgraph-optimization.json --out runs/fedgraph-optimization
.\.venv\Scripts\python.exe -m tracast.cli holdout --config configs/holdout.json --out runs/holdout
```

On Linux or WSL use `python3.11` and `.venv/bin/python`. The installed `tracast` command runs everything too (`tracast --help` lists the subcommands). Use a fresh output folder for each run. The [run guide](docs/reproducibility.md) has the rest.

## What's next

- Bigger shared models and longer training. The scores on the first 16 sensors got worse as the training population grew, every federated arm's best round was its last, and training alone beat one set of shared weights for every area. A bigger shared model, or a few specialized ones, is the first thing to try, with the fine-tuning result (which already closes the gap) as the one to compare against.
- A closer DCRNN: the paper's encoder-decoder with scheduled sampling, a second layer, and training for accuracy on all 207 sensors (mine predicts every horizon at once from the last hidden state, and is trained up to 64 sensors and only timed at 207).
- Real deployment effects (partial participation, stragglers, dropped clients, a network) that a single-machine simulation can't show, and more rounds and learning-rate schedules for FedAvg and FedProx.
- More than one geographic cut, and more seeds. The partitions are fixed k-means solutions and the federated studies use one or two seeds, so a different cut or another seed could change the smaller gaps.
- Held-out sensors beyond one area: a random set of sensors, the graph model with a vector, and a better way to start a new sensor than the average vector (predicting it from descriptors didn't work).
- Intervals that are joint across horizons and sensors, and conformal methods with guarantees conditional on the traffic state instead of only on average.
- More data: another year or another city, and drift with real change points (roadworks, a sensor replaced) and a change detector that decides when to adapt.

## What's where

- `src/tracast/`: data loading, models (dense, shared-weight and the DCRNN-style diffusion GRU), the benchmark, scaling, conformal intervals, drift replay, held-out sensors and the federated code (the first small study and the federated graph models)
- `tests/`: unit tests on small constructed arrays, no real data needed
- `configs/`: one JSON file per experiment
- `reports/`: saved results for every experiment, the technical report and the brief
- `docs/`: the run guide and the audit of the code and data
- `tools/`: scripts that rebuild the README, figures and reports from the saved results

The code is MIT licensed. The data and the DCRNN graph have their own terms, see [DATA.md](DATA.md).
