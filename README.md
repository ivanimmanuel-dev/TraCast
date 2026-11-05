# TraCast

TraCast forecasts traffic speed 5 to 60 minutes ahead from the last hour of sensor readings. The data is METR-LA: 207 road sensors in Los Angeles, March to June 2012. Everything runs on a CPU laptop.

I wanted to know four things. Do recurrent models beat simple baselines? Does the road graph help? Does knowing the time of day help? And does federated training lose anything compared with training on all the data in one place?

[Technical report](reports/research-report.md) · [Two-page brief](reports/research-brief.pdf) · [Run guide](docs/reproducibility.md) · [Data sources](DATA.md) · [Tests](https://github.com/ivanimmanuel-dev/TraCast/actions/workflows/tests.yml)

## What I found

- **The clock helps more than anything else I tried.** Adding the time of day and a weekend flag took the GRU from 3.255 to 3.007 mph MAE, a change of -0.249 mph (95% CI -0.324 to -0.181), and at the 60-minute horizon from 3.889 to 3.391. Training longer helped too, but less: five epochs gave 3.409.
- **The road graph doesn't give a reliable gain.** The neighbor averages help a little without the clock (MAE -0.038 mph) and hurt a little with it (+0.031). The best three models are within 0.031 mph of each other, so I can't rank them.
- **Everything beats persistence (4.100).** The historical average (3.751) and a linear model (3.622) land in between, and the GRUs are well ahead of both. (At 15 minutes persistence is still ahead of the historical average.)
- **Federated averaging is about as good as training each client alone.** At 5 rounds with balanced groups, federated averaging gets 3.774 mph, local-only 3.796 and centralized 3.750. With congestion-grouped clients it is 3.836 against 3.769.

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
```

On Linux or WSL use `python3.11` and `.venv/bin/python`. The installed `tracast` command runs everything too (`tracast --help` lists the subcommands). Use a fresh output folder for each run. The [run guide](docs/reproducibility.md) has the rest.

## What's where

- `src/tracast/`: data loading, models, the benchmark and the federated code
- `tests/`: unit tests on small constructed arrays, no real data needed
- `configs/`: one JSON file per experiment
- `reports/`: saved results for every experiment, the technical report and the brief
- `docs/`: the run guide
- `tools/`: scripts that rebuild the README, figures and reports from the saved results

The code is MIT licensed. The data and the DCRNN graph have their own terms, see [DATA.md](DATA.md).
