# TraCast: traffic speed forecasting on METR-LA

Technical report, November 2026

## Summary

I compared simple baselines, GRU and LSTM models and a graph-aware GRU for forecasting traffic speed 5 to 60 minutes ahead on METR-LA, using 16 sensors and a chronological split. Every model beats persistence (4.100 mph MAE); the best reaches 2.985. The biggest single improvement came from giving the model the time of day (the GRU goes from 3.255 to 3.007), not from changing the architecture or adding the road graph, which changes MAE by -0.038 mph without the clock and +0.031 with it. A federated simulation (three clients, twelve sensors) ends up close to training each client alone.

## 1. What I wanted to find out

This started as a way to learn how traffic forecasting works from end to end. I ended up with four questions:

1. Do GRU and LSTM models beat simple baselines like persistence, a historical average and a linear model?
2. Does the road graph help?
3. Does knowing the time of day help?
4. Does federated training lose anything compared with training on all the data in one place, or with each client training alone?

To keep the comparisons fair I wrote one pipeline (data checks, split, normalization, masks, metrics) and ran every model through it.

## 2. Data

METR-LA has speed readings from 207 loop detectors in Los Angeles every five minutes from March 1 to June 27, 2012: 34,272 timestamps and 7,094,304 readings. The file comes from a Hugging Face copy pinned to one revision and checked against a SHA-256 hash ([DATA.md](../DATA.md) has the sources and licenses). On load I check the timestamps, the five-minute spacing, the sensor IDs and that the values are numeric, finite and non-negative. Timestamps are kept as given because the data doesn't say which timezone they use.

Most experiments use 16 sensors (548,352 readings), picked from the road graph (section 3). Rows are split in time: the first 70% is train (March 1 to May 23), the next 10% validation (to June 4) and the last 20% test (June 4 to June 27). Windows are built after the split, so none crosses a boundary. That leaves 23,967 train, 3,404 validation and 6,832 test windows. A model sees 12 steps (one hour) and predicts the next 12.

Zeros and NaNs count as missing. Inputs are forward-filled within their split (leading gaps get the sensor's training mean), each sensor is normalized with the mean and standard deviation of its observed training readings, and missing targets count toward neither the loss nor the scores. Treating zero as missing can throw away real stopped traffic. For a given seed, every model gets the same windows, statistics, masks and batch order.

## 3. Models

**Baselines.** Persistence repeats the last reading. The historical average predicts the training mean for the same sensor, weekday and five-minute slot, and uses no recent readings. The linear model is one linear layer from the flattened input window to all 12 x 16 forecasts.

**GRU and LSTM.** One recurrent layer with 32 hidden units reads all 16 sensors at each step. A linear head turns the last hidden state into all 12 forecast steps for all 16 sensors at once.

**Graph-GRU.** The graph comes from the DCRNN authors' distance table (revision `602afd9d767d3aa1c9b3eac51710d6aeee12c227`). Only pairs between the 207 METR-LA sensors are used. Their finite costs set the standard deviation s of a Gaussian kernel, W[i,j] = exp(-(D[i,j] / s)^2), which I keep when it is at least 0.1 and drop for self pairs. That gives 1,515 directed edges among the 207 sensors. I picked the 16 sensors by walking the graph breadth-first from the first sensor, following edges in either direction and taking stronger ones first, which gives a connected subset with 85 directed edges. Neither the choice of sensors nor the weights use any speed data.

For each sensor and time step, Graph-GRU takes its own normalized speed plus two averages: one over the sensors that feed into it and one over the sensors it feeds, both weighted by the edge weights. A small shared layer (Linear 3 to 4, then ReLU) turns those three numbers into four features per sensor. The 16 x 4 features go into the GRU and the same kind of linear head. "Self only" zeroes the two averages and changes nothing else, so it has the same 15,760 parameters. Every model here reads all 16 sensors, so this ablation tests the graph-weighted averages, not whether neighboring sensors matter.

![Selected sensors and graph edges](benchmark/topology.png)

**Calendar inputs.** Models marked "+ calendar" get three extra inputs at every step: the sine and cosine of the time of day, and a weekend flag.

**Training.** All learned models use batch size 64, Adam with learning rate 0.001, masked MAE in normalized units and gradient clipping at 5. The checkpoint with the lowest validation MAE is the one scored on the test set.

Everything ran on a Ryzen 7 5825U laptop (8 cores, about 14 GB of RAM) with 4 PyTorch threads, Python 3.11.9, CPU PyTorch 2.5.1, NumPy 1.26.4 and pandas 2.2.3. MKLDNN is off and deterministic algorithms are on, so a repeat with the same seed gives the same numbers.

## 4. First pass: five epochs

I started with five epochs per learned model and three seeds (7, 42, 2026). Training time sums the update loops, including batch loading and excluding validation. Forward time is the median of 30 passes over the first test batch, per window, after 5 warmups. Peak memory (Windows PeakWorkingSetSize for the whole process) was about 320 MiB for every model.

| Model | MAE (SD) | RMSE (SD) | Parameters | Train time, s | Forward, ms/window |
|---|---:|---:|---:|---:|---:|
| Persistence | 4.100 | 8.549 | 0 | 0.00 | 0.0001 |
| GRU | 3.409 (0.032) | 7.179 (0.033) | 11,136 | 11.12 (1.79) | 0.0104 (0.0009) |
| LSTM | 3.363 (0.007) | 7.091 (0.031) | 12,736 | 10.89 (0.97) | 0.0143 (0.0021) |
| Graph-GRU | 3.364 (0.036) | 7.107 (0.106) | 15,760 | 12.73 (1.54) | 0.0211 (0.0088) |
| Graph-GRU, self only | 3.368 (0.023) | 7.073 (0.047) | 15,760 | 12.63 (1.43) | 0.0166 (0.0004) |

LSTM and Graph-GRU end up at about the same MAE, and the LSTM has fewer parameters. Graph-GRU minus self-only is -0.0137, +0.0158, -0.0151 mph by seed (mean -0.0043), so the neighbor averages barely matter and the sign changes between seeds. Most of Graph-GRU's lead over the GRU stays when the averages are zeroed.

![Forecast error by horizon](benchmark/horizons.png)

![Validation and training convergence](benchmark/convergence.png)

Five epochs turned out to be too short. In 12 of 12 runs the best validation epoch was the last one, so the models were still improving when training stopped.

## 5. Training longer, and adding calendar inputs

Each model trains for up to 60 epochs, stops after 8 epochs with no validation gain, and halves the learning rate after 2. Models marked "+ calendar" also get the time of day (sine and cosine) and a weekend flag as inputs. I added two simple baselines: a historical average (the training mean for each sensor, weekday and 5-minute slot) and a linear model over the whole input window. 5 seeds: 7, 42, 2026, 123, 999.

| Model | MAE (SD) | MAE range over test days | 15 min | 30 min | 60 min | RMSE (SD) | Parameters | Epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 4.100 | 3.681-4.469 | 3.316 | 4.048 | 5.290 | 8.549 | 0 | - |
| Historical average | 3.751 | 3.300-4.185 | 3.751 | 3.751 | 3.752 | 7.405 | 0 | - |
| Linear | 3.622 (0.001) | 3.218-3.988 | 2.967 | 3.602 | 4.572 | 7.386 (0.005) | 37,056 | 43 |
| Linear + calendar | 3.597 (0.003) | 3.196-3.962 | 2.961 | 3.584 | 4.512 | 7.373 (0.007) | 43,968 | 39 |
| GRU | 3.255 (0.026) | 2.892-3.602 | 2.805 | 3.243 | 3.889 | 6.873 (0.085) | 11,136 | 37 |
| GRU + calendar | 3.007 (0.025) | 2.687-3.313 | 2.730 | 3.023 | 3.391 | 6.373 (0.078) | 11,424 | 42 |
| LSTM | 3.271 (0.013) | 2.906-3.617 | 2.845 | 3.259 | 3.885 | 6.913 (0.046) | 12,736 | 32 |
| Graph-GRU | 3.193 (0.038) | 2.844-3.527 | 2.779 | 3.195 | 3.765 | 6.743 (0.134) | 15,760 | 37 |
| Graph-GRU + calendar | 3.016 (0.031) | 2.695-3.324 | 2.727 | 3.026 | 3.422 | 6.403 (0.095) | 16,048 | 35 |
| Graph-GRU (self only) | 3.231 (0.037) | 2.872-3.574 | 2.829 | 3.229 | 3.793 | 6.854 (0.109) | 15,760 | 33 |
| Graph-GRU (self only) + calendar | 2.985 (0.039) | 2.666-3.292 | 2.732 | 3.002 | 3.327 | 6.363 (0.162) | 16,048 | 45 |

MAE and RMSE are test means in mph; 15/30/60 min are MAE at those horizons. The range column resamples the 24 test days to show how much a score depends on which days ended up in the test set. Persistence and the historical average aren't trained, so they run once.

Graph-GRU (self only) + calendar is lowest at 2.985 mph (persistence 4.100, historical average 3.751), but GRU + calendar and Graph-GRU + calendar are within 0.031 mph of it, so I can't rank them. Training longer helped: the GRU went from 3.409 at five epochs to 3.255. Calendar inputs gave the biggest jump: the GRU changes by -0.249 mph (95% CI -0.324 to -0.181). Graph-GRU minus its self-only version (the same model with the neighbor averages zeroed) is -0.038 mph (95% CI -0.062 to -0.018) without calendar inputs and +0.031 mph (95% CI +0.008 to +0.055) with calendar inputs. The sign flips, so the graph doesn't give a reliable gain.

| Difference in MAE (first - second) | Estimate (mph) | 95% CI over test days |
|---|---:|---:|
| GRU + calendar - GRU | -0.249 | -0.324 to -0.181 |
| Linear + calendar - linear | -0.025 | -0.035 to -0.015 |
| Graph-GRU + calendar - Graph-GRU | -0.177 | -0.229 to -0.130 |
| LSTM - GRU | +0.016 | -0.021 to +0.057 |
| Graph-GRU - self only | -0.038 | -0.062 to -0.018 |
| Graph-GRU - self only, both with calendar | +0.031 | +0.008 to +0.055 |
| Graph-GRU - GRU, both with calendar | +0.010 | -0.007 to +0.025 |
| GRU - historical average | -0.496 | -0.672 to -0.320 |
| GRU + calendar - historical average | -0.744 | -0.926 to -0.573 |
| Historical average - persistence | -0.349 | -0.551 to -0.135 |

Negative means the first model is better. Both models in a row are scored on the same windows and the same resampled days. The intervals are about which days were held out; the seed SDs in the table above cover random initialization.

![Forecast error by horizon, longer-training study](figures/extended-horizons.png)

## 6. Federated learning

The centralized models above read all 16 sensors together. For federated learning I wanted a model that means the same thing on every client, so the federated task uses a univariate GRU shared across sensors: one sensor per example, 3,756 parameters (15,024 bytes).

The setup is twelve sensors, the first 4,032 timestamps (fourteen days), three clients with four sensors each (no sensor on two clients), three seeds (7, 42, 2026) and one local epoch per round. The runtime is Flower 1.42.0 (`ClientApp`, `ServerApp`, `FedAvg`) through `flwr run` on Ray, with all clients on one machine. Each sensor is normalized with its own training statistics. The server averages weights, weighted by each client's number of usable training examples (examples with at least one observed target). Errors are pooled as sums and counts before computing MAE and RMSE.

There are two references. Centralized trains one model on all clients' examples with the same starting weights, scalers, masks and total number of examples. Local-only trains each client by itself with the same local work and no communication, and each client keeps the round with its own best validation MAE. Optimizers restart at every local fit, and no optimizer state is shared.

Sensors are ranked by the fraction of their training readings below 40 mph. Balanced groups interleave those ranks across clients and congestion-grouped clients get consecutive blocks. The mean Jensen-Shannon divergence between each client's training speed histogram and the pooled one is 0.0169 bits for balanced groups and 0.0307 for congestion groups.

| Sensor grouping | Rounds | Federated MAE | Centralized MAE | Local-only MAE | Model weights sent |
|---|---:|---:|---:|---:|---:|
| Balanced | 1 | 4.047 | 3.792 | 4.003 | 180,288 B |
| Balanced | 3 | 3.809 | 3.765 | 3.826 | 450,720 B |
| Balanced | 5 | 3.774 | 3.750 | 3.796 | 721,152 B |
| Congestion-grouped | 5 | 3.836 | 3.750 | 3.769 | 721,152 B |

At 5 rounds with balanced groups, federated averaging gets 3.774 mph, local-only 3.796 and centralized 3.750. With congestion-grouped clients it is 3.836 against 3.769. The gaps are about the size of the seed spread, so in this setup training alone is about as good as federating.

Going from 1 to 5 rounds cuts balanced MAE by 0.273 mph and sends 4 times as many model-weight bytes. The byte counts cover weights sent for training, validation and the final test; message overhead isn't counted. Congestion grouping raises MAE by 0.0616 mph at five rounds while lowering RMSE by 0.0072, so the change in the speed distribution doesn't hit both measures the same way.

As a check, the centralized reference is identical for both ownership rules, and for each seed the global model after r rounds matches the one from the longer runs. [Client assignments, per-seed numbers and the communication count](federated-study/findings.md).

![Communication and convergence](federated-study/tradeoffs.png)

## 7. What I take from this

- **Give the model the clock.** Time of day and a weekend flag beat every architecture change I tried. By 30 minutes ahead, a plain historical average already beats persistence.
- **The graph wasn't worth it here.** The neighbor averages helped a little without the clock and hurt a little with it, and Graph-GRU with the clock is not clearly different from a GRU with the clock.
- **Check training length before comparing models.** After five epochs the LSTM and Graph-GRU both looked better than the GRU by about 0.046 mph. With proper training the LSTM is level with the GRU (+0.016) and Graph-GRU is 0.062 ahead, a lead that disappears once the clock is added.
- **Federated averaging didn't beat training alone** for this small model and short slice, and pooling the data was best.

What this doesn't cover: one 16-sensor subset picked by a graph walk (not representative of every road, and neighbors outside the subset are missing), one 24-day test window (scores move by about 0.3 mph with the choice of days, though the gaps between models are much tighter), no hyperparameter search, and a federated task that is univariate and simulated on one machine, so its errors shouldn't be compared with the centralized tables. Timings are rough, and the byte counts only cover model weights.

## 8. Reproducing it

The [README](../README.md) and the [run guide](../docs/reproducibility.md) have the commands. They fetch checksum-verified data and use pinned dependencies. Each experiment's folder in `reports/` has its config, source and data hashes, sensor IDs, split ranges, scaler values, histories and per-horizon counts. Unit tests run on small constructed arrays, and the experiment commands are the real-data runs.

## References

1. Li, Y., Yu, R., Shahabi, C., Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR. https://arxiv.org/abs/1707.01926
2. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP. https://arxiv.org/abs/1406.1078
3. Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. McMahan, H. B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS. https://arxiv.org/abs/1602.05629
5. Beutel, D. J. et al. (2020). Flower: A Friendly Federated Learning Research Framework. https://arxiv.org/abs/2007.14390
