# TraCast: recurrent encoders, road-distance graph features, and federated traffic forecasting

Technical report - October 2026

## Abstract

TraCast evaluates traffic-speed forecasting on the real METR-LA dataset with chronological evaluation, training-only normalization, and masked targets. A fixed 16-sensor subset is selected from the published DCRNN road-distance graph. GRU, LSTM, and Graph-GRU models receive the same training windows and five-epoch budgets across seeds 7, 42, and 2026. Mean test MAE is 3.409 mph for GRU, 3.363 for LSTM, and 3.364 for Graph-GRU, compared with 4.100 for persistence. Removing graph neighbors changes mean MAE by 0.0043 mph, with mixed seed-level effects. A separate Flower study evaluates shared univariate forecasting across disjoint sensor partitions and records accuracy, convergence, sample counts, and parameter communication.

## 1. Introduction

Traffic forecasts depend on both temporal patterns and relationships between sensors. This project asks two questions: how recurrent encoder choice and explicit road-distance context affect forecasting under a fixed training budget, and how local training plus model averaging changes accuracy and communication relative to pooled training. The centralized graph benchmark and federated experiment are separate tasks with their own matched references.

## 2. Related work

[Li et al. (2018)](https://arxiv.org/abs/1707.01926) combine directed diffusion with recurrent forecasting and distribute METR-LA sensor data and distance relationships. TraCast uses those relationships for a simpler one-hop spatial aggregation layer; it does not implement the DCRNN diffusion recurrent cell or scheduled-sampling decoder. [Cho et al. (2014)](https://arxiv.org/abs/1406.1078) introduce the gated recurrent encoder used as the GRU reference. [Hochreiter and Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735) introduce LSTM. [McMahan et al. (2017)](https://arxiv.org/abs/1602.05629) describe federated averaging. [Flower](https://arxiv.org/abs/2007.14390) supplies the client/server runtime used here.

## 3. Data and preprocessing

The checksum-verified HDF5 contains 34,272 five-minute timestamps and 207 sensors, from March 1 through June 27, 2012: 7,094,304 speed entries. The benchmark selects 16 sensors, giving 548,352 speed entries before missing-value masking. Raw timestamps are retained without assigning an undocumented timezone. Numeric columns, unique sensor IDs, increasing timestamps, five-minute spacing, finite nonnegative observations, and file hashes are checked before selection.

Rows split 70/10/20 before sequence construction. Training spans March 1 00:00 to May 23 07:05, validation May 23 07:10 to June 4 04:40, and test June 4 04:45 to June 27 23:55. These contain 23,967, 3,404, and 6,832 windows respectively. Every window stays within its split. Twelve observed time steps predict the next twelve steps, covering 5-60 minute forecast horizons.

Zero and NaN readings are missing. Inputs forward-fill within their split and use the sensor's training mean for leading gaps. A mean and population standard deviation are fitted to each sensor's observed training readings; constant sensors use a scale of one. Missing targets contribute to neither loss nor scores. The same windows, statistics, masks, and batch ordering are used across model architectures for a given seed. The zero-as-missing convention can exclude genuine stopped traffic.

## 4. Distance graph and architecture

The author-distributed distance table is pinned to revision `602afd9d767d3aa1c9b3eac51710d6aeee12c227` and checked by SHA-256. Endpoints outside the canonical 207 METR-LA sensors are excluded. Their aligned finite distance costs set a population standard deviation s. For a supplied directed cost D[i,j], W[i,j] = exp(-(D[i,j]/s)^2), retaining values at least 0.1 and removing self pairs. No speed correlation, validation score, or test target enters graph construction.

A breadth-first traversal from the first canonical sensor selects a connected 16-sensor subset using the union of edge directions, with stronger weights first and ID tie-breaking. Its induced graph contains 85 directed relationships. Full IDs, coordinates, weights, source hashes, and the kernel scale are in [protocol.json](benchmark/protocol.json). Edges express the published distance relationships; straight lines in the figure are not road polylines.

Graph-GRU computes separate row-normalized incoming and outgoing means. At each sensor and time step, a shared Linear(3,4) plus ReLU transforms its own normalized speed and two neighbor means into four features. The 16 node embeddings flatten to 64 features, enter a one-layer GRU with hidden width 32, and feed a linear head producing 12 x 16 forecasts. The self-only ablation sets both neighbor operators to zero while retaining the same learned projection, recurrent encoder, head, initialization rule, and parameter count.

The GRU reference has 16 input channels, 32 hidden units, and the same direct horizon head. LSTM replaces the recurrent encoder with a 32-unit LSTM. These models have different parameter counts; equal epochs and example exposure do not imply equal parameter counts or FLOPs.

![Selected topology](benchmark/topology.png)

## 5. Experimental setup

All learned models train for five epochs using batch size 64, Adam at 0.001, masked MAE in normalized units, and gradient clipping at 5. Every model processes the same training-window budget. The checkpoint with lowest validation MAE is selected separately per model and seed. Test scoring occurs after selection.

Trials run sequentially in fresh Python processes on an AMD Ryzen 7 5825U (8 cores, 16 logical processors), with four PyTorch threads. Windows reports 14,861,852,672 bytes of physical memory. Python 3.11.9, CPU PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3, and the remaining runtime packages are pinned. MKLDNN is disabled and deterministic PyTorch algorithms are enabled.

Training time sums update loops, including batch retrieval and excluding validation. Inference timing uses five warmups followed by thirty forward passes on the same first test batch and reports the median per window. It excludes inverse scaling, loading, and metric computation. Full test-evaluation time is logged separately. Windows PeakWorkingSetSize measures the whole fresh process, including HDF5/CSV loading and training; it is not model-only memory. Reported timings describe this workstation run, during which documentation work also occurred.

## 6. Centralized results

Values are mean (sample SD) across three training seeds. Persistence is deterministic and evaluated once. Seed variation uses one shared test block and does not quantify uncertainty across future dates or networks.

| Model | MAE, mph | RMSE, mph | Parameters | Train updates, s | Forward ms/window |
|---|---:|---:|---:|---:|---:|
| Persistence | 4.100 | 8.549 | 0 | 0.00 | 0.0001 |
| GRU | 3.409 (0.032) | 7.179 (0.033) | 11,136 | 11.12 (1.79) | 0.0104 (0.0009) |
| LSTM | 3.363 (0.007) | 7.091 (0.031) | 12,736 | 10.89 (0.97) | 0.0143 (0.0021) |
| Graph-GRU | 3.364 (0.036) | 7.107 (0.106) | 15,760 | 12.73 (1.54) | 0.0211 (0.0088) |
| Graph-GRU, self only | 3.368 (0.023) | 7.073 (0.047) | 15,760 | 12.63 (1.43) | 0.0166 (0.0004) |

![Forecast horizon errors](benchmark/horizons.png)

![Validation and training convergence](benchmark/convergence.png)

### Graph ablation

Graph-GRU minus self-only MAE differences by seed are -0.013652, +0.015787, -0.015134 mph. Their mean is -0.004333 mph. Graph-GRU improves over the raw-input GRU in all three runs, but the edge contribution is small and changes sign. The self-only projection retains most of the mean MAE improvement over the smaller GRU reference. LSTM has nearly the same mean MAE as Graph-GRU, with fewer parameters and lower seed dispersion in this configuration.

Whole-process peak memory was approximately 320 MiB for all architectures. Runtime and preprocessing dominate this measurement, so those small differences should not be interpreted as architecture-specific memory savings.

## 7. Federated learning


Twelve completed local Flower experiments: 12 sensors, 14 days, three clients with four disjoint sensors each, and seeds 7/42/2026. The model is the existing shared univariate GRU, with 3,756 float32 parameters. Every round contains one local epoch and all three clients participate. The official ClientApp, ServerApp, and FedAvg run through the Flower 1.42.0 CLI/SuperLink and Ray runtime.

### Results

Mean (sample SD) across the three seeds; errors in mph. Checkpoints are selected by validation MAE within each round budget.

| Client partition | Rounds | FedAvg MAE | FedAvg RMSE | Pooled MAE | Pooled RMSE | Parameter bytes |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 1 | 4.0470 (0.0599) | 7.7919 (0.1019) | 3.7924 (0.0101) | 7.5361 (0.0182) | 180,288 |
| balanced | 3 | 3.8093 (0.0357) | 7.5289 (0.0254) | 3.7653 (0.0344) | 7.5511 (0.0066) | 450,720 |
| balanced | 5 | 3.7742 (0.0138) | 7.5200 (0.0121) | 3.7497 (0.0467) | 7.5350 (0.0265) | 721,152 |
| congestion | 5 | 3.8358 (0.0288) | 7.5128 (0.0197) | 3.7497 (0.0467) | 7.5350 (0.0265) | 721,152 |

Balanced-partition mean MAE fell by 0.2729 mph from one to five rounds, while estimated parameter payload increased fourfold. The three-to-five-round gain was 0.0351 mph. Seed 42 retained the same selected test result at three and five rounds. Five-round FedAvg remained 0.0245 mph above the mean pooled-reference MAE.

Congestion-grouped clients had +0.0616 mph higher mean MAE than balanced clients at five rounds, but their mean RMSE difference was -0.0072 mph. The distribution change affects the two error measures differently in this experiment.

![Communication trade-offs and validation convergence](federated-study/tradeoffs.png)

### Partition rule and data controls

Both rules use the same first twelve sensors in canonical traffic-file order and the first 4,032 timestamps. The same 70/10/20 chronological row boundaries apply to every client and case. Per-sensor normalization and zero/NaN target masks use the existing pipeline.

Sensors are ranked by their fraction of observed **training** speeds below 40 mph. The balanced rule interleaves rank positions across three clients; the congestion rule assigns consecutive rank blocks. Neither uses validation/test readings or geography. These labels describe simulated sensor ownership, not cities or organizations. Balanced groups are not assumed to be IID.

The observed training-speed histogram bins are 0/10/20/.../80/+inf mph, excluding missing readings. Mean client-to-pooled Jensen-Shannon divergence was 0.016882 bits for balanced ownership and 0.030693 bits for congestion grouping. This descriptive statistic is calculated from training data and is constant across seeds; it is not averaged as three independent data samples.

| Rule | Client | Sensor IDs | Usable train windows | Observed training fraction below 40 mph |
|---|---:|---|---:|---:|
| balanced | 0 | 767541, 767471, 773869, 773062 | 10,779 | 0.0506 |
| balanced | 1 | 767620, 737529, 717447, 717816 | 10,730 | 0.0760 |
| balanced | 2 | 765604, 767542, 717445, 717446 | 10,788 | 0.1301 |
| congestion | 0 | 767541, 767620, 765604, 767471 | 10,781 | 0.0018 |
| congestion | 1 | 737529, 767542, 773869, 717447 | 10,780 | 0.0573 |
| congestion | 2 | 717445, 773062, 717816, 717446 | 10,736 | 0.1984 |

Client sample counts mean usable sensor-window examples. A window with no observed target contributes to neither local training nor its FedAvg example weight. Validation/test retain every candidate window and pool errors only over valid targets.

The pooled reference uses the same shared univariate architecture, initial model, scalers, masks, and total example exposure. Its dataset follows canonical sensor order regardless of client assignment. Adam resets at each local fit and each pooled epoch block; no optimizer state is federated. Partial batches create different total optimizer-step counts, which are stored in the case records.

### Reproducibility checks

The pooled reference's complete numerical output and state hash match exactly across balanced and congestion ownership for every seed. Global model-state hashes from the one- and three-round cases match the corresponding prefixes of the five-round cases. [summary.json](federated-study/summary.json) records all checks. Seed-42 LSTM and Graph-GRU repeats for the separate centralized benchmark also matched their test metrics, validation histories, and checkpoint hashes.

All twelve cases completed. Their total measured wall time was 687.0 seconds (11.45 minutes), including launch overhead, simulation, selected test scoring, the pooled reference, and plots. Cases ran sequentially with one client actor active, two reserved Ray CPUs, one PyTorch thread, and a 128 MiB Ray object-store cap. Whole-study peak RAM was not measured.

### Per-seed records

| Case | Selected round | FedAvg MAE | FedAvg RMSE | Pooled MAE | Case wall seconds |
|---|---:|---:|---:|---:|---:|
| b1-s7 | 1 | 4.111763 | 7.906256 | 3.792999 | 56.0 |
| b3-s7 | 2 | 3.850131 | 7.556947 | 3.759755 | 49.9 |
| b5-s7 | 5 | 3.758373 | 7.521437 | 3.712816 | 61.8 |
| c5-s7 | 4 | 3.810094 | 7.494595 | 3.712816 | 65.5 |
| b1-s42 | 1 | 3.993612 | 7.710778 | 3.782038 | 38.7 |
| b3-s42 | 3 | 3.783720 | 7.507262 | 3.734071 | 52.1 |
| b5-s42 | 3 | 3.783720 | 7.507262 | 3.734071 | 68.3 |
| c5-s42 | 2 | 3.866865 | 7.533776 | 3.734071 | 66.1 |
| b1-s2026 | 1 | 4.035705 | 7.758775 | 3.802187 | 43.7 |
| b3-s2026 | 3 | 3.794028 | 7.522578 | 3.802187 | 55.3 |
| b5-s2026 | 4 | 3.780410 | 7.531418 | 3.802187 | 67.6 |
| c5-s2026 | 3 | 3.830434 | 7.510014 | 3.802187 | 62.0 |

Each [case directory](federated-study/cases) contains config, source/data/dependency hashes, client metadata, all round metrics, selected test scores, pooled histories, status, and the runtime transcript. Log coloring and trailing whitespace are removed for readability. Raw partitions, model checkpoints, and local Flower service state stay outside Git.

### Communication definition

A model contains 15,024 bytes of float32 tensor values. Each round has three training downloads, three uploads, and three validation downloads: 135,216 bytes. Selected-model test evaluation adds 45,072 bytes. Total payload is therefore (3 x rounds + 1) x 3 x 15,024: 180,288 / 450,720 / 721,152 bytes for 1 / 3 / 5 rounds.

These estimates exclude array headers, metrics, config, message framing, app packaging, transport overhead, latency, and energy. They describe parameter payload rather than measured network traffic. All scoring pools absolute errors, squared errors, and valid target counts; client RMSEs are not averaged.

### Scope

Three seeds describe initialization and local-shuffle variation on one test block. The fourteen-day slice, twelve sensors, predefined congestion threshold, and short round budgets limit the conclusion to this setup. The selected model is univariate, so it is separate from the full-date-range multivariate graph benchmark. No federated graph partitioning, client dropout, differential privacy, secure aggregation, or sensor-failure experiment is included.

```powershell
.\.venv\Scripts\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study-new
```

[Configuration](federated-study/config.json) - [All measured cases](federated-study/cases.json) - [Summary and controls](federated-study/summary.json) - [Full research report](research-report.md)


Clients are logical sensor partitions on one machine. The shared GRU has one sensor per example so input/output weights retain the same meaning on every client. Sensor normalization uses local observed training readings only. The reference pools those same normalized examples, matches the initial model and exposure, and uses canonical sensor ordering. Model averaging is weighted by usable training-window counts. Evaluation pools absolute errors, squared errors, and valid target counts before calculating MAE/RMSE. Shared files on the simulation host provide logical ownership, not a privacy boundary. No differential privacy or secure aggregation is evaluated.

## 8. Discussion and limits

The learned centralized encoders improve over persistence on this sensor subset and test period. The experiments give little evidence that these one-hop distance features outperform the self-only projection under a five-epoch budget. Only three initialization seeds are used, with no hyperparameter search, temporal resampling, or alternative graph-selection study. The topology-only BFS subset is connected but not representative of every road in METR-LA. Excluding neighbors outside the subset removes spatial context.

The fixed train/validation/test period, short local federated budgets, simulated ownership, and differing model capacities limit broader conclusions. The federated model is univariate; the centralized graph benchmark is multivariate. Their absolute errors should be compared within each experiment, not across the two tables. Peak memory is a process high-water mark. Communication estimates include tensor values only, excluding serialization, metrics, configuration, transport framing, latency, and energy. Raw data provenance and the distribution's license declarations are recorded in [DATA.md](../DATA.md).

## 9. Reproducibility

[README commands](../README.md) fetch hash-checked inputs, run tests, execute the model benchmark, and launch Flower. Experiment records retain resolved configuration, seeds, source/data/dependency hashes, sensor IDs, adjacency, split ranges, scaler values, per-horizon counts, histories, timing definitions, and checkpoint hashes. Historical results remain under `phase1-verified` and `flower-verified` Git tags. The benchmark source commit is `10066eb159ed60cb82ee16c132efa3a5a2b79365`; source content hashes identify its executed working tree. Timings and process IDs are not treated as deterministic numerical outputs.

The benchmark and Flower runs use separate output directories and never overwrite completed experiments. Raw HDF5 files, topology CSVs, checkpoints, generated client partitions, and local Flower state are excluded from Git. Unit tests cover schema validation, chronological boundaries, train-only scaling, masking, horizon pooling, graph direction/alignment, changed-edge predictions, LSTM gradients, actual Flower aggregation and message dispatch, and failed launch states. The experiment CLIs provide the real-data integration runs.

## References

1. Li, Y., Yu, R., Shahabi, C., Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR. https://arxiv.org/abs/1707.01926
2. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP. https://arxiv.org/abs/1406.1078
3. Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. McMahan, H. B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS. https://arxiv.org/abs/1602.05629
5. Beutel, D. J. et al. (2020). Flower: A Friendly Federated Learning Research Framework. https://arxiv.org/abs/2007.14390
6. Flower 1.42.0 ClientApp, ServerApp and FedAvg documentation. https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
