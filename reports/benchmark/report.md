# Centralized model benchmark

## 3. Data and preprocessing

The checksum-verified HDF5 contains 34,272 five-minute timestamps and 207 sensors, from March 1 through June 27, 2012: 7,094,304 speed entries. The benchmark selects 16 sensors, giving 548,352 speed entries before missing-value masking. Raw timestamps are retained without assigning an undocumented timezone. Numeric columns, unique sensor IDs, increasing timestamps, five-minute spacing, finite nonnegative observations, and file hashes are checked before selection.

Rows split 70/10/20 before sequence construction. Training spans March 1 00:00 to May 23 07:05, validation May 23 07:10 to June 4 04:40, and test June 4 04:45 to June 27 23:55. These contain 23,967, 3,404, and 6,832 windows respectively. Every window stays within its split. Twelve observed time steps predict the next twelve steps, covering 5-60 minute forecast horizons.

Zero and NaN readings are missing. Inputs forward-fill within their split and use the sensor's training mean for leading gaps. A mean and population standard deviation are fitted to each sensor's observed training readings; constant sensors use a scale of one. Missing targets contribute to neither loss nor scores. The same windows, statistics, masks, and batch ordering are used across model architectures for a given seed. The zero-as-missing convention can exclude genuine stopped traffic.

## 4. Distance graph and architecture

The author-distributed distance table is pinned to revision `602afd9d767d3aa1c9b3eac51710d6aeee12c227` and checked by SHA-256. Endpoints outside the canonical 207 METR-LA sensors are excluded. Their aligned finite distance costs set a population standard deviation s. For a supplied directed cost D[i,j], W[i,j] = exp(-(D[i,j]/s)^2), retaining values at least 0.1 and removing self pairs. No speed correlation, validation score, or test target enters graph construction.

A breadth-first traversal from the first canonical sensor selects a connected 16-sensor subset using the union of edge directions, with stronger weights first and ID tie-breaking. Its induced graph contains 85 directed relationships. Full IDs, coordinates, weights, source hashes, and the kernel scale are in [protocol.json](protocol.json). Edges express the published distance relationships; straight lines in the figure are not road polylines.

Graph-GRU computes separate row-normalized incoming and outgoing means. At each sensor and time step, a shared Linear(3,4) plus ReLU transforms its own normalized speed and two neighbor means into four features. The 16 node embeddings flatten to 64 features, enter a one-layer GRU with hidden width 32, and feed a linear head producing 12 x 16 forecasts. The self-only ablation sets both neighbor operators to zero while retaining the same learned projection, recurrent encoder, head, initialization rule, and parameter count.

The GRU reference has 16 input channels, 32 hidden units, and the same direct horizon head. LSTM replaces the recurrent encoder with a 32-unit LSTM. These models have different parameter counts; equal epochs and example exposure do not imply equal parameter counts or FLOPs.

![Selected topology](topology.png)

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

![Forecast horizon errors](horizons.png)

![Validation and training convergence](convergence.png)

### Graph ablation

Graph-GRU minus self-only MAE differences by seed are -0.013652, +0.015787, -0.015134 mph. Their mean is -0.004333 mph. Graph-GRU improves over the raw-input GRU in all three runs, but the edge contribution is small and changes sign. The self-only projection retains most of the mean MAE improvement over the smaller GRU reference. LSTM has nearly the same mean MAE as Graph-GRU, with fewer parameters and lower seed dispersion in this configuration.

Whole-process peak memory was approximately 320 MiB for all architectures. Runtime and preprocessing dominate this measurement, so those small differences should not be interpreted as architecture-specific memory savings.


[Full research report](../research-report.md)
