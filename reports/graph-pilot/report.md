# Graph-GRU pilot

The pilot uses 16 sensors selected by breadth-first traversal of the published DCRNN distance graph, the first seven days of METR-LA, seed 42, and one training epoch. The selected graph has 85 directed non-self edges. Sensor IDs, distance-kernel scale, adjacency weights, file hashes, and split ranges are recorded in `protocol.json`.

| Model | Test MAE (mph) | Training updates (s) | Parameters | Peak process memory (MiB) |
|---|---:|---:|---:|---:|
| Persistence | 4.917716 | 0 | 0 | 319.35 |
| GRU | 6.996140 | 0.129335 | 11,136 | 319.94 |
| Graph-GRU | 7.278309 | 0.269741 | 15,760 | 319.89 |

The Graph-GRU computes directed incoming and outgoing distance-weighted means, combines them with each sensor's own normalized speed through a shared learned projection, and passes the resulting features to a GRU. The graph scale uses published distances among the 207 aligned sensors; no speed correlation or held-out target is used to construct the graph. Missing-value handling, chronological splits, and training-only normalization reuse the existing pipeline.

All 65 tests passed, including graph-direction, sensor-alignment, node-permutation, gradient, and changed-connection tests. The one-epoch pilot checks execution and cost. Its learned models remain behind persistence. The larger benchmark fixes its configuration before training and reports all selected seeds.
