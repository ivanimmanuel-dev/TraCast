# Graph-GRU pilot

A quick check that the graph code runs: 16 sensors picked by walking the DCRNN distance graph, the first seven days of METR-LA, seed 42 and one epoch. The graph has 85 directed edges. Sensor IDs, the kernel scale, the edge weights, file hashes and split ranges are in [protocol.json](protocol.json).

| Model | Test MAE (mph) | Training updates (s) | Parameters | Peak process memory (MiB) |
|---|---:|---:|---:|---:|
| Persistence | 4.917716 | 0 | 0 | 319.35 |
| GRU | 6.996140 | 0.129335 | 11,136 | 319.94 |
| Graph-GRU | 7.278309 | 0.269741 | 15,760 | 319.89 |

Graph-GRU takes directed incoming and outgoing distance-weighted averages, combines them with each sensor's own normalized speed through a shared learned layer and feeds the result to a GRU. The graph uses the published distances between the 207 sensors, so no speed data or held-out target went into building it. Splits, missing-value handling and normalization are the same as everywhere else.

After one epoch both learned models are behind persistence, which is why the [full benchmark](../benchmark/report.md) uses five epochs, the whole date range, three seeds, an LSTM and a self-only graph ablation.
