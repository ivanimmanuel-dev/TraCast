# METR-LA provenance

## Origin and distribution

Original research: Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018.

- Paper: https://arxiv.org/abs/1707.01926
- Author repository and original distribution instructions: https://github.com/liyaguang/DCRNN
- Author-linked data folder: https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX

TraCast downloads the HDF5 file from the accessible `jimmygao3218/METRLA` mirror, without using its graph pickle or any precomputed splits:

- Mirror: https://huggingface.co/datasets/jimmygao3218/METRLA
- Immutable revision: `ec9772c0e8dd48b3470a6381dff1888257a9ffe4`
- File: https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5
- Published file metadata: https://huggingface.co/datasets/jimmygao3218/METRLA/blob/ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5
- SHA-256: `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
- Bytes downloaded and verified on October 26, 2026: `57038056`

The download matches the mirror's published content hash and METR-LA's schema, dates, and sensor count.

## License and attribution

The pinned mirror's dataset card declares **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ . Its saved card is [docs/mirror-card.md](docs/mirror-card.md). Attribute the original DCRNN authors and the mirror uploader, link the sources, and identify preprocessing changes when sharing derived material.

The DCRNN repository has an **MIT software license**, copyright 2017 Yaguang Li. A copy is in [docs/DCRNN-LICENSE.txt](docs/DCRNN-LICENSE.txt), sourced from https://github.com/liyaguang/DCRNN/blob/master/LICENSE . The original traffic-data distribution does not specify separate dataset terms; the mirror's CC BY declaration and the upstream software license are distinct.

Raw datasets remain local and excluded from Git. TraCast code is reserved to the repository owner; the upstream notice covers the DCRNN material.

## Processing

Speeds remain in mph. Zeros and NaNs are treated as missing. The raw file is unchanged. The baseline selects the first N sensor columns; the graph benchmark selects sensors by topology. Processing splits rows by time, fits observed training statistics, forward-fills inputs within each split, and masks missing targets. Experiment manifests record the exact sensor IDs, ranges, statistics, and input hash.

```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```


## Sensor-distance graph

The graph experiment uses the DCRNN author's sensor-distance CSV, canonical sensor IDs, and location table from revision `602afd9d767d3aa1c9b3eac51710d6aeee12c227`:

- [Distance costs](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/distances_la_2012.csv)
- [Canonical IDs](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_ids.txt)
- [Sensor locations](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv)
- [Upstream kernel construction](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/gen_adj_mx.py)

| File | Bytes | SHA-256 |
|---|---:|---|
| distances_la_2012.csv | 6,393,348 | `a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5` |
| graph_sensor_ids.txt | 1,448 | `3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990` |
| graph_sensor_locations.csv | 6,341 | `eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f` |

The distance table includes other sensors. Only pairs whose endpoints belong to the aligned 207 METR-LA sensors enter the graph. Their 11,753 finite costs set the Gaussian scale. A weight is `exp(-(cost / std)^2)`, retained at 0.1 or above, with self pairs removed. This yields 1,515 directed non-self relationships among the 207 sensors. The 16-sensor subset is chosen by breadth-first traversal from the first canonical sensor, using the union of directions and ordering neighbors by descending weight then ID. Its induced graph has 85 directed relationships. Neither selection nor weights use traffic-speed values.

The model preserves direction: incoming and outgoing weights are normalized separately. Isolated neighbor channels contribute zero; each sensor's own value is a separate input. Displayed lines join supplied sensor coordinates and indicate graph relationships, not road polylines. The files remain in ignored `data/topology/`; hashes, selected IDs, coordinates, and derived weights are retained in experiment protocols. The upstream repository's MIT notice is retained in `docs/DCRNN-LICENSE.txt`.
