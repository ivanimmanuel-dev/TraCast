# Data

## METR-LA

METR-LA comes from *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting* by Yaguang Li, Rose Yu, Cyrus Shahabi and Yan Liu (ICLR 2018).

- Paper: https://arxiv.org/abs/1707.01926
- The authors' repo, with the original download instructions: https://github.com/liyaguang/DCRNN
- The original Google Drive folder: https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX

The Drive folder is awkward to script, so TraCast downloads the HDF5 file from a Hugging Face mirror, `jimmygao3218/METRLA`. It ignores the mirror's graph pickle and precomputed splits.

- Mirror: https://huggingface.co/datasets/jimmygao3218/METRLA
- Revision: `ec9772c0e8dd48b3470a6381dff1888257a9ffe4`
- File: https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5
- SHA-256: `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
- Size: 57,038,056 bytes (downloaded and checked on October 26, 2026)

`tracast fetch` checks that hash, and loading the file checks it again along with the schema, the dates and the sensor count.

## License

The mirror's dataset card says CC BY 4.0 ([terms](https://creativecommons.org/licenses/by/4.0/)); a copy of the card is in [docs/mirror-card.md](docs/mirror-card.md). If you share anything derived from the data, credit the DCRNN authors and the mirror uploader, link the sources and say what preprocessing you did.

The DCRNN repo is MIT licensed, copyright 2017 Yaguang Li, and its notice is in [docs/DCRNN-LICENSE.txt](docs/DCRNN-LICENSE.txt) (from the [upstream license](https://github.com/liyaguang/DCRNN/blob/master/LICENSE)). The original data release doesn't state separate terms for the data, so the mirror's CC BY label and the upstream software license are two different things.

The raw data stays out of git. TraCast's own code is MIT licensed, see [LICENSE](LICENSE).

## How TraCast uses it

Speeds stay in mph, zeros and NaNs count as missing, and the raw file is never changed. The early experiments take the first N sensor columns, and the graph experiments pick sensors from the road graph. Rows are split by time, the training statistics come from observed training readings only, inputs are forward-filled within each split and missing targets are masked. Each experiment's manifest records the sensor IDs, ranges, statistics and input hash.

```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## Sensor graph

The graph experiments use the DCRNN authors' distance table, sensor IDs and sensor locations from revision `602afd9d767d3aa1c9b3eac51710d6aeee12c227`:

- [Distance costs](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/distances_la_2012.csv)
- [Sensor IDs](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_ids.txt)
- [Sensor locations](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv)
- [How upstream builds the kernel](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/gen_adj_mx.py)

| File | Bytes | SHA-256 |
|---|---:|---|
| distances_la_2012.csv | 6,393,348 | `a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5` |
| graph_sensor_ids.txt | 1,448 | `3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990` |
| graph_sensor_locations.csv | 6,341 | `eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f` |

The distance table also lists sensors outside METR-LA, so only pairs between the 207 METR-LA sensors are used. Their 11,753 finite costs set the standard deviation of a Gaussian kernel. A weight is `exp(-(cost / std)^2)`, kept when it is 0.1 or more, with self pairs dropped. That leaves 1,515 directed edges among the 207 sensors.

The 16-sensor subset comes from a breadth-first walk from the first canonical sensor, following edges in either direction and visiting stronger ones first (ties broken by ID). Its induced graph has 85 directed edges. Neither the choice of sensors nor the weights use any speed data.

The model keeps edge direction: incoming and outgoing weights are normalized separately, a sensor with no neighbors gets zeros, and each sensor's own speed is a separate input. The lines in the topology plots join the sensors' coordinates to show graph edges, not road shapes. The files live in the ignored `data/topology/`; the hashes, chosen IDs, coordinates and weights are saved in each experiment's protocol file.
