# METR-LA provenance

## Origin and distribution

Original research: Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018.

- Paper: https://arxiv.org/abs/1707.01926
- Author repository and original distribution instructions: https://github.com/liyaguang/DCRNN
- Author-linked data folder: https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX

TrafficMind downloads the HDF5 file from the accessible `jimmygao3218/METRLA` mirror, without using its graph pickle or any precomputed splits:

- Mirror: https://huggingface.co/datasets/jimmygao3218/METRLA
- Immutable revision: `ec9772c0e8dd48b3470a6381dff1888257a9ffe4`
- File: https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5
- Published file metadata: https://huggingface.co/datasets/jimmygao3218/METRLA/blob/ec9772c0e8dd48b3470a6381dff1888257a9ffe4/metr-la.h5
- SHA-256: `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
- Bytes downloaded and verified on October 26, 2026: `57038056`

The downloaded bytes match the mirror's published content hash. The original author-linked Google Drive copy was not independently downloaded and compared. The mirror's schema, dates, and sensor count match METR-LA. No claim is made that this independently establishes the original data's chain of custody.

## License and attribution

The pinned mirror's dataset card declares **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ . Its saved card is [docs/mirror-card.md](docs/mirror-card.md). Attribute the original DCRNN authors and the mirror uploader, link the sources, and identify preprocessing changes when sharing derived material.

The DCRNN repository has an **MIT software license**, copyright 2017 Yaguang Li. A copy is in [docs/DCRNN-LICENSE.txt](docs/DCRNN-LICENSE.txt), sourced from https://github.com/liyaguang/DCRNN/blob/master/LICENSE . That software license alone does not establish a separate license grant from the original traffic-data provider. The mirror's authority to relicense the underlying data has not been independently established. The original distribution README does not supply separate dataset-specific terms.

TrafficMind keeps raw data local and excluded from Git. This repository does not redistribute the HDF5 file. No project-wide license is selected for the new TrafficMind code yet; the upstream license file applies to its upstream material.

## Processing

Speeds remain in mph. Zeros and NaNs are treated as missing. The raw file is unchanged. Processing selects the first N sensor columns, splits rows by time, fits observed training statistics, forward-fills inputs within each split, and masks missing targets. Experiment manifests record the exact sensor IDs, ranges, statistics, and input hash.

```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```
