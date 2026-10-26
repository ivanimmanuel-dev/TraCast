from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REVISION = "ec9772c0e8dd48b3470a6381dff1888257a9ffe4"
URL = f"https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/{REVISION}/metr-la.h5"
SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != SHA256:
            raise ValueError("Existing dataset checksum differs from the pinned METR-LA file")
        return path
    partial = path.with_suffix(".partial")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if digest(partial) != SHA256:
            raise ValueError("Downloaded dataset checksum mismatch")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    path.with_suffix(".source.json").write_text(json.dumps({
        "url": URL, "revision": REVISION, "sha256": SHA256,
        "mirror_declared_license": "CC-BY-4.0",
    }, indent=2) + "\n", encoding="utf-8")
    return path


def validate(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Expected valid datetime timestamps")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing")
    if len(frame) < 2 or not (np.diff(frame.index.asi8) == pd.Timedelta(minutes=5).value).all():
        raise ValueError("Expected a continuous five-minute grid; do not silently bridge gaps")
    if not frame.columns.is_unique or frame.columns.hasnans:
        raise ValueError("Sensor IDs must be unique and nonmissing")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("Speed columns must be numeric")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Infinite or negative speed")


def load(path, sensors=16, rows=None):
    if digest(path) != SHA256:
        raise ValueError("Expected the pinned METR-LA checksum")
    frame = pd.read_hdf(path, key="df")
    # The original HDF5 stores legacy pandas frequency metadata as bytes.
    # Rebuild from timestamps, then validate their actual spacing.
    frame.index = pd.DatetimeIndex(frame.index.to_numpy(copy=True), name=frame.index.name)
    validate(frame)
    if frame.shape != (34272, 207):
        raise ValueError(f"Unexpected METR-LA shape: {frame.shape}")
    if not 1 <= sensors <= frame.shape[1]:
        raise ValueError("Invalid sensor count")
    if rows is not None and not 1 <= rows <= len(frame):
        raise ValueError("Invalid row limit")
    return frame.iloc[:rows, :sensors].astype("float32")


def missing_report(frame):
    missing = frame.isna() | frame.eq(0)
    return {
        "rows": len(frame), "sensors": len(frame.columns),
        "start": str(frame.index[0]), "end": str(frame.index[-1]),
        "nan_count": int(frame.isna().sum().sum()),
        "zero_count": int(frame.eq(0).sum().sum()),
        "missing_fraction": float(missing.to_numpy().mean()),
        "per_sensor": {str(sensor): {"nan": int(frame[sensor].isna().sum()),
            "zero": int(frame[sensor].eq(0).sum()), "missing_fraction": float(missing[sensor].mean())}
            for sensor in frame.columns},
    }


def split(frame, history=12, horizon=12):
    n = len(frame)
    a, b = int(n * .7), int(n * .8)
    parts = {"train": frame.iloc[:a], "val": frame.iloc[a:b], "test": frame.iloc[b:]}
    if min(map(len, parts.values())) < history + horizon:
        raise ValueError("Each chronological split needs history + horizon rows")
    return parts


class Scaler:
    def fit(self, train):
        observed = train.mask(train.eq(0))
        self.mean = observed.mean().to_numpy(dtype=np.float32)
        self.std = observed.std(ddof=0).to_numpy(dtype=np.float32)
        if not np.isfinite(self.mean).all():
            raise ValueError("A selected sensor has no observed training values")
        self.std = np.where(self.std < 1e-6, 1., self.std).astype(np.float32)
        return self

    def inputs(self, frame):
        observed = frame.mask(frame.eq(0))
        # Forward fill only: every input uses observations from its own past.
        filled = observed.ffill().fillna(dict(zip(frame.columns, self.mean)))
        return ((filled.to_numpy(dtype=np.float32) - self.mean) / self.std).astype(np.float32)


class Windows(Dataset):
    def __init__(self, frame, scaler, history=12, horizon=12):
        if history < 1 or horizon < 1 or len(frame) < history + horizon:
            raise ValueError("Invalid window sizes")
        self.x = scaler.inputs(frame)
        raw = frame.to_numpy(dtype=np.float32)
        self.mask = np.isfinite(raw) & (raw != 0)
        self.y = np.where(self.mask, raw, 0).astype(np.float32)
        self.history, self.horizon = history, horizon
        self.mean, self.std = scaler.mean, scaler.std

    def __len__(self):
        return len(self.x) - self.history - self.horizon + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index + self.history
        stop = start + self.horizon
        return (torch.from_numpy(self.x[index:start]),
                torch.from_numpy(self.y[start:stop]), torch.from_numpy(self.mask[start:stop]))
