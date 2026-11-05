from pathlib import Path
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REVISION = "ec9772c0e8dd48b3470a6381dff1888257a9ffe4"
URL = (
    f"https://huggingface.co/datasets/jimmygao3218/METRLA/resolve/{REVISION}/metr-la.h5"
)
SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, partial, timeout=120, retries=3, delay=2.0):
    """Download url into partial, retrying network errors and 5xx with backoff.

    Client errors (like a 404) aren't retried, and the caller checks the checksum.
    """
    partial = Path(partial)
    for attempt in range(retries + 1):
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as response,
                partial.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            return partial
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == retries:
                raise
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ):
            if attempt == retries:
                raise
        time.sleep(delay * 2**attempt)


def fetch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != SHA256:
            raise ValueError(
                "Existing dataset checksum differs from the pinned METR-LA file"
            )
        return path
    partial = path.with_suffix(".partial")
    try:
        download(URL, partial)
        if digest(partial) != SHA256:
            raise ValueError("Downloaded dataset checksum mismatch")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    path.with_suffix(".source.json").write_text(
        json.dumps(
            {
                "url": URL,
                "revision": REVISION,
                "sha256": SHA256,
                "mirror_declared_license": "CC-BY-4.0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def validate(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Expected valid datetime timestamps")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing")
    if (
        len(frame) < 2
        or not (np.diff(frame.index.asi8) == pd.Timedelta(minutes=5).value).all()
    ):
        raise ValueError(
            "Expected a continuous five-minute grid; do not silently bridge gaps"
        )
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
    frame.index = pd.DatetimeIndex(
        frame.index.to_numpy(copy=True), name=frame.index.name
    )
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
        "rows": len(frame),
        "sensors": len(frame.columns),
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
        "nan_count": int(frame.isna().sum().sum()),
        "zero_count": int(frame.eq(0).sum().sum()),
        "missing_fraction": float(missing.to_numpy().mean()),
        "per_sensor": {
            str(sensor): {
                "nan": int(frame[sensor].isna().sum()),
                "zero": int(frame[sensor].eq(0).sum()),
                "missing_fraction": float(missing[sensor].mean()),
            }
            for sensor in frame.columns
        },
    }


def split(frame, history=12, horizon=12):
    n = len(frame)
    a, b = int(n * 0.7), int(n * 0.8)
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
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        return self

    def inputs(self, frame):
        observed = frame.mask(frame.eq(0))
        # Forward fill only: every input uses observations from its own past.
        filled = observed.ffill().fillna(dict(zip(frame.columns, self.mean)))
        return ((filled.to_numpy(dtype=np.float32) - self.mean) / self.std).astype(
            np.float32
        )


TIME_CHANNELS = 3


def calendar_features(index):
    """Time of day (sin, cos) and a weekend flag for each timestamp, as float32.

    They only depend on the clock, so they're known for every forecast target.
    """
    index = pd.DatetimeIndex(index)
    minute = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    phase = 2 * np.pi * minute / 1440
    weekend = index.dayofweek.to_numpy() >= 5
    return np.stack([np.sin(phase), np.cos(phase), weekend], axis=1).astype(np.float32)


class Windows(Dataset):
    """Chronological input/target windows.

    With time_features the calendar columns come after the sensor channels, so
    x[..., :sensors] is always the speeds. Targets and masks don't change.
    """

    def __init__(self, frame, scaler, history=12, horizon=12, time_features=False):
        if history < 1 or horizon < 1 or len(frame) < history + horizon:
            raise ValueError("Invalid window sizes")
        self.x = scaler.inputs(frame)
        self.extra = 0
        if time_features:
            self.x = np.concatenate([self.x, calendar_features(frame.index)], axis=1)
            self.extra = TIME_CHANNELS
        self.times = frame.index
        raw = frame.to_numpy(dtype=np.float32)
        self.mask = np.isfinite(raw) & (raw != 0)
        self.y = np.where(self.mask, raw, 0).astype(np.float32)
        self.history, self.horizon = history, horizon
        self.mean, self.std = scaler.mean, scaler.std

    def __len__(self):
        return len(self.x) - self.history - self.horizon + 1

    def origin_days(self):
        """Day of each window's first forecast target, as codes plus labels (for resampling days)."""
        first = self.times[self.history : self.history + len(self)]
        codes, labels = pd.factorize(first.normalize())
        return codes.astype(np.int64), [str(day.date()) for day in labels]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index + self.history
        stop = start + self.horizon
        return (
            torch.from_numpy(self.x[index:start]),
            torch.from_numpy(self.y[start:stop]),
            torch.from_numpy(self.mask[start:stop]),
        )
