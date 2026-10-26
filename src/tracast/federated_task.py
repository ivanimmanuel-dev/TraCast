"""Shared single-sensor task for federated and pooled-central training."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .data import Scaler, Windows, split, missing_report
from .experiment import seed_all
from .model import GRU, Metrics


@dataclass
class FederatedConfig:
    clients: int = 3
    sensors_per_client: int = 2
    rows: int = 2016
    history: int = 12
    horizon: int = 12
    hidden: int = 32
    batch_size: int = 64
    rounds: int = 3
    local_epochs: int = 1
    lr: float = .001
    seed: int = 42
    threads: int = 1
    ray_cpus: int = 2
    client_cpus: int = 2
    round_timeout: int = 180
    partition_rule: str = "contiguous"

    def validate(self):
        for name in self.__dataclass_fields__:
            if name in ("lr", "partition_rule"):
                continue
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} has an invalid integer value")
        if self.partition_rule not in ("contiguous", "balanced", "congestion"):
            raise ValueError("Unknown sensor partition rule")
        if self.clients < 2 or self.clients * self.sensors_per_client > 207:
            raise ValueError("Need at least two nonempty clients within 207 sensors")
        if not 1 <= self.rows <= 34272 or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Invalid rows or learning rate")
        if self.ray_cpus < self.client_cpus:
            raise ValueError("Ray needs enough CPUs for one client")
        if self.seed + 10000 * self.rounds + self.clients >= 2**32:
            raise ValueError("Seed schedule exceeds NumPy range")


def congestion_scores(frame, config):
    train=split(frame,config.history,config.horizon)["train"].mask(lambda value:value.eq(0))
    count=train.notna().sum()
    if (count==0).any():
        raise ValueError("A sensor has no observed training values")
    return {str(sensor):float(train[sensor].lt(40).sum()/count[sensor]) for sensor in train.columns}


def partitions(frame, config):
    expected = config.clients * config.sensors_per_client
    if frame.shape[1] != expected or not frame.columns.is_unique:
        raise ValueError("Selected sensor count must match the disjoint partitions")
    if config.partition_rule == "contiguous":
        return [frame.iloc[:, i * config.sensors_per_client:(i + 1) * config.sensors_per_client].copy()
                for i in range(config.clients)]
    scores=congestion_scores(frame,config)
    ranked=sorted(frame.columns,key=lambda sensor:(scores[str(sensor)],str(sensor)))
    if config.partition_rule == "balanced":
        groups=[ranked[i::config.clients] for i in range(config.clients)]
    elif config.partition_rule == "congestion":
        groups=[ranked[i*config.sensors_per_client:(i+1)*config.sensors_per_client] for i in range(config.clients)]
    else:
        raise ValueError("Unknown sensor partition rule")
    return [frame.loc[:,sensors].copy() for sensors in groups]


def distribution(frame, config):
    train=split(frame,config.history,config.horizon)["train"]
    values=train.to_numpy().reshape(-1)
    values=values[np.isfinite(values)&(values>0)]
    if not len(values):
        raise ValueError("No observed client training values")
    hist,_=np.histogram(values,bins=[0,10,20,30,40,50,60,70,80,np.inf])
    return {"observed_readings":len(values),"mean_mph":float(values.mean()),
        "congestion_fraction_below_40":float((values<40).mean()),
        "speed_histogram":hist.tolist(),"histogram_bins_mph":[0,10,20,30,40,50,60,70,80,"+inf"]}


class SensorWindows(Dataset):
    """One sensor per example; reuse Phase 1 windows without materializing them."""
    def __init__(self, frame, scaler, history, horizon, training=False):
        self.windows = []
        self.indices = []
        for sensor in range(frame.shape[1]):
            local = Scaler()
            local.mean, local.std = scaler.mean[sensor:sensor+1], scaler.std[sensor:sensor+1]
            windows = Windows(frame.iloc[:, sensor:sensor+1], local, history, horizon)
            self.windows.append(windows)
            for index in range(len(windows)):
                start = index + history
                if not training or windows.mask[start:start+horizon].any():
                    self.indices.append((sensor, index))
        self.candidate_count = len(self.windows) * (len(frame) - history - horizon + 1)
        if not self.indices:
            raise ValueError("No usable windows")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sensor, position = self.indices[index]
        windows = self.windows[sensor]
        x, y, mask = windows[position]
        return x, y, mask, torch.from_numpy(windows.mean.copy()).reshape(1, 1), torch.from_numpy(windows.std.copy()).reshape(1, 1)


def bundle(frame, config):
    parts = split(frame, config.history, config.horizon)
    scaler = Scaler().fit(parts["train"])
    datasets = {name: SensorWindows(part, scaler, config.history, config.horizon, training=name == "train")
                for name, part in parts.items()}
    metadata = {
        "sensor_ids": list(map(str, frame.columns)),
        "training_distribution": distribution(frame,config),
        "normalization": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist(), "fit": "local observed train values only"},
        "splits": {name: {"first": str(part.index[0]), "last": str(part.index[-1]), "rows": len(part),
            "examples": len(datasets[name]), "candidate_examples": datasets[name].candidate_count}
            for name, part in parts.items()},
        "missing": {name: missing_report(part) for name, part in parts.items()},
    }
    return datasets, metadata


def load_partition(job, client_id):
    job = Path(job)
    config = FederatedConfig(**json.loads((job / "config.json").read_text()))
    with np.load(job / "partitions" / f"client-{client_id}.npz", allow_pickle=False) as saved:
        frame = pd.DataFrame(saved["values"], index=pd.DatetimeIndex(saved["timestamps"]), columns=saved["sensors"].tolist())
    return bundle(frame, config)


def new_model(config):
    return GRU(1, config.hidden, config.horizon)


def loader(dataset, config, seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=seed is not None,
                      num_workers=0, generator=generator)


def train_epochs(model, dataset, config, seed):
    seed_all(seed, config.threads)
    model.train()
    # Reset at every local fit and centralized epoch block; no optimizer state is federated.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_sum, target_count, steps, examples = 0., 0, 0, 0
    batches = loader(dataset, config, seed)
    for _ in range(config.local_epochs):
        for x, y, mask, mean, std in batches:
            count = int(mask.sum())
            if count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)[mask] - ((y - mean) / std)[mask]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite local loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item() * count
            target_count += count
            examples += len(x)
            steps += 1
    if not target_count:
        raise ValueError("No observed training targets")
    return {"loss-sum": loss_sum, "target-count": target_count, "optimizer-steps": steps,
            "processed-examples": examples, "train-normalized-mae": loss_sum / target_count}


def score(model, dataset, config):
    metrics = Metrics(config.horizon)
    if model is not None:
        model.eval()
    with torch.no_grad():
        for x, y, mask, mean, std in loader(dataset, config):
            prediction = model(x) if model is not None else x[:, -1:].expand(-1, config.horizon, -1)
            metrics.update(prediction * std + mean, y, mask)
    return {"ae": metrics.absolute.tolist(), "se": metrics.squared.tolist(), "count": metrics.count.tolist(),
            "num-examples": len(dataset)}


def pool(records):
    if not records:
        raise ValueError("No evaluation records")
    horizon = len(records[0]["count"])
    if any(any(len(record[key]) != horizon for key in ("ae", "se", "count")) for record in records):
        raise ValueError("Inconsistent horizon statistics")
    metrics = Metrics(horizon)
    for record in records:
        ae, se = np.asarray(record["ae"]), np.asarray(record["se"])
        counts = np.asarray(record["count"])
        if not (np.isfinite(ae).all() and np.isfinite(se).all()) or (ae < 0).any() or (se < 0).any() or (counts < 0).any():
            raise ValueError("Invalid sufficient statistics")
        metrics.absolute += torch.tensor(ae, dtype=torch.float64)
        metrics.squared += torch.tensor(se, dtype=torch.float64)
        metrics.count += torch.tensor(counts, dtype=torch.int64)
    return metrics.result()


def parameter_bytes(model):
    return sum(value.numel() * value.element_size() for value in model.state_dict().values())


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
