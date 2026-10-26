from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SHA256, URL, Scaler, Windows, digest, load, missing_report, split
from .model import GRU, Metrics


@dataclass
class Config:
    sensors: int = 16
    rows: int | None = None
    history: int = 12
    horizon: int = 12
    hidden: int = 32
    batch_size: int = 64
    epochs: int = 5
    lr: float = .001
    seed: int = 42
    threads: int = 4

    def validate(self):
        for name in ("sensors", "history", "horizon", "hidden", "batch_size", "epochs", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.rows is not None and (type(self.rows) is not int or self.rows < 1):
            raise ValueError("rows must be null or a positive integer")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def evaluate(model, loader, mean, std, horizon):
    scores = Metrics(horizon)
    if model is not None:
        model.eval()
    with torch.no_grad():
        for x, y, mask in loader:
            if model is None:
                prediction = (x[:, -1:] * std + mean).expand(-1, horizon, -1)
            else:
                prediction = model(x) * std + mean
            scores.update(prediction, y, mask)
    return scores.result()


def plots(train, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = train.mask(train.eq(0))
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    observed.iloc[:288, :min(4, observed.shape[1])].plot(ax=axes[0], linewidth=.8)
    axes[0].set(title="First training day", ylabel="Speed (mph)", xlabel="Timestamp (timezone unspecified)")
    observed.stack().plot.hist(bins=40, ax=axes[1])
    axes[1].set(title="Observed training speeds", xlabel="Speed (mph)")
    (1 - observed.notna().mean()).plot.bar(ax=axes[2])
    axes[2].set(title="Training missing fraction: zero or NaN", ylabel="Fraction", xlabel="Sensor ID", ylim=(0, 1))
    fig.savefig(Path(out) / "training_eda.png", dpi=130)
    plt.close(fig)


def code_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
                   + list((root / "configs").glob("*.json"))
                   + [root / "pyproject.toml", root / "requirements-lock.txt"])
    return {str(path.relative_to(root)).replace(os.sep, "/"): digest(path) for path in paths if path.exists()}


def run(config, data, out):
    config.validate()
    seed_all(config.seed, config.threads)
    frame = load(data, config.sensors, config.rows)
    parts = split(frame, config.history, config.horizon)
    scaler = Scaler().fit(parts["train"])
    datasets = {name: Windows(part, scaler, config.history, config.horizon) for name, part in parts.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "running"})
    write_json(out / "config.json", asdict(config))
    write_json(out / "missing.json", {name: missing_report(part) for name, part in {"selected": frame, **parts}.items()})
    write_json(out / "scaler.json", {"fit": "train observed values only", "sensor_ids": list(map(str, frame.columns)),
        "mean": scaler.mean.tolist(), "std": scaler.std.tolist()})
    source = code_manifest()
    identity = {"config": asdict(config), "data_sha256": SHA256, "code": source}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                         capture_output=True, text=True)
    write_json(out / "manifest.json", {
        "run_id": run_id, **identity, "data_url": URL,
        "selected_values_sha256": hashlib.sha256(frame.to_numpy().tobytes()).hexdigest(),
        "sensor_ids": list(map(str, frame.columns)), "python": sys.version,
        "platform": platform.platform(), "device": "cpu", "deterministic_algorithms": True,
        "mkldnn": False, "command": sys.argv,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "versions": {name: version(name) for name in ("numpy", "pandas", "tables", "matplotlib", "torch")},
        "splits": {name: {"start": str(part.index[0]), "end": str(part.index[-1]),
            "rows": len(part), "windows": len(datasets[name])} for name, part in parts.items()},
        "protocol": "70/10/20 rows; windows contained in each split; zero/NaN masked; train-only per-sensor scaling",
    })
    plots(parts["train"], out)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {name: DataLoader(ds, batch_size=config.batch_size, shuffle=name == "train",
                               num_workers=0, generator=generator if name == "train" else None)
               for name, ds in datasets.items()}
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    model = GRU(config.sensors, config.hidden, config.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    best = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, target_count = 0., 0
        for x, y, mask in loaders["train"]:
            count = int(mask.sum())
            if count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x)
            target = (y - mean) / std
            loss = (predicted[mask] - target[mask]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item() * count
            target_count += count
        if target_count == 0:
            raise ValueError("No observed training targets")
        validation = evaluate(model, loaders["val"], mean, std, config.horizon)
        row = {"epoch": epoch, "train_normalized_mae": loss_sum / target_count, "val": validation}
        history.append(row)
        print(f"epoch {epoch}: val MAE {validation['mae']:.4f} mph", flush=True)
        if validation["mae"] < best:
            best = validation["mae"]
            torch.save(model.state_dict(), out / "best.pt")
        write_json(out / "history.json", history)
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True))
    results = {"units": "mph", "selection": "lowest validation MAE; test evaluated after selection",
        "best_epoch": min(history, key=lambda row: row["val"]["mae"])["epoch"],
        "persistence": {}, "gru": {}}
    for name in ("val", "test"):
        results["persistence"][name] = evaluate(None, loaders[name], mean, std, config.horizon)
        results["gru"][name] = evaluate(model, loaders[name], mean, std, config.horizon)
    write_json(out / "metrics.json", results)
    write_json(out / "status.json", {"status": "complete", "run_id": run_id})
    return results
