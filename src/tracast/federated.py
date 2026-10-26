"""Prepare, execute Flower/Ray, then run the comparable centralized GRU."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from .data import SHA256, load, digest
from .experiment import code_manifest, seed_all, write_json
from .federated_task import FederatedConfig, bundle, load_partition, new_model, partitions, pool, score, train_epochs, state_hash


def prepare(config, data, out):
    config.validate()
    frame = load(data, config.clients * config.sensors_per_client, config.rows)
    groups = partitions(frame, config)
    # Validate all client scalers and windows before creating an output directory.
    prepared = [bundle(group, config) for group in groups]
    job = Path(out).resolve()
    job.mkdir(parents=True, exist_ok=False)
    (job / "partitions").mkdir()
    write_json(job / "status.json", {"status": "prepared"})
    write_json(job / "config.json", asdict(config))
    for client_id, group in enumerate(groups):
        np.savez_compressed(job / "partitions" / f"client-{client_id}.npz", values=group.to_numpy(),
                            timestamps=group.index.to_numpy(), sensors=np.asarray(group.columns, dtype=str))
    write_json(job / "clients.json", [{"client_id": i, **metadata} for i, (_, metadata) in enumerate(prepared)])
    source = code_manifest()
    identity = {"config": asdict(config), "data_sha256": SHA256, "code": source}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    partition_hashes = {path.name: digest(path) for path in sorted((job / "partitions").glob("*.npz"))}
    write_json(job / "manifest.json", {**identity,
        "run_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "partition_files": partition_hashes, "python": sys.version, "platform": platform.platform(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None, "command": sys.argv,
        "versions": {name: version(name) for name in ("flwr", "ray", "numpy", "pandas", "torch", "matplotlib")},
        "task": "shared GRU(1): one sensor per example, disjoint sensor ownership",
        "runtime": "Flower CLI / managed local SuperLink / Ray; one actor slot in the default config",
        "deterministic": {"torch_algorithms": True, "mkldnn": False, "aggregate_order": "client-id ascending"}})
    seed_all(config.seed, config.threads)
    model = new_model(config)
    torch.save(model.state_dict(), job / "initial.pt")
    write_json(job / "initial.json", {"state_sha256": state_hash(model)})
    return job


def centralized(config, job):
    bundles = [load_partition(job, i)[0] for i in range(config.clients)]
    datasets = {stage: ConcatDataset([data[stage] for data in bundles]) for stage in ("train", "val", "test")}
    seed_all(config.seed, config.threads)
    model = new_model(config)
    model.load_state_dict(torch.load(job / "initial.pt", map_location="cpu", weights_only=True))
    history, best = [], float("inf")
    for round_id in range(1, config.rounds + 1):
        train = train_epochs(model, datasets["train"], config, config.seed + 10000 * round_id + config.clients)
        validation = pool([score(model, datasets["val"], config)])
        history.append({"round": round_id, "train": train, "validation": validation, "state_sha256": state_hash(model)})
        if validation["mae"] < best:
            best = validation["mae"]
            torch.save(model.state_dict(), job / "best-centralized.pt")
        write_json(job / "centralized-history.json", history)
    model.load_state_dict(torch.load(job / "best-centralized.pt", map_location="cpu", weights_only=True))
    result = {"best_round": min(history, key=lambda row: row["validation"]["mae"])["round"],
        "test": pool([score(model, datasets["test"], config)]), "state_sha256": state_hash(model),
        "train_examples_per_epoch": len(datasets["train"]),
        "processed_examples": sum(row["train"]["processed-examples"] for row in history),
        "optimizer_steps": sum(row["train"]["optimizer-steps"] for row in history),
        "persistence_test": pool([score(None, datasets["test"], config)])}
    write_json(job / "centralized.json", result)
    return result


def plots(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    federated = json.loads((job / "federated.json").read_text())
    central = json.loads((job / "centralized.json").read_text())
    rounds = json.loads((job / "rounds.json").read_text())
    epochs = json.loads((job / "centralized-history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for metric, ax in zip(("mae", "rmse"), axes):
        ax.plot([r["round"] for r in rounds], [r["validation"][metric] for r in rounds], "o-", label="FedAvg")
        ax.plot([r["round"] for r in epochs], [r["validation"][metric] for r in epochs], "o-", label="Centralized")
        ax.set(title=f"Validation {metric.upper()}", xlabel="Round / centralized epoch block", ylabel="mph")
        ax.set_xticks([r["round"] for r in rounds])
        ax.legend()
    fig.savefig(job / "convergence.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for metric, ax in zip(("mae", "rmse"), axes):
        for name, result in [("FedAvg", federated["test"]), ("Centralized", central["test"]), ("Persistence", central["persistence_test"])]:
            ax.plot([r["minutes"] for r in result["per_horizon"]], [r[metric] for r in result["per_horizon"]], label=name)
        ax.set(title=f"Selected-model test {metric.upper()}", xlabel="Forecast horizon (minutes)", ylabel="mph")
        ax.legend()
    fig.savefig(job / "horizon-metrics.png", dpi=140)
    plt.close(fig)


def execute(job):
    job = Path(job).resolve()
    config = FederatedConfig(**json.loads((job / "config.json").read_text()))
    if json.loads((job / "status.json").read_text())["status"] != "prepared":
        raise ValueError("Execute requires a fresh prepared job")
    write_json(job / "status.json", {"status": "running"})
    scripts = Path(sys.executable).parent
    env = os.environ.copy()
    env.update({"FLWR_TELEMETRY_ENABLED": "0", "RAY_USAGE_STATS_ENABLED": "0",
        "FLWR_DISABLE_UPDATE_CHECK": "1", "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION": "1", "FLWR_HOME": str(job / ".flower"),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0", "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(128 * 1024 * 1024),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    overrides = job / "flower-run.toml"
    overrides.write_text("job-dir = " + json.dumps(str(job)) + "\n", encoding="utf-8")
    federation = (f"num-supernodes={config.clients} client-resources-num-cpus={config.client_cpus} "
                  f"client-resources-num-gpus=0.0 init-args-num-cpus={config.ray_cpus} init-args-num-gpus=0")
    command = [str(scripts / ("flwr.exe" if os.name == "nt" else "flwr")), "run", ".",
               "--run-config", str(overrides), "--federation-config", federation, "--stream"]
    write_json(job / "launcher.json", {"command": command, "flwr_home": env["FLWR_HOME"],
        "ray_object_store_max_bytes": 128 * 1024 * 1024})
    try:
        with (job / "flower.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=config.round_timeout * (2 * config.rounds + 1) + 180)
        if not (job / "federated.json").exists():
            raise RuntimeError("Flower exited without completed federated metrics")
        centralized(config, job)
        plots(job)
        write_json(job / "status.json", {"status": "complete"})
    except BaseException as error:
        write_json(job / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description="TraCast Flower simulation and matched centralized GRU")
    parser.add_argument("--config", default="configs/federated-smoke.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = FederatedConfig(**json.loads(Path(args.config).read_text()))
    job = prepare(config, args.data, args.out)
    if not args.prepare_only:
        execute(job)
    print(job, flush=True)


if __name__ == "__main__":
    main()
