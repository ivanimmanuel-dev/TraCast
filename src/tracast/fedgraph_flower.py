"""Flower apps for the federated graph models (single-machine simulation).

Clients read only their own prepared partition (their sensors' readings, their local
graph) and exchange only the learned parameters. The server uses Flower's FedAvg or
FedProx strategy, aggregates in client-id order, and writes a result file with the
same layout as the in-process run in fedgraph.py, so the two can be compared directly.

Run a prepared job with `python -m tracast.fedgraph_flower --job DIR`.
"""

import argparse
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from . import fedgraph as F
from .data import digest
from .experiment import write_json
from .resources import peak_memory

DAILY = ("daily-abs", "daily-sq", "daily-count")


def read_config(job):
    return F.GraphFedConfig(**json.loads((Path(job) / "config.json").read_text()))


def write_job(prep, config, job, start):
    """Write what the clients and the server need; one file per client partition."""
    config.validate()
    if config.model in F.POOLED_ONLY:
        raise ValueError(f"{config.model} can't be federated")
    job = Path(job)
    (job / "clients").mkdir(parents=True, exist_ok=False)
    clients = F.build_clients(prep, config)
    write_json(job / "config.json", asdict(config))
    for client, group in zip(clients, prep["groups"]):
        frame = prep["frame"].iloc[:, group]
        np.savez_compressed(
            job / "clients" / f"client-{client.client_id}.npz",
            values=frame.to_numpy(),
            timestamps=frame.index.to_numpy(),
            sensors=np.asarray(frame.columns, dtype=str),
            weights=client.weights,
        )
    torch.save(start, job / "initial.pt")
    labels = clients[0].datasets["test"].origin_days()[1]
    write_json(job / "test-days.json", labels)
    write_json(
        job / "setup.json",
        {
            **F.describe(prep, clients, config),
            "partition_files": {
                p.name: digest(p) for p in sorted((job / "clients").glob("*.npz"))
            },
        },
    )
    write_json(job / "status.json", {"status": "prepared"})
    return clients


def load_client(job, client_id, config):
    with np.load(Path(job) / "clients" / f"client-{client_id}.npz") as saved:
        frame = F.frame_from_arrays(
            saved["values"], saved["timestamps"], saved["sensors"].tolist()
        )
        weights = saved["weights"]
    return F.ClientData(frame, weights, config, client_id)


# --- client ---------------------------------------------------------------------------

app = ClientApp()


def _setup(msg, context):
    settings = msg.content["config"]
    job = Path(str(settings["job-dir"]))
    config = read_config(job)
    client_id = int(context.node_config["partition-id"])
    if not 0 <= client_id < config.clients:
        raise ValueError("Client ID outside the prepared partitions")
    data = load_client(job, client_id, config)
    model = F.build_model(config, data.weights)
    params = OrderedDict(msg.content["arrays"].to_torch_state_dict())
    return client_id, config, data, model, params, settings


@app.train()
def train(msg: Message, context: Context):
    client_id, config, data, model, params, settings = _setup(msg, context)
    round_id = int(settings["server-round"])
    mu = float(settings["proximal-mu"]) if "proximal-mu" in settings else 0.0
    seed = config.seed + 10000 * round_id + client_id
    updated, stats = F.local_update(model, params, data, config, seed, mu)
    stats["client-id"] = client_id
    stats["sensor-count"] = len(data.sensors)
    stats["peak-memory-bytes"] = peak_memory()["bytes"]
    return Message(
        content=RecordDict(
            {"arrays": ArrayRecord(updated), "metrics": MetricRecord(stats)}
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    client_id, config, data, model, params, settings = _setup(msg, context)
    stage = str(settings["stage"])
    if stage not in ("val", "test"):
        raise ValueError("Only held-out evaluation stages are supported")
    F.load_parameters(model, params)
    record = F.evaluate(model, data, stage, config, with_days=stage == "test")
    metrics = {k: record[k] for k in ("ae", "se", "count")}
    if stage == "test":
        daily = record["daily"]
        metrics.update(
            {
                "daily-abs": daily["abs"],
                "daily-sq": daily["sq"],
                "daily-count": daily["count"],
            }
        )
    metrics["client-id"] = client_id
    metrics["sensor-count"] = len(data.sensors)
    metrics["num-examples"] = len(data.datasets[stage]) * len(data.sensors)
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)


# --- server ---------------------------------------------------------------------------


def checked_replies(replies, clients):
    replies = list(replies)
    if len(replies) != clients or any(reply.has_error() for reply in replies):
        raise RuntimeError("All configured clients must reply successfully")
    ids = [int(r.content["metrics"]["client-id"]) for r in replies]
    if sorted(ids) != list(range(clients)):
        raise RuntimeError("Duplicate or missing client partitions")
    return sorted(replies, key=lambda r: int(r.content["metrics"]["client-id"]))


def train_metrics(records, weighted_by_key):
    values = [dict(r["metrics"]) for r in records]
    count = sum(v["target-count"] for v in values)
    if count <= 0:
        raise ValueError("No local training targets")
    return MetricRecord(
        {
            "train-normalized-mae": sum(v["loss-sum"] for v in values) / count,
            "optimizer-steps": sum(v["optimizer-steps"] for v in values),
        }
    )


def evaluation_metrics(records, weighted_by_key):
    pooled = F.pool([dict(r["metrics"]) for r in records])
    return MetricRecord({"mae": pooled["mae"], "rmse": pooled["rmse"]})


def stats_of(reply):
    metrics = dict(reply.content["metrics"])
    record = {k: list(metrics[k]) for k in ("ae", "se", "count")}
    if "daily-abs" in metrics:
        record["daily-abs"] = list(metrics["daily-abs"])
        record["daily-sq"] = list(metrics["daily-sq"])
        record["daily-count"] = [int(v) for v in metrics["daily-count"]]
    return record


class Logged:
    """Round records in the same layout as fedgraph.run_federated."""

    def setup_logging(self, config, job, start):
        self.config, self.job = config, Path(job)
        self.rows, self.params = [], {}
        self.best, self.best_round = float("inf"), None
        self.size = F.parameter_bytes(start)
        self.train_rows = {}

    def aggregate_train(self, server_round, replies):
        replies = checked_replies(replies, self.config.clients)
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is None or metrics is None:
            raise RuntimeError("The strategy did not produce a model")
        params = OrderedDict(arrays.to_torch_state_dict())
        self.params[server_round] = OrderedDict(
            (k, v.clone()) for k, v in params.items()
        )
        n = self.config.clients
        self.rows.append(
            {
                "round": server_round,
                "train_normalized_mae": float(metrics["train-normalized-mae"]),
                "client_train_seconds": [
                    float(r.content["metrics"]["train-seconds"]) for r in replies
                ],
                "client_peak_memory_bytes": [
                    int(r.content["metrics"]["peak-memory-bytes"]) for r in replies
                ],
                "optimizer_steps": int(metrics["optimizer-steps"]),
                "state_sha256": F.state_hash(params),
                "communication_bytes": {
                    "train_download": n * self.size,
                    "train_upload": n * self.size,
                    "validation_download": n * self.size,
                    "round_total": 3 * n * self.size,
                },
            }
        )
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        replies = checked_replies(replies, self.config.clients)
        result = super().aggregate_evaluate(server_round, replies)
        records = [stats_of(r) for r in replies]
        pooled = F.pool(records)
        row = self.rows[-1]
        if row["round"] != server_round:
            raise RuntimeError("Evaluation round mismatch")
        row["validation"] = {"mae": pooled["mae"], "rmse": pooled["rmse"]}
        row["client_validation_mae"] = [F.pool([r])["mae"] for r in records]
        if pooled["mae"] < self.best:
            self.best, self.best_round = pooled["mae"], server_round
        print(
            f"federated round {server_round}: val MAE {pooled['mae']:.4f} "
            f"train {row['train_normalized_mae']:.4f}",
            flush=True,
        )
        return result


class LoggedFedAvg(Logged, FedAvg):
    pass


class LoggedFedProx(Logged, FedProx):
    pass


def build_server_app(job_dir):
    server = ServerApp()

    @server.main()
    def main(grid, context):
        began = time.perf_counter()
        job = Path(job_dir).resolve()
        config = read_config(job)
        config.validate()
        start = torch.load(job / "initial.pt", map_location="cpu", weights_only=True)
        options = dict(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=config.clients,
            min_evaluate_nodes=config.clients,
            min_available_nodes=config.clients,
            weighted_by_key="num-examples",
            train_metrics_aggr_fn=train_metrics,
            evaluate_metrics_aggr_fn=evaluation_metrics,
        )
        if config.algorithm == "fedprox":
            strategy = LoggedFedProx(proximal_mu=config.proximal_mu, **options)
        else:
            strategy = LoggedFedAvg(**options)
        strategy.setup_logging(config, job, start)
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(start),
            num_rounds=config.rounds,
            timeout=600,
            train_config=ConfigRecord({"job-dir": str(job)}),
            evaluate_config=ConfigRecord({"job-dir": str(job), "stage": "val"}),
        )
        if strategy.best_round is None or len(strategy.rows) != config.rounds:
            raise RuntimeError("No complete federated run")
        best = strategy.params[strategy.best_round]
        messages = [
            Message(
                content=RecordDict(
                    {
                        "arrays": ArrayRecord(best),
                        "config": ConfigRecord(
                            {
                                "job-dir": str(job),
                                "stage": "test",
                                "server-round": strategy.best_round,
                            }
                        ),
                    }
                ),
                dst_node_id=node,
                message_type="evaluate",
            )
            for node in sorted(grid.get_node_ids())
        ]
        replies = checked_replies(
            grid.send_and_receive(messages, timeout=600), config.clients
        )
        labels = json.loads((job / "test-days.json").read_text())
        records = []
        for reply in replies:
            raw = stats_of(reply)
            records.append(
                {
                    "ae": raw["ae"],
                    "se": raw["se"],
                    "count": raw["count"],
                    "daily": {
                        "days": labels,
                        "abs": raw["daily-abs"],
                        "sq": raw["daily-sq"],
                        "count": raw["daily-count"],
                    },
                }
            )
        result = {
            "best_round": strategy.best_round,
            "test": F.pool(records),
            "daily": F.pool_days(records),
            "clients": [
                {
                    "client": int(r.content["metrics"]["client-id"]),
                    "sensors": int(r.content["metrics"]["sensor-count"]),
                    "metrics": F.pool([F.slim(rec)]),
                }
                for r, rec in zip(replies, records)
            ],
            "history": strategy.rows,
            "parameter_count": F.parameter_count(best),
            "parameter_bytes": strategy.size,
            "state_sha256": F.state_hash(best),
            "communication": {
                "training_and_validation_bytes": config.rounds
                * 3
                * config.clients
                * strategy.size,
                "selected_test_download_bytes": config.clients * strategy.size,
                "total_parameter_bytes": (config.rounds * 3 + 1)
                * config.clients
                * strategy.size,
                "scope": "Uncompressed parameter tensors only; metrics, configs and transport overhead excluded",
            },
            "train_seconds": sum(sum(r["client_train_seconds"]) for r in strategy.rows),
            "peak_client_process_bytes": max(
                max(r["client_peak_memory_bytes"]) for r in strategy.rows
            ),
            "wall_seconds": time.perf_counter() - began,
        }
        write_json(job / "flower-result.json", result)

    return server


# --- launcher -------------------------------------------------------------------------


def run_simulation(job):
    from flwr.simulation import run_simulation as simulate

    job = Path(job).resolve()
    config = read_config(job)
    cpus = max(1, config.threads)
    simulate(
        server_app=build_server_app(job),
        client_app=app,
        num_supernodes=config.clients,
        backend_config={
            "client_resources": {"num_cpus": cpus, "num_gpus": 0.0},
            "init_args": {"num_cpus": cpus, "num_gpus": 0, "include_dashboard": False},
        },
    )
    if not (job / "flower-result.json").exists():
        raise RuntimeError("Flower exited without writing a result")
    return json.loads((job / "flower-result.json").read_text())


def execute(job, timeout=3 * 3600):
    """Run the simulation in a fresh process (Ray and torch settings stay isolated)."""
    import subprocess

    job = Path(job).resolve()
    env = os.environ.copy()
    env.update(
        {
            "FLWR_TELEMETRY_ENABLED": "0",
            "RAY_USAGE_STATS_ENABLED": "0",
            "FLWR_DISABLE_UPDATE_CHECK": "1",
            "FLWR_HOME": str(job / ".flower"),
            "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": str(256 * 1024 * 1024),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    write_json(job / "status.json", {"status": "running"})
    try:
        with (job / "flower.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-m", "tracast.fedgraph_flower", "--job", str(job)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        write_json(job / "status.json", {"status": "complete"})
    except BaseException as error:
        write_json(
            job / "status.json",
            {"status": "failed", "type": type(error).__name__, "error": str(error)},
        )
        raise
    return json.loads((job / "flower-result.json").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    run_simulation(args.job)


if __name__ == "__main__":
    main()
