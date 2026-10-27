"""Official Flower FedAvg with ordered aggregation and auditable round records."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg

from .experiment import seed_all, write_json
from .federated_task import (
    FederatedConfig,
    new_model,
    parameter_bytes,
    pool,
    state_hash,
)


def train_metrics(records, weighted_by_key):
    values = [dict(record["metrics"]) for record in records]
    loss = sum(value["loss-sum"] for value in values)
    count = sum(value["target-count"] for value in values)
    if count <= 0:
        raise ValueError("No local training targets")
    return MetricRecord(
        {
            "train-normalized-mae": loss / count,
            "target-count": count,
            "num-examples": sum(value["num-examples"] for value in values),
            "optimizer-steps": sum(value["optimizer-steps"] for value in values),
            "processed-examples": sum(value["processed-examples"] for value in values),
        }
    )


def evaluation_metrics(records, weighted_by_key):
    pooled = pool([dict(record["metrics"]) for record in records])
    return MetricRecord({"mae": pooled["mae"], "rmse": pooled["rmse"]})


def checked_replies(replies, clients):
    replies = list(replies)
    if len(replies) != clients or any(reply.has_error() for reply in replies):
        raise RuntimeError("All configured clients must reply successfully")
    ids = [int(reply.content["metrics"]["client-id"]) for reply in replies]
    if sorted(ids) != list(range(clients)):
        raise RuntimeError("Duplicate or missing client partitions")
    return sorted(replies, key=lambda reply: int(reply.content["metrics"]["client-id"]))


class LoggedFedAvg(FedAvg):
    def __init__(self, config, job):
        super().__init__(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=config.clients,
            min_evaluate_nodes=config.clients,
            min_available_nodes=config.clients,
            weighted_by_key="num-examples",
            train_metrics_aggr_fn=train_metrics,
            evaluate_metrics_aggr_fn=evaluation_metrics,
        )
        self.config, self.job = config, Path(job)
        self.rounds = []
        self.best_mae, self.best_round = float("inf"), None
        self.bytes = parameter_bytes(new_model(config))

    def aggregate_train(self, server_round, replies):
        replies = checked_replies(replies, self.config.clients)
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is None or metrics is None:
            raise RuntimeError("FedAvg did not produce a model")
        model = new_model(self.config)
        model.load_state_dict(arrays.to_torch_state_dict())
        torch.save(model.state_dict(), self.job / f"round-{server_round:03d}.pt")
        row = {
            "round": server_round,
            "train": dict(metrics),
            "state_sha256": state_hash(model),
            "clients": [
                {
                    "client_id": int(reply.content["metrics"]["client-id"]),
                    "train": dict(reply.content["metrics"]),
                }
                for reply in replies
            ],
            "communication": {
                "train_download_bytes": self.config.clients * self.bytes,
                "train_upload_bytes": self.config.clients * self.bytes,
                "validation_download_bytes": self.config.clients * self.bytes,
                "round_parameter_bytes": 3 * self.config.clients * self.bytes,
                "cumulative_parameter_bytes": server_round
                * 3
                * self.config.clients
                * self.bytes,
            },
        }
        self.rounds.append(row)
        write_json(self.job / "rounds.json", self.rounds)
        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        replies = checked_replies(replies, self.config.clients)
        result = super().aggregate_evaluate(server_round, replies)
        statistics = [dict(reply.content["metrics"]) for reply in replies]
        pooled = pool(statistics)
        row = self.rounds[-1]
        if row["round"] != server_round:
            raise RuntimeError("Evaluation round mismatch")
        row["validation"] = pooled
        for client, stats in zip(row["clients"], statistics):
            client["validation_statistics"] = stats
            client["validation"] = pool([stats])
        if pooled["mae"] < self.best_mae:
            self.best_mae, self.best_round = pooled["mae"], server_round
        write_json(self.job / "rounds.json", self.rounds)
        print(
            f"round {server_round}: validation MAE={pooled['mae']:.6f}, RMSE={pooled['rmse']:.6f} mph",
            flush=True,
        )
        return result


def build_server_app(job_dir=None):
    server = ServerApp()

    @server.main()
    def main(grid, context):
        job = Path(job_dir or context.run_config["job-dir"]).resolve()
        config = FederatedConfig(**json.loads((job / "config.json").read_text()))
        config.validate()
        seed_all(config.seed, config.threads)
        model = new_model(config)
        initial = torch.load(job / "initial.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(initial)
        strategy = LoggedFedAvg(config, job)
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(model.state_dict()),
            num_rounds=config.rounds,
            timeout=config.round_timeout,
            train_config=ConfigRecord({"job-dir": str(job)}),
            evaluate_config=ConfigRecord({"job-dir": str(job), "stage": "val"}),
        )
        if strategy.best_round is None or len(strategy.rounds) != config.rounds:
            raise RuntimeError("No complete federated run")
        selected = torch.load(
            job / f"round-{strategy.best_round:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(selected)
        torch.save(selected, job / "best-fedavg.pt")
        messages = [
            Message(
                content=RecordDict(
                    {
                        "arrays": ArrayRecord(model.state_dict()),
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
            grid.send_and_receive(messages, timeout=config.round_timeout),
            config.clients,
        )
        statistics = [dict(reply.content["metrics"]) for reply in replies]
        result = {
            "best_round": strategy.best_round,
            "test": pool(statistics),
            "test_clients": [
                {
                    "client_id": stats["client-id"],
                    "statistics": stats,
                    "metrics": pool([stats]),
                }
                for stats in statistics
            ],
            "state_sha256": state_hash(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_parameter_bytes": strategy.bytes,
            "communication": {
                "training_and_validation_bytes": config.rounds
                * 3
                * config.clients
                * strategy.bytes,
                "selected_test_download_bytes": config.clients * strategy.bytes,
                "total_parameter_bytes": (config.rounds * 3 + 1)
                * config.clients
                * strategy.bytes,
                "scope": "Uncompressed model tensors only; metric/config/serialization/transport overhead excluded",
            },
        }
        write_json(job / "federated.json", result)

    return server


app = build_server_app()
