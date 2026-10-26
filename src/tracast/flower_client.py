"""Flower Message API client: only its own prepared sensor partition is read."""
import json
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .federated_task import FederatedConfig, load_partition, new_model, score, train_epochs
from .experiment import seed_all

app = ClientApp()


def setup(msg, context):
    settings = msg.content["config"]
    client_id = int(context.node_config["partition-id"])
    job = Path(str(settings["job-dir"]))
    config = FederatedConfig(**json.loads((job / "config.json").read_text()))
    if not 0 <= client_id < config.clients:
        raise ValueError("Client ID outside the prepared partitions")
    seed_all(config.seed, config.threads)
    model = new_model(config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    datasets, _ = load_partition(job, client_id)
    return client_id, config, model, datasets, settings


@app.train()
def train(msg: Message, context: Context):
    client_id, config, model, datasets, settings = setup(msg, context)
    round_id = int(settings["server-round"])
    metrics = train_epochs(model, datasets["train"], config, config.seed + 10000 * round_id + client_id)
    metrics.update({"num-examples": len(datasets["train"]), "client-id": client_id})
    return Message(content=RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    client_id, config, model, datasets, settings = setup(msg, context)
    stage = str(settings["stage"])
    if stage not in ("val", "test"):
        raise ValueError("Only held-out evaluation stages are supported")
    metrics = score(model, datasets[stage], config)
    metrics["client-id"] = client_id
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
