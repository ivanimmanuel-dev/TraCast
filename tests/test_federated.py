"""Constructed arrays check federated mechanics; real-data runs are separate."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from tracast.federated_task import (
    FederatedConfig,
    bundle,
    new_model,
    partitions,
    pool,
    parameter_bytes,
    train_epochs,
    state_hash,
)
from tracast.flower_server import LoggedFedAvg, checked_replies


@pytest.fixture(autouse=True)
def flower_task_identity(monkeypatch):
    # Normally initialized by Flower's runtime; unit messages have no runtime.
    from flwr.supercore.task_identity import TaskIdentity

    for name, value in (("_run_id", 1), ("_node_id", 0), ("_task_id", 1)):
        monkeypatch.setattr(TaskIdentity, name, value)


@pytest.fixture
def config():
    return FederatedConfig(rows=300, hidden=4, rounds=1, threads=1)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {str(i): np.arange(300, dtype=np.float32) / (i + 1) + 10 for i in range(6)},
        index=pd.date_range("2012-03-01", periods=300, freq="5min"),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"clients": 1},
        {"rows": 0},
        {"local_epochs": 0},
        {"sensors_per_client": 100},
        {"lr": float("nan")},
        {"ray_cpus": 1},
        {"rounds": True},
    ],
)
def test_federated_config_rejects_invalid(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_disjoint_sensors_and_consistent_times(frame, config):
    groups = partitions(frame, config)
    assert [list(group.columns) for group in groups] == [
        ["0", "1"],
        ["2", "3"],
        ["4", "5"],
    ]
    metadata = [bundle(group, config)[1] for group in groups]
    for item in metadata[1:]:
        for stage in ("train", "val", "test"):
            assert (
                item["splits"][stage]["first"] == metadata[0]["splits"][stage]["first"]
            )
            assert item["splits"][stage]["last"] == metadata[0]["splits"][stage]["last"]
    with pytest.raises(ValueError):
        partitions(frame.iloc[:, :5], config)


def test_local_scalers_ignore_heldout_and_other_clients(frame, config):
    group = partitions(frame, config)[0]
    data, before = bundle(group, config)
    altered = frame.copy()
    altered.iloc[210:, :2] = 10000
    altered.iloc[:, 2:] = 99999
    other_data, after = bundle(partitions(altered, config)[0], config)
    assert before["normalization"] == after["normalization"]
    assert before["normalization"]["mean"][0] == pytest.approx(
        group.iloc[:210, 0].mean()
    )
    for i in range(len(data["train"])):
        assert all(
            torch.equal(a, b) for a, b in zip(data["train"][i], other_data["train"][i])
        )


def test_shared_model_shape_and_window_targets(frame, config):
    data, metadata = bundle(frame.iloc[:, :2], config)
    assert len(data["train"]) == 2 * (210 - 23)
    x, y, mask, mean, std = data["train"][0]
    assert x.shape == (12, 1) and y.shape == (12, 1)
    np.testing.assert_array_equal(y.numpy()[:, 0], frame.iloc[12:24, 0])
    assert new_model(config)(x.unsqueeze(0)).shape == (1, 12, 1)
    assert data["train"][187][1][0] == frame.iloc[12, 1]
    assert len(data["val"]) == 14


def test_empty_training_targets_excluded(frame, config):
    frame.iloc[12:40, 0] = 0
    data, metadata = bundle(frame.iloc[:, :2], config)
    assert len(data["train"]) < metadata["splits"]["train"]["candidate_examples"]
    assert all(item[2].any() for item in data["train"])


def test_pool_errors_before_rmse():
    records = [
        {"ae": [1.0, 6.0], "se": [1.0, 18.0], "count": [1, 2]},
        {"ae": [6.0, 4.0], "se": [12.0, 16.0], "count": [3, 1]},
        {"ae": [6.0, 10.0], "se": [18.0, 50.0], "count": [2, 2]},
    ]
    result = pool(records)
    assert result["mae"] == pytest.approx(33 / 11)
    assert result["rmse"] == pytest.approx(np.sqrt(115 / 11))
    assert [row["count"] for row in result["per_horizon"]] == [6, 5]
    with pytest.raises(ValueError):
        pool([])


def reply(client_id, arrays, metrics, kind="train"):
    request = Message(
        content=RecordDict(), dst_node_id=client_id + 10, message_type=kind
    )
    content = {"metrics": MetricRecord({"client-id": client_id, **metrics})}
    if arrays is not None:
        content["arrays"] = ArrayRecord(arrays)
    return Message(content=RecordDict(content), reply_to=request)


def test_official_fedavg_weighting_order_and_bytes(config, tmp_path):
    strategy = LoggedFedAvg(config, tmp_path)
    replies = []
    for i, value in enumerate((1.0, 3.0, 5.0)):
        model = new_model(config)
        weights = {
            name: torch.full_like(tensor, value)
            for name, tensor in model.state_dict().items()
        }
        metrics = {
            "num-examples": i + 1,
            "loss-sum": 2.0,
            "target-count": 1,
            "optimizer-steps": 1,
            "processed-examples": i + 1,
            "train-normalized-mae": 2.0,
        }
        replies.append(reply(i, weights, metrics))
    arrays, metrics = strategy.aggregate_train(1, list(reversed(replies)))
    for tensor in arrays.to_torch_state_dict().values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 22 / 6))
    assert metrics["num-examples"] == 6
    assert strategy.rounds[0]["communication"][
        "round_parameter_bytes"
    ] == 9 * parameter_bytes(new_model(config))
    stats = {"num-examples": 2, "ae": [1.0] * 12, "se": [2.0] * 12, "count": [1] * 12}
    strategy.aggregate_evaluate(
        1, [reply(i, None, stats, "evaluate") for i in range(3)]
    )
    assert strategy.best_round == 1
    assert strategy.rounds[0]["validation"]["rmse"] == pytest.approx(np.sqrt(2))
    assert [row["client_id"] for row in strategy.rounds[0]["clients"]] == [0, 1, 2]


def test_failed_or_duplicate_clients_stop():
    def ok(i):
        return SimpleNamespace(
            has_error=lambda: False, content={"metrics": {"client-id": i}}
        )

    with pytest.raises(RuntimeError):
        checked_replies([ok(0), ok(1)], 3)
    with pytest.raises(RuntimeError):
        checked_replies([ok(0), ok(0), ok(2)], 3)


def test_training_repeats(frame, config):
    from tracast.experiment import seed_all

    data, _ = bundle(frame.iloc[:, :2], config)
    seed_all(42, 1)
    first = new_model(config)
    initial = {k: v.clone() for k, v in first.state_dict().items()}
    a = train_epochs(first, data["train"], config, 123)
    second = new_model(config)
    second.load_state_dict(initial)
    b = train_epochs(second, data["train"], config, 123)
    assert a == b and state_hash(first) == state_hash(second)
    assert a["processed-examples"] == len(data["train"])


def test_clientapp_message_dispatch(frame, config, tmp_path, monkeypatch):
    import tracast.federated as runner
    from tracast.flower_client import app

    monkeypatch.setattr(runner, "load", lambda *args: frame)
    job = runner.prepare(config, "constructed-test-fixture", tmp_path / "job")
    initial = torch.load(job / "initial.pt", weights_only=True)
    context = Context(
        run_id=1,
        node_id=10,
        node_config={"partition-id": 0, "num-partitions": 3},
        state=RecordDict(),
        run_config={},
    )
    request = Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(initial),
                "config": ConfigRecord({"job-dir": str(job), "server-round": 1}),
            }
        ),
        dst_node_id=10,
        message_type="train",
    )
    trained = app(request, context)
    assert not trained.has_error()
    assert trained.content["metrics"]["num-examples"] == 374
    evaluation = Message(
        content=RecordDict(
            {
                "arrays": trained.content["arrays"],
                "config": ConfigRecord(
                    {"job-dir": str(job), "stage": "val", "server-round": 1}
                ),
            }
        ),
        dst_node_id=10,
        message_type="evaluate",
    )
    evaluated = app(evaluation, context)
    assert pool([dict(evaluated.content["metrics"])])["mae"] >= 0
    assert evaluated.content["metrics"]["num-examples"] == 14


@pytest.mark.parametrize("failure", ["nonzero_exit", "missing_metrics"])
def test_launcher_checks_runtime_and_failure_status(
    frame, config, tmp_path, monkeypatch, failure
):
    import subprocess
    import tomllib
    import tracast.federated as runner

    monkeypatch.setattr(runner, "load", lambda *args: frame)
    job = runner.prepare(config, "constructed-test-fixture", tmp_path / "job")

    def launch(command, **kwargs):
        assert "run" in command and "--stream" in command
        values = command[command.index("--federation-config") + 1]
        parsed = tomllib.loads(values.replace(" ", "\n"))
        assert type(parsed["init-args-num-gpus"]) is int
        assert parsed["num-supernodes"] == 3
        assert kwargs["env"]["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] == "1"
        assert kwargs["env"]["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == str(
            128 * 1024 * 1024
        )
        if failure == "nonzero_exit":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", launch)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        runner.execute(job)
    assert json.loads((job / "status.json").read_text())["status"] == "failed"
    assert not (job / "centralized.json").exists()


@pytest.mark.parametrize("rule", ["balanced", "congestion"])
def test_partition_ranking_uses_training_only(frame, config, rule):
    from tracast.federated_task import partitions, congestion_scores

    config = replace(config, partition_rule=rule)
    groups = partitions(frame, config)
    changed = frame.copy()
    changed.iloc[210:, :] = 99999
    assert [list(group.columns) for group in groups] == [
        list(group.columns) for group in partitions(changed, config)
    ]
    assert congestion_scores(frame, config) == congestion_scores(changed, config)
    assert sorted(sensor for group in groups for sensor in group.columns) == sorted(
        frame.columns
    )
    assert all(len(group.columns) == 2 for group in groups)


def test_pooled_reference_independent_of_sensor_ownership(
    frame, config, tmp_path, monkeypatch
):
    import tracast.federated as runner

    monkeypatch.setattr(runner, "load", lambda *args: frame)
    results = []
    for rule in ("balanced", "congestion"):
        settings = replace(config, partition_rule=rule)
        job = runner.prepare(settings, "constructed-test-fixture", tmp_path / rule)
        results.append(runner.centralized(settings, job))
    assert results[0] == results[1]


def test_histogram_divergence():
    from tracast.federated_study import js_divergence

    assert js_divergence([1, 0], [1, 0]) == 0
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1)
    assert js_divergence([1, 2], [2, 1]) == pytest.approx(js_divergence([2, 1], [1, 2]))
    with pytest.raises(ValueError):
        js_divergence([0, 0], [1, 0])


@pytest.mark.parametrize(
    "first,second",
    [
        ([1, 2], [1]),
        ([[1, 2]], [[1, 2]]),
        ([float("nan")], [1]),
        ([float("inf")], [1]),
        ([-1, 2], [1, 2]),
    ],
)
def test_histogram_rejects_invalid_counts(first, second):
    from tracast.federated_study import js_divergence

    with pytest.raises(ValueError):
        js_divergence(first, second)


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": [7, 7]},
        {"seeds": [True, 42]},
        {"seeds": [-1, 42]},
        {"balanced_round_budgets": []},
        {"balanced_round_budgets": [2, 2]},
        {"balanced_round_budgets": [False, 2]},
        {"congestion_round_budgets": [5]},
    ],
)
def test_study_rejects_settings_before_creating_runs(config, tmp_path, changes):
    from dataclasses import asdict
    from tracast.federated_study import run

    settings = {
        "base": asdict(config),
        "seeds": [7, 42],
        "balanced_round_budgets": [1, 2],
        "congestion_round_budgets": [2],
    } | changes
    out = tmp_path / "study"
    with pytest.raises(ValueError):
        run(settings, "unused", out)
    assert not out.exists()


def test_study_uses_configured_budgets_and_checks_controls(config, tmp_path):
    from copy import deepcopy
    from dataclasses import asdict
    from tracast.federated_study import aggregate, plot, validate_settings

    settings = {
        "base": asdict(config),
        "seeds": [7, 42],
        "balanced_round_budgets": [1, 2],
        "congestion_round_budgets": [1, 2],
    }
    validate_settings(settings)
    cases = []
    for seed in settings["seeds"]:
        for rule in ("balanced", "congestion"):
            for budget in (1, 2):
                cases.append(
                    {
                        "case_id": f"{rule[0]}{budget}-s{seed}",
                        "seed": seed,
                        "partition_rule": rule,
                        "round_budget": budget,
                        "config": asdict(config),
                        "fedavg": {
                            "test": {"mae": 4.0, "rmse": 8.0},
                            "communication": {"total_parameter_bytes": 100 * budget},
                        },
                        "centralized": {
                            "test": {"mae": 3.0, "rmse": 7.0},
                            "persistence_test": {"mae": 5.0},
                            "state_sha256": str(seed),
                        },
                        "heterogeneity": {"mean_js_bits": 0.1},
                        "wall_seconds": 1.0,
                        "rounds": [
                            {
                                "state_sha256": f"{seed}-{i}",
                                "validation": {"mae": 4.0 / i},
                            }
                            for i in range(1, budget + 1)
                        ],
                    }
                )
    summary = aggregate(cases)
    assert len(summary["centralized_reference_identical"]) == 4
    assert all(summary["round_prefix_identical"].values())
    assert "At 2 rounds" in summary["brief_findings"]
    plot(tmp_path, cases, summary)
    assert (tmp_path / "tradeoffs.png").stat().st_size > 0

    changed = deepcopy(cases)
    changed[-1]["centralized"]["state_sha256"] = "different-model"
    with pytest.raises(ValueError, match="reproducibility"):
        aggregate(changed)
    changed = deepcopy(cases)
    changed[0]["rounds"][0]["state_sha256"] = "different-prefix"
    with pytest.raises(ValueError, match="reproducibility"):
        aggregate(changed)
