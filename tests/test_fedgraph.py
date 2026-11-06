"""Federated graph models: partitions, local graphs, aggregation and leakage.

Everything runs on small constructed arrays; the real-data runs are separate.
"""

from collections import OrderedDict

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

from tracast import fedgraph as F
from tracast import fedgraph_flower as FF


@pytest.fixture(autouse=True)
def flower_task_identity(monkeypatch):
    # Normally initialized by Flower's runtime; unit messages have no runtime.
    from flwr.supercore.task_identity import TaskIdentity

    for name, value in (("_run_id", 1), ("_node_id", 0), ("_task_id", 1)):
        monkeypatch.setattr(TaskIdentity, name, value)


GROUPS = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7])]


def synthetic(rows=400, sensors=8, seed=0):
    rng = np.random.default_rng(seed)
    phase = np.arange(sensors)[None, :] * 0.4
    values = 50 + 10 * np.sin(np.arange(rows)[:, None] / 18 + phase)
    values = (values + rng.normal(0, 1, (rows, sensors))).astype(np.float32)
    frame = pd.DataFrame(
        values,
        index=pd.date_range("2012-03-01", periods=rows, freq="5min"),
        columns=[str(i) for i in range(sensors)],
    )
    weights = np.zeros((sensors, sensors), dtype=np.float32)
    for a in range(sensors):
        for b in range(sensors):
            if a != b and abs(a - b) <= 2:
                weights[a, b] = 1.0 / (1 + abs(a - b))
    return frame, weights


def make_prep(frame, weights, groups=GROUPS):
    return {
        "frame": frame,
        "ids": list(frame.columns),
        "weights": weights,
        "groups": list(groups),
        "graph_meta": {},
    }


@pytest.fixture
def config():
    return F.GraphFedConfig(
        sensors=16,
        rows=400,
        hidden=4,
        clients=3,
        rounds=1,
        local_stride=2,
        batch_size=16,
        threads=1,
    )


@pytest.fixture
def setup(config):
    frame, weights = synthetic()
    prep = make_prep(frame, weights)
    return prep, F.build_clients(prep, config), F.initial_parameters(config)


# --- configuration --------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"clients": 1},
        {"sensors": 8},
        {"rounds": 0},
        {"local_stride": 0},
        {"lr": float("nan")},
        {"model": "gcn"},
        {"partition": "alphabetical"},
        {"algorithm": "fedavg", "proximal_mu": 0.1},
        {"algorithm": "fedprox", "proximal_mu": -1.0},
        {"rows": 20},
        {"seed": -1},
        {"rounds": True},
    ],
)
def test_config_rejects_invalid(config, changes):
    with pytest.raises(ValueError):
        F.with_config(config, **changes)


# --- partitions -----------------------------------------------------------------------


def test_kmeans_recovers_separated_blobs_and_is_deterministic():
    rng = np.random.default_rng(1)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    points = np.concatenate([c + rng.normal(0, 0.3, (12, 2)) for c in centers])
    labels = F.kmeans(points, 3, seed=5)
    assert len(set(labels[:12])) == len(set(labels[12:24])) == 1
    assert len({labels[0], labels[12], labels[24]}) == 3
    assert np.array_equal(labels, F.kmeans(points, 3, seed=5))
    # cluster numbers follow the centroids: west to east, ties south to north
    assert (labels[0], labels[24], labels[12]) == (0, 1, 2)


def test_kmeans_rejects_impossible_requests():
    with pytest.raises(ValueError):
        F.kmeans(np.zeros((3, 2)), 5)


@pytest.mark.parametrize("rule", ["geographic", "balanced", "congestion", "random"])
def test_partitions_are_disjoint_and_cover_every_sensor(config, rule):
    frame, _ = synthetic(sensors=9)
    locations = {
        str(i): {"latitude": 34 + (i // 3) * 0.2, "longitude": -118 + (i % 3) * 0.05}
        for i in range(9)
    }
    groups = F.partition_groups(frame, locations, F.with_config(config, partition=rule))
    flat = np.concatenate(groups)
    assert sorted(flat.tolist()) == list(range(9))
    assert all(len(g) for g in groups)


def test_geographic_partition_gives_clients_areas_of_different_size(config):
    frame, _ = synthetic(sensors=10)
    # sensors 0-5 are one tight area and 6-9 another far away
    locations = {
        str(i): {
            "latitude": 34.0 + 0.001 * i if i < 6 else 34.5 + 0.001 * i,
            "longitude": -118.0,
        }
        for i in range(10)
    }
    cfg = F.with_config(config, clients=2)
    groups = F.partition_groups(frame, locations, cfg)
    assert [len(g) for g in groups] == [6, 4]


def test_congestion_ranking_uses_training_rows_only(config):
    frame, _ = synthetic()
    changed = frame.copy()
    changed.iloc[int(len(frame) * 0.7) :] = 5.0  # validation and test turn into a jam
    assert np.array_equal(
        F.congestion_scores(frame, config), F.congestion_scores(changed, config)
    )
    locations = {str(i): {"latitude": 34.0, "longitude": -118.0} for i in range(8)}
    cfg = F.with_config(config, partition="congestion")
    first = F.partition_groups(frame, locations, cfg)
    second = F.partition_groups(changed, locations, cfg)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


# --- local graphs ---------------------------------------------------------------------


def test_local_graphs_hold_no_edge_between_clients():
    _, weights = synthetic()
    graphs = F.local_graphs(weights, GROUPS)
    assert [g.shape for g in graphs] == [(3, 3), (2, 2), (3, 3)]
    block = F.block_graph(weights, GROUPS)
    for i, a in enumerate(GROUPS):
        for j, b in enumerate(GROUPS):
            if i != j:
                assert not block[np.ix_(a, b)].any()
    for g, local in zip(GROUPS, graphs):
        assert np.array_equal(block[np.ix_(g, g)], local)


def test_graph_report_counts_cut_edges_and_isolated_sensors():
    weights = np.zeros((4, 4), dtype=np.float32)
    weights[0, 1] = weights[1, 0] = 1.0  # inside client 0
    weights[1, 2] = 0.5  # crosses to client 1
    weights[3, 2] = 0.5  # crosses to client 1... which owns 2 and 3
    report = F.graph_report(weights, [np.array([0, 1]), np.array([2, 3])])
    assert report["directed_edges"] == 4
    assert report["edges_kept"] == 3
    assert report["weight_kept_fraction"] == pytest.approx(2.5 / 3.0)
    # sensor 3's only edge stays inside client 1; the cut edge was 1 -> 2
    assert [c["locally_isolated_sensors"] for c in report["clients"]] == [0, 0]
    alone = F.graph_report(weights, [np.array([0, 2]), np.array([1, 3])])
    assert alone["edges_kept"] == 0
    assert [c["locally_isolated_sensors"] for c in alone["clients"]] == [2, 2]


# --- client data and leakage ----------------------------------------------------------


def test_client_scaler_and_training_windows_ignore_everything_else(config):
    frame, weights = synthetic()
    base = F.ClientData(frame.iloc[:, [0, 1, 2]], weights[:3, :3], config, 0)
    changed = frame.copy()
    changed.iloc[:, 3:] = 999.0  # other clients' sensors
    changed.iloc[int(len(frame) * 0.7) :, :3] = 1.0  # this client's val and test rows
    other = F.ClientData(changed.iloc[:, [0, 1, 2]], weights[:3, :3], config, 0)
    assert np.array_equal(base.scaler.mean, other.scaler.mean)
    assert np.array_equal(base.scaler.std, other.scaler.std)
    assert np.array_equal(
        base.datasets["train"].x, other.datasets["train"].x
    ) and np.array_equal(base.datasets["train"].y, other.datasets["train"].y)
    assert not np.array_equal(base.datasets["test"].y, other.datasets["test"].y)


def test_client_rejects_a_graph_for_other_sensors(config):
    frame, weights = synthetic()
    with pytest.raises(ValueError):
        F.ClientData(frame.iloc[:, :3], weights, config, 0)


def test_training_never_reads_validation_or_test_rows(config):
    frame, weights = synthetic()
    cfg = F.with_config(config, rounds=2)
    first = F.run_federated(
        cfg,
        F.build_clients(make_prep(frame, weights), cfg),
        F.initial_parameters(cfg),
        log=lambda *a: None,
    )
    changed = frame.copy()
    # rewrite the test period; training weights and validation scores must not move
    changed.iloc[int(len(frame) * 0.8) :] = 1.0
    second = F.run_federated(
        cfg,
        F.build_clients(make_prep(changed, weights), cfg),
        F.initial_parameters(cfg),
        log=lambda *a: None,
    )
    assert [h["state_sha256"] for h in first["history"]] == [
        h["state_sha256"] for h in second["history"]
    ]
    assert [h["validation"]["mae"] for h in first["history"]] == [
        h["validation"]["mae"] for h in second["history"]
    ]
    assert first["best_round"] == second["best_round"]
    assert first["test"]["mae"] != second["test"]["mae"]


# --- models and parameters ------------------------------------------------------------


@pytest.mark.parametrize("model", ["sgru", "sgraph"])
def test_parameters_do_not_depend_on_the_number_of_sensors(config, model):
    cfg = F.with_config(config, model=model)
    small = F.build_model(cfg, np.zeros((2, 2), dtype=np.float32))
    large = F.build_model(cfg, synthetic(sensors=8)[1])
    a, b = F.parameter_dict(small), F.parameter_dict(large)
    assert list(a) == list(b)
    assert all(a[k].shape == b[k].shape for k in a)
    assert F.parameter_count(a) == F.parameter_count(b)
    # graph buffers are not parameters, so they are never exchanged
    assert not any("incoming" in k or "outgoing" in k for k in a)


def test_load_parameters_checks_names_and_shapes(config):
    model = F.build_model(config, synthetic()[1])
    params = F.parameter_dict(model)
    wrong = OrderedDict(params)
    wrong["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        F.load_parameters(model, wrong)
    bad = OrderedDict((k, v.clone()) for k, v in params.items())
    first = next(iter(bad))
    bad[first] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        F.load_parameters(model, bad)


def test_same_start_weights_for_every_arm(config):
    first, second = F.initial_parameters(config), F.initial_parameters(config)
    assert F.state_hash(first) == F.state_hash(second)
    other = F.initial_parameters(F.with_config(config, seed=1))
    assert F.state_hash(first) != F.state_hash(other)


# --- local updates and aggregation ----------------------------------------------------


def test_local_update_is_deterministic_and_counts_examples(config, setup):
    _, clients, start = setup
    model = F.build_model(config, clients[0].weights)
    a, stats_a = F.local_update(model, start, clients[0], config, 11)
    b, stats_b = F.local_update(
        F.build_model(config, clients[0].weights), start, clients[0], config, 11
    )
    assert F.state_hash(a) == F.state_hash(b)
    assert stats_a["num-examples"] == len(clients[0].datasets["train"]) * 3
    assert stats_a["processed-windows"] == len(clients[0].datasets["train"]) // 2
    assert F.state_hash(a) != F.state_hash(start)


def test_fedavg_is_the_example_weighted_mean_of_the_local_updates(config, setup):
    _, clients, start = setup
    result = F.run_federated(config, clients, start, log=lambda *a: None)
    updates, weights = [], []
    for c in clients:
        updated, stats = F.local_update(
            F.build_model(config, c.weights),
            start,
            c,
            config,
            config.seed + 10000 + c.client_id,
        )
        updates.append(updated)
        weights.append(stats["num-examples"])
    # examples are windows x sensors, so a client with 2 of 8 sensors weighs 2/3 of one with 3
    assert weights[1] * 3 == weights[0] * 2 and weights[0] == weights[2]
    factors = [w / sum(weights) for w in weights]
    expected = OrderedDict()
    for name in start:
        total = None
        for update, factor in zip(updates, factors):
            part = update[name].numpy() * factor
            total = part if total is None else total + part
        expected[name] = torch.from_numpy(np.asarray(total))
    assert F.state_hash(expected) == result["state_sha256"]
    # clients with more sensors count for more
    assert factors[1] < factors[0]


def test_fedprox_stays_closer_to_the_global_model_as_mu_grows(config, setup):
    _, clients, start = setup
    distances = []
    for mu in (0.0, 1.0, 100.0):
        model = F.build_model(config, clients[0].weights)
        updated, _ = F.local_update(model, start, clients[0], config, 3, mu)
        distances.append(
            float(sum((updated[k] - start[k]).pow(2).sum() for k in start).sqrt())
        )
    assert distances[0] > distances[1] > distances[2]


def test_fedprox_uses_the_squared_norm(config, setup):
    """With one step and no data gradient, the proximal term alone moves w by lr-sized
    Adam steps in the direction of the global model; the squared form has a gradient
    mu * (w - w_global), which vanishes at the anchor."""
    _, clients, start = setup
    model = F.build_model(config, clients[0].weights)
    F.load_parameters(model, start)
    anchor = [p.detach().clone() for p in model.parameters()]
    prox = sum((p - a).pow(2).sum() for p, a in zip(model.parameters(), anchor))
    assert prox.item() == 0.0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    prox = sum((p - a).pow(2).sum() for p, a in zip(model.parameters(), anchor))
    count = sum(p.numel() for p in model.parameters())
    assert prox.item() == pytest.approx(0.25 * count)


def test_pooled_arm_sees_the_same_training_examples_as_the_federated_arms(
    config, setup
):
    prep, clients, _ = setup
    pooled = F.pooled_client(prep, config, "full")
    assert pooled.examples == sum(c.examples for c in clients)
    # the local-edges pooled model has exactly the clients' graphs and no others
    local = F.pooled_client(prep, config, "local")
    for g, c in zip(GROUPS, clients):
        assert np.array_equal(local.weights[np.ix_(g, g)], c.weights)
    assert np.count_nonzero(local.weights) == sum(
        np.count_nonzero(c.weights) for c in clients
    )
    with pytest.raises(ValueError):
        F.pooled_client(prep, config, "global")


def test_local_only_and_pooled_runs_are_deterministic(config, setup):
    prep, clients, start = setup
    one = F.run_local_only(config, clients, start, log=lambda *a: None)
    two = F.run_local_only(config, clients, start, log=lambda *a: None)
    assert one["test"] == two["test"]
    pooled = F.pooled_client(prep, config, "full")
    a = F.run_pooled(config, pooled, start, "p", prep["groups"], lambda *x: None)
    b = F.run_pooled(config, pooled, start, "p", prep["groups"], lambda *x: None)
    assert a["state_sha256"] == b["state_sha256"] and a["test"] == b["test"]


def test_pooled_model_scores_on_clients_add_up_to_the_pooled_score(config, setup):
    prep, _, start = setup
    pooled = F.pooled_client(prep, config, "full")
    result = F.run_pooled(config, pooled, start, "p", prep["groups"], lambda *x: None)
    records = [
        F.evaluate(F.build_model(config, pooled.weights), pooled, "test", config)
    ]
    assert records[0]["count"] == [
        sum(c["metrics"]["per_horizon"][h]["count"] for c in result["clients"])
        for h in range(config.horizon)
    ]


def test_pool_adds_sufficient_statistics_before_taking_ratios():
    a = {"ae": [2.0], "se": [4.0], "count": [1]}  # MAE 2 on one target
    b = {"ae": [12.0], "se": [100.0], "count": [3]}  # MAE 4 on three targets
    pooled = F.pool([a, b])
    assert pooled["mae"] == pytest.approx(14 / 4)  # not the mean of 2 and 4
    assert pooled["rmse"] == pytest.approx(np.sqrt(104 / 4))
    with pytest.raises(ValueError):
        F.pool([])


def test_persistence_and_historical_average_need_no_training(config, setup):
    prep, clients, _ = setup
    persistence = F.persistence_arm(clients, config)
    average = F.historical_average_arm(prep, clients, config)
    assert persistence["test"]["mae"] > 0 and average["test"]["mae"] > 0
    assert len(persistence["clients"]) == 3
    # each client's observed targets add up to the total
    counts = [c["metrics"]["per_horizon"][0]["count"] for c in persistence["clients"]]
    assert sum(counts) == persistence["test"]["per_horizon"][0]["count"]


# --- the Flower apps ------------------------------------------------------------------


def test_clientapp_train_equals_the_in_process_local_update(config, setup, tmp_path):
    prep, clients, start = setup
    job = tmp_path / "job"
    job.mkdir()
    FF.write_job(prep, config, job, start)
    context = Context(
        run_id=1,
        node_id=10,
        node_config={"partition-id": 1, "num-partitions": 3},
        state=RecordDict(),
        run_config={},
    )
    request = Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(start),
                "config": ConfigRecord(
                    {"job-dir": str(job), "server-round": 1, "proximal-mu": 0.5}
                ),
            }
        ),
        dst_node_id=10,
        message_type="train",
    )
    reply = FF.app(request, context)
    assert not reply.has_error()
    expected, stats = F.local_update(
        F.build_model(config, clients[1].weights),
        start,
        clients[1],
        config,
        config.seed + 10000 + 1,
        0.5,
    )
    got = OrderedDict(reply.content["arrays"].to_torch_state_dict())
    assert F.state_hash(got) == F.state_hash(expected)
    metrics = reply.content["metrics"]
    assert metrics["num-examples"] == stats["num-examples"]
    assert metrics["client-id"] == 1 and metrics["sensor-count"] == 2
    evaluation = Message(
        content=RecordDict(
            {
                "arrays": reply.content["arrays"],
                "config": ConfigRecord({"job-dir": str(job), "stage": "test"}),
            }
        ),
        dst_node_id=10,
        message_type="evaluate",
    )
    scored = FF.app(evaluation, context)
    record = dict(scored.content["metrics"])
    model = F.build_model(config, clients[1].weights)
    F.load_parameters(model, expected)
    direct = F.evaluate(model, clients[1], "test", config)
    assert (
        list(record["ae"]) == direct["ae"] and list(record["count"]) == direct["count"]
    )
    assert list(record["daily-abs"]) == direct["daily"]["abs"]


def test_clientapp_refuses_validation_leaks_and_unknown_clients(
    config, setup, tmp_path
):
    prep, _, start = setup
    job = tmp_path / "job"
    job.mkdir()
    FF.write_job(prep, config, job, start)

    def message(stage, partition):
        context = Context(
            run_id=1,
            node_id=10,
            node_config={"partition-id": partition, "num-partitions": 3},
            state=RecordDict(),
            run_config={},
        )
        request = Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord(start),
                    "config": ConfigRecord({"job-dir": str(job), "stage": stage}),
                }
            ),
            dst_node_id=10,
            message_type="evaluate",
        )
        return FF.app(request, context)

    with pytest.raises(ValueError):
        message("train", 0)  # only held-out stages can be scored
    with pytest.raises(ValueError):
        message("val", 7)  # no such client partition


def test_job_files_hold_only_each_clients_own_sensors(config, setup, tmp_path):
    prep, _, start = setup
    job = tmp_path / "job"
    job.mkdir()
    FF.write_job(prep, config, job, start)
    for client_id, group in enumerate(GROUPS):
        with np.load(job / "clients" / f"client-{client_id}.npz") as saved:
            assert saved["sensors"].tolist() == [str(i) for i in group]
            assert saved["values"].shape[1] == len(group)
            assert saved["weights"].shape == (len(group), len(group))
            assert np.array_equal(
                saved["weights"], prep["weights"][np.ix_(group, group)]
            )
    with pytest.raises(FileExistsError):
        FF.write_job(prep, config, job, start)  # never overwrite a prepared job


def test_server_rejects_missing_or_duplicate_client_replies():
    def reply(client):
        return Message(
            content=RecordDict({"metrics": MetricRecord({"client-id": client})}),
            dst_node_id=0,
            message_type="train",
        )

    ordered = FF.checked_replies([reply(2), reply(0), reply(1)], 3)
    assert [int(r.content["metrics"]["client-id"]) for r in ordered] == [0, 1, 2]
    with pytest.raises(RuntimeError):
        FF.checked_replies([reply(0), reply(0), reply(1)], 3)
    with pytest.raises(RuntimeError):
        FF.checked_replies([reply(0), reply(1)], 3)


def test_round_budgets_are_the_prefixes_of_a_longer_run(config, setup):
    _, clients, start = setup
    long = F.with_config(config, rounds=3, budgets=[1, 2, 3])
    result = F.run_federated(long, clients, start, log=lambda *a: None)
    table = result["budgets"]
    assert set(table) == {"1", "2", "3"}
    assert table["3"]["mae"] == result["test"]["mae"]
    assert table["3"]["best_round"] == result["best_round"]
    assert all(table[str(b)]["best_round"] <= b for b in (1, 2, 3))
    # stopping after two rounds gives exactly what a two-round run gives
    short = F.with_config(config, rounds=2)
    again = F.run_federated(short, clients, start, log=lambda *a: None)
    assert table["2"]["mae"] == again["test"]["mae"]
    assert table["2"]["best_round"] == again["best_round"]
    n, size = len(clients), result["parameter_bytes"]
    assert [table[str(b)]["communication_bytes"] for b in (1, 2, 3)] == [
        (3 * b + 1) * n * size for b in (1, 2, 3)
    ]


@pytest.mark.parametrize("budgets", [[0], [4], [1.5], [True]])
def test_budgets_must_be_round_counts(config, budgets):
    with pytest.raises(ValueError):
        F.with_config(config, rounds=3, budgets=budgets)


def test_personalization_never_gets_worse_than_the_federated_model_on_validation(
    config, setup
):
    _, clients, start = setup
    cfg = F.with_config(config, rounds=2, personalize_rounds=2)
    result = F.run_federated(cfg, clients, start, log=lambda *a: None)
    tuned = result["personalized"]
    assert tuned["rounds"] == 2 and len(tuned["clients"]) == len(clients)
    kept = result["history"][result["best_round"] - 1]["client_validation_mae"]
    for row, federated in zip(tuned["clients"], kept):
        assert 0 <= row["best_round"] <= 2
        # round 0 is the federated model itself, so the kept model can't be worse on the
        # client's own validation data than it is
        assert row["validation_mae"] <= federated + 1e-12
    # the federated scores themselves don't change when fine-tuning is switched on
    plain = F.run_federated(
        F.with_config(cfg, personalize_rounds=0), clients, start, log=lambda *a: None
    )
    assert plain["test"] == result["test"]
    assert plain["state_sha256"] == result["state_sha256"]


def test_zero_personalization_rounds_give_back_the_federated_scores(config, setup):
    _, clients, start = setup
    cfg = F.with_config(config, rounds=2)
    result = F.run_federated(cfg, clients, start, log=lambda *a: None)
    assert "personalized" not in result
    models = F._models(cfg, clients)
    zero = F.personalize(
        cfg, clients, models, _best_params(cfg, clients, start), 0, lambda *a: None
    )
    assert zero["test"] == result["test"]
    assert all(row["best_round"] == 0 for row in zero["clients"])


def _best_params(cfg, clients, start):
    """The federated model a run keeps, rebuilt through the public pieces."""
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    models = F._models(cfg, clients)
    params, best, kept = start, float("inf"), start
    for r in range(1, cfg.rounds + 1):
        replies = []
        for c in clients:
            updated, stats = F.local_update(
                models[c.client_id], params, c, cfg, cfg.seed + 10000 * r + c.client_id
            )
            replies.append(
                RecordDict(
                    {
                        "arrays": ArrayRecord(updated),
                        "metrics": MetricRecord(
                            {"num-examples": stats["num-examples"]}
                        ),
                    }
                )
            )
        params = OrderedDict(
            aggregate_arrayrecords(replies, "num-examples").to_torch_state_dict()
        )
        for m in models.values():
            F.load_parameters(m, params)
        validation, _ = F._validation_mae(models, clients, cfg)
        if validation["mae"] < best:
            best, kept = (
                validation["mae"],
                OrderedDict((k, v.clone()) for k, v in params.items()),
            )
    return kept


def test_personalization_rounds_must_be_nonnegative(config):
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            F.with_config(config, personalize_rounds=bad)
