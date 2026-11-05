"""Nested sensor subsets, shared-weight models and common-core scoring."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from tracast.benchmark import BenchmarkConfig, make_loaders, model_for
from tracast.data import Scaler, Windows
from tracast.experiment import evaluate
from tracast.shared import SharedRecurrent
from tracast.topology import connected_subset, nested_subset, order


def chain_graph(n, skip=()):
    weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        if i not in skip:
            weights[i, i + 1], weights[i + 1, i] = 0.6, 0.5
    return weights


def test_order_visits_every_sensor_once_and_starts_at_the_first():
    ids = [str(i) for i in range(8)]
    graph = chain_graph(8, skip={3})  # two components: 0-3 and 4-7
    result = order(graph, ids)
    assert sorted(result) == list(range(8)) and result[0] == 0
    assert result[:4] == [0, 1, 2, 3] and result[4:] == [4, 5, 6, 7]


def test_isolated_sensors_come_after_the_first_component():
    ids = list("abcde")
    graph = chain_graph(5, skip={2, 3})  # a-b-c, then d and e alone
    assert order(graph, ids) == [0, 1, 2, 3, 4]
    chosen, sub = nested_subset(graph, ids, 5)
    assert chosen == ids and sub.shape == (5, 5)


def test_prefixes_are_nested_and_match_the_original_subset_rule():
    rng = np.random.default_rng(3)
    n = 30
    graph = (rng.random((n, n)) < 0.08) * rng.random((n, n))
    np.fill_diagonal(graph, 0)
    graph = graph.astype(np.float32)
    ids = [f"s{i:02d}" for i in range(n)]
    full = order(graph, ids)
    for k in (5, 10, 20):
        assert nested_subset(graph, ids, k)[0] == [ids[i] for i in full[:k]]
    # While the first component is big enough, the prefix is the old connected subset.
    undirected = np.maximum(graph, graph.T) > 0
    reach, queue = {0}, [0]
    while queue:
        for v in np.flatnonzero(undirected[queue.pop()]):
            if v not in reach:
                reach.add(int(v))
                queue.append(int(v))
    for k in range(1, len(reach) + 1):
        ours, ours_graph = nested_subset(graph, ids, k)
        theirs, theirs_graph = connected_subset(graph, ids, k)
        assert ours == theirs
        np.testing.assert_array_equal(ours_graph, theirs_graph)


def test_nested_subset_rejects_bad_sizes():
    graph = chain_graph(4)
    for count in (0, 5):
        with pytest.raises(ValueError):
            nested_subset(graph, list("abcd"), count)


# --- shared models -----------------------------------------------------------


def parameters(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("kind", ["gru", "lstm"])
def test_parameter_count_does_not_depend_on_the_sensor_count(kind):
    small = SharedRecurrent(4, 8, 3, extra=3, weights=chain_graph(4), kind=kind)
    large = SharedRecurrent(40, 8, 3, extra=3, weights=chain_graph(40), kind=kind)
    assert parameters(small) == parameters(large)
    x = torch.randn(2, 12, 43)
    assert small(x[..., :7]).shape == (2, 3, 4)
    assert large(x).shape == (2, 3, 40)


def test_without_a_graph_a_sensor_only_sees_itself_and_the_calendar():
    torch.manual_seed(0)
    model = SharedRecurrent(5, 8, 3, extra=3)
    x = torch.randn(2, 12, 8)
    before = model(x)
    changed = x.clone()
    changed[..., 3] += 5.0  # sensor 3's speeds
    after = model(changed)
    others = [0, 1, 2, 4]
    assert torch.equal(before[..., others], after[..., others])
    assert not torch.allclose(before[..., 3], after[..., 3])
    shifted = x.clone()
    shifted[..., 5:] += 1.0  # calendar channels reach every sensor
    assert not torch.allclose(before, model(shifted))


def test_with_a_graph_only_neighbors_leak_in():
    torch.manual_seed(1)
    graph = np.zeros((4, 4), dtype=np.float32)
    graph[0, 1] = 1.0  # sensor 0 -> sensor 1
    model = SharedRecurrent(4, 8, 3, weights=graph)
    x = torch.randn(2, 12, 4)
    base = model(x)
    changed = x.clone()
    changed[..., 0] += 3.0
    moved = model(changed)
    # Sensor 1 has sensor 0 as an upstream neighbor, sensors 2 and 3 don't.
    assert not torch.allclose(base[..., 1], moved[..., 1])
    assert torch.equal(base[..., [2, 3]], moved[..., [2, 3]])


def test_sensor_permutation_permutes_the_forecasts():
    torch.manual_seed(2)
    graph = chain_graph(4)
    perm = torch.tensor([2, 0, 3, 1])
    model = SharedRecurrent(4, 8, 3, extra=3, weights=graph)
    same = SharedRecurrent(4, 8, 3, extra=3, weights=graph[perm][:, perm])
    weights_only = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith(("incoming", "outgoing"))
    }
    same.load_state_dict(weights_only, strict=False)
    x = torch.randn(2, 12, 7)
    permuted = x.clone()
    permuted[..., :4] = x[..., :4][..., perm]
    torch.testing.assert_close(
        same(permuted), model(x)[..., perm], atol=1e-6, rtol=1e-5
    )


def test_shared_models_learn_through_every_parameter():
    model = SharedRecurrent(4, 8, 3, extra=3, weights=chain_graph(4), embedding=2)
    model(torch.randn(2, 12, 7)).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_invalid_shared_models_are_rejected():
    with pytest.raises(ValueError):
        SharedRecurrent(4, kind="rnn")
    with pytest.raises(ValueError):
        SharedRecurrent(4, weights=np.zeros((3, 3), dtype=np.float32))


def test_benchmark_builds_the_shared_models():
    config = BenchmarkConfig(sensors=4, hidden=8)
    graph = chain_graph(4)
    for name in ("sgru", "slstm", "sgraph"):
        plain = model_for(name, config, graph)
        timed = model_for(name + "_time", config, graph)
        assert parameters(timed) > parameters(plain)
        assert plain(torch.randn(2, 12, 7)).shape == (2, 12, 4)
    assert model_for("sgraph", config, graph).graph
    assert not model_for("sgru", config, graph).graph


@pytest.mark.parametrize(
    "changes", [{"train_stride": 0}, {"core_sizes": [0]}, {"core_sizes": [17]}]
)
def test_new_scaling_options_validate(changes):
    with pytest.raises(ValueError):
        replace(BenchmarkConfig(), **changes).validate()


# --- common-core scoring and strided epochs ----------------------------------


@pytest.fixture
def frame():
    index = pd.date_range("2012-03-05", periods=400, freq="5min")
    rng = np.random.default_rng(0)
    values = (
        50
        + 10 * np.sin(np.arange(400)[:, None] / 30 + np.arange(6))
        + rng.normal(0, 1, (400, 6))
    )
    return pd.DataFrame(values.astype("float32"), index=index, columns=list("abcdef"))


def test_core_scores_equal_scoring_the_first_sensors_alone(frame):
    scaler = Scaler().fit(frame)
    windows = Windows(frame, scaler)
    loader = torch.utils.data.DataLoader(windows, batch_size=50)
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    full = evaluate(None, loader, mean, std, 12, subsets=[2, 4])
    narrow = frame.iloc[:, :2]
    small_scaler = Scaler().fit(narrow)
    small = Windows(narrow, small_scaler)
    alone = evaluate(
        None,
        torch.utils.data.DataLoader(small, batch_size=50),
        torch.from_numpy(small_scaler.mean),
        torch.from_numpy(small_scaler.std),
        12,
    )
    assert full["core"]["2"]["mae"] == pytest.approx(alone["mae"])
    assert full["core"]["2"]["rmse"] == pytest.approx(alone["rmse"])
    assert set(full["core"]) == {"2", "4"}
    assert "core" not in evaluate(None, loader, mean, std, 12)


def test_train_stride_shortens_the_epoch_and_keeps_it_seeded(frame):
    scaler = Scaler().fit(frame)
    datasets = {"train": Windows(frame, scaler), "val": Windows(frame, scaler)}
    config = replace(BenchmarkConfig(), batch_size=10, train_stride=4)
    first = make_loaders(datasets, config, torch.Generator().manual_seed(5))
    second = make_loaders(datasets, config, torch.Generator().manual_seed(5))
    a, b = list(first["train"]), list(second["train"])
    assert sum(len(x) for x, _, _ in a) == len(datasets["train"]) // 4
    assert all(torch.equal(u[0], v[0]) for u, v in zip(a, b))
    plain = make_loaders(datasets, replace(config, train_stride=1), torch.Generator())
    assert sum(len(x) for x, _, _ in plain["train"]) == len(datasets["train"])
    assert sum(len(x) for x, _, _ in first["val"]) == len(datasets["val"])
