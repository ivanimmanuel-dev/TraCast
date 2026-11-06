"""Diffusion convolution and the DCGRU model, checked against small known graphs."""

import numpy as np
import pytest
import torch

from tracast import fedgraph as F
from tracast.diffusion import (
    DCRNN,
    DiffusionConv,
    hop_stack,
    random_walk,
    transition_matrices,
)


def chain(n=4):
    """0 -> 1 -> 2 -> ... directed, with unit weights."""
    weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        weights[i, i + 1] = 1.0
    return weights


def random_graph(n, seed=0, density=0.4):
    rng = np.random.default_rng(seed)
    weights = rng.random((n, n)).astype(np.float32)
    weights[rng.random((n, n)) > density] = 0
    np.fill_diagonal(weights, 0)
    return weights


# --- transition matrices --------------------------------------------------------------


def test_random_walk_rows_sum_to_one_and_isolated_rows_stay_zero():
    weights = np.array([[0, 2, 2], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    walk = random_walk(weights)
    assert walk[0].tolist() == [0.0, 0.5, 0.5]
    assert walk[1].tolist() == [0.0, 0.0, 1.0]
    assert walk[2].tolist() == [0.0, 0.0, 0.0]  # no outgoing edge: nothing to walk to
    assert torch.isfinite(walk).all()


def test_forward_and_reverse_follow_and_oppose_the_road_direction():
    forward, reverse = transition_matrices(chain(), "both")
    # forward: from 1 the walk goes to 2; reverse: from 1 the walk goes back to 0
    assert forward[1].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert reverse[1].tolist() == [1.0, 0.0, 0.0, 0.0]
    # the first sensor has no upstream, the last has no downstream
    assert reverse[0].sum() == 0 and forward[3].sum() == 0


def test_reverse_walk_is_normalized_by_in_degree():
    weights = np.array([[0, 0, 1], [0, 0, 3], [0, 0, 0]], dtype=np.float32)
    (reverse,) = transition_matrices(weights, "reverse")
    # node 2 receives from 0 (weight 1) and 1 (weight 3), so D_in^-1 W^T gives 1/4, 3/4
    assert reverse[2].tolist() == [0.25, 0.75, 0.0]
    assert reverse.shape == (3, 3)


def test_modes_give_the_expected_number_of_matrices():
    graph = random_graph(5)
    assert len(transition_matrices(graph, "both")) == 2
    assert len(transition_matrices(graph, "forward")) == 1
    assert len(transition_matrices(graph, "reverse")) == 1
    (symmetric,) = transition_matrices(graph, "symmetric")
    undirected = np.maximum(graph, graph.T)
    expected = random_walk(undirected)
    assert torch.equal(symmetric, expected)
    with pytest.raises(ValueError):
        transition_matrices(graph, "diagonal")


@pytest.mark.parametrize(
    "bad",
    [np.ones((2, 3)), -np.ones((2, 2)), np.array([[0.0, np.nan], [0.0, 0.0]])],
)
def test_random_walk_rejects_invalid_adjacency(bad):
    with pytest.raises(ValueError):
        random_walk(bad)


def test_hop_stack_holds_the_powers_of_each_matrix():
    graph = random_graph(5, seed=3)
    powers = hop_stack(graph, 3, "both")
    forward, reverse = transition_matrices(graph, "both")
    assert powers.shape == (6, 5, 5)
    assert torch.allclose(powers[0], forward)
    assert torch.allclose(powers[1], forward @ forward, atol=1e-6)
    assert torch.allclose(powers[2], forward @ forward @ forward, atol=1e-6)
    assert torch.allclose(powers[3], reverse)
    assert torch.allclose(powers[5], reverse @ reverse @ reverse, atol=1e-6)
    assert hop_stack(graph, 0).shape == (0, 5, 5)
    with pytest.raises(ValueError):
        hop_stack(graph, -1)


def test_multi_hop_walks_reach_sensors_several_edges_away():
    powers = hop_stack(chain(4), 3, "forward")
    # forward walks along 0 -> 1 -> 2 -> 3: two hops from 0 land on 2, three on 3
    assert powers[1][0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert powers[2][0].tolist() == [0.0, 0.0, 0.0, 1.0]
    one = hop_stack(chain(4), 1, "forward")
    assert one.shape[0] == 1 and one[0][0].tolist() == [0.0, 1.0, 0.0, 0.0]


# --- the convolution ------------------------------------------------------------------


def test_diffusion_conv_matches_the_formula_computed_by_hand():
    torch.manual_seed(0)
    graph = random_graph(5, seed=2)
    powers = hop_stack(graph, 2, "both").double()
    layer = DiffusionConv(3, 4, hops=4).double()
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    got = layer(x, powers)
    weight = layer.linear.weight  # (4, (1 + 4) * 3)
    expected = layer.linear.bias.expand(2, 5, 4).clone()
    blocks = [x] + [torch.einsum("ij,bjc->bic", p, x) for p in powers]
    for k, block in enumerate(blocks):
        expected = expected + block @ weight[:, 3 * k : 3 * (k + 1)].T
    assert torch.allclose(got, expected, atol=1e-10)


def test_diffusion_conv_with_no_hops_ignores_the_graph():
    layer = DiffusionConv(3, 4, hops=0)
    x = torch.randn(2, 5, 3)
    assert torch.equal(layer(x, torch.zeros(0, 5, 5)), layer.linear(x))


# --- the model ------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["both", "forward", "reverse", "symmetric"])
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_output_shape_and_finite_values(mode, steps):
    model = DCRNN(
        random_graph(6), hidden=8, horizon=5, extra=3, diffusion_steps=steps, mode=mode
    )
    out = model(torch.randn(4, 12, 6 + 3))
    assert out.shape == (4, 5, 6)
    assert torch.isfinite(out).all()


def test_parameter_count_follows_the_diffusion_depth_not_the_sensor_count():
    def count(sensors, steps, mode="both", hidden=8, extra=3):
        model = DCRNN(
            random_graph(sensors), hidden, 12, extra, diffusion_steps=steps, mode=mode
        )
        return sum(p.numel() for p in model.parameters())

    assert count(6, 2) == count(40, 2)
    # (1 + 2 K) blocks of (1 + extra + hidden) inputs feed 3 * hidden gate outputs
    inputs = 1 + 3 + 8
    expected = (
        (1 + 4) * inputs * (2 * 8) + 2 * 8 + (1 + 4) * inputs * 8 + 8 + 8 * 12 + 12
    )
    assert count(6, 2) == expected
    assert count(6, 3) > count(6, 2) > count(6, 1) > count(6, 0)
    assert count(6, 2, "forward") < count(6, 2, "both")


def test_zero_hops_makes_the_forecast_independent_of_the_graph():
    torch.manual_seed(1)
    first = DCRNN(random_graph(6, 0), 8, 4, 3, diffusion_steps=0)
    second = DCRNN(random_graph(6, 1), 8, 4, 3, diffusion_steps=0)
    second.load_state_dict(
        {k: v for k, v in first.state_dict().items() if k != "powers"}, strict=False
    )
    x = torch.randn(3, 12, 9)
    assert torch.equal(first(x), second(x))


def test_with_hops_the_forecast_depends_on_the_graph():
    torch.manual_seed(2)
    first = DCRNN(random_graph(6, 0), 8, 4, 3, diffusion_steps=2)
    second = DCRNN(random_graph(6, 1), 8, 4, 3, diffusion_steps=2)
    second.load_state_dict(
        {k: v for k, v in first.state_dict().items() if k != "powers"}, strict=False
    )
    x = torch.randn(3, 12, 9)
    assert not torch.allclose(first(x), second(x))


def test_relabelling_the_sensors_relabels_the_forecast():
    torch.manual_seed(3)
    graph = random_graph(6, 4)
    perm = np.array([3, 0, 5, 1, 4, 2])
    model = DCRNN(graph, 8, 4, 3, diffusion_steps=2)
    twin = DCRNN(graph[np.ix_(perm, perm)], 8, 4, 3, diffusion_steps=2)
    twin.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k != "powers"}, strict=False
    )
    x = torch.randn(3, 12, 9)
    shuffled = x.clone()
    shuffled[..., :6] = x[..., :6][..., perm]
    assert torch.allclose(model(x)[..., perm], twin(shuffled), atol=1e-5)


def test_sensors_without_any_edge_still_train():
    graph = random_graph(6, 5)
    graph[2, :] = 0
    graph[:, 2] = 0  # sensor 2 has no edge in either direction
    model = DCRNN(graph, 8, 4, 3, diffusion_steps=2)
    out = model(torch.randn(3, 12, 9))
    loss = out.abs().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_graph_buffers_are_not_parameters():
    model = DCRNN(random_graph(6), 8, 4, 3)
    names = [n for n, _ in model.named_parameters()]
    assert "powers" not in names
    assert "powers" in dict(model.named_buffers())


def test_a_few_steps_reduce_the_loss_on_a_fixed_batch():
    torch.manual_seed(4)
    model = DCRNN(random_graph(6, 6), 8, 4, 3, diffusion_steps=2)
    x = torch.randn(16, 12, 9)
    target = torch.randn(16, 4, 6) * 0.3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = (model(x) - target).abs().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def test_stacked_layers_add_parameters_and_keep_the_output_shape():
    one = DCRNN(random_graph(5), 8, 4, 3, layers=1)
    two = DCRNN(random_graph(5), 8, 4, 3, layers=2)
    assert sum(p.numel() for p in two.parameters()) > sum(
        p.numel() for p in one.parameters()
    )
    assert two(torch.randn(2, 12, 8)).shape == (2, 4, 5)
    with pytest.raises(ValueError):
        DCRNN(random_graph(5), layers=0)


# --- federated use --------------------------------------------------------------------


def test_federated_config_builds_the_diffusion_model_with_shared_parameters():
    config = F.GraphFedConfig(
        sensors=16, hidden=8, model="dcrnn", diffusion_steps=2, diffusion_mode="both"
    )
    config.validate()
    small = F.build_model(config, random_graph(3))
    large = F.build_model(config, random_graph(11))
    a, b = F.parameter_dict(small), F.parameter_dict(large)
    assert list(a) == list(b) and all(a[k].shape == b[k].shape for k in a)
    start = F.initial_parameters(config)
    assert list(start) == list(a)
    F.load_parameters(large, start)
    with pytest.raises(ValueError):
        F.with_config(config, diffusion_mode="sideways")


# --- the benchmark ------------------------------------------------------------------


def test_model_names_encode_the_options():
    from tracast.diffusion import parse_name

    assert parse_name("sgru") is None and parse_name("dcrnnx") is None
    assert parse_name("dcrnn") == {
        "steps": 2,
        "mode": "both",
        "permute": False,
        "hidden": None,
        "layers": 1,
    }
    got = parse_name("dcrnn_k3_fwd_h16_l2")
    assert got == {
        "steps": 3,
        "mode": "forward",
        "permute": False,
        "hidden": 16,
        "layers": 2,
    }
    assert parse_name("dcrnn_bwd")["mode"] == "reverse"
    assert parse_name("dcrnn_sym")["mode"] == "symmetric"
    assert parse_name("dcrnn_perm")["permute"] is True
    for bad in ("dcrnn_k", "dcrnn_h0", "dcrnn_l9", "dcrnn_banana", "dcrnn_k10"):
        assert parse_name(bad) is None


def test_benchmark_accepts_and_builds_the_diffusion_names():
    import argparse

    from tracast.benchmark import (
        BenchmarkConfig,
        check_trial,
        model_for,
        valid_model,
    )

    config = BenchmarkConfig(sensors=6, hidden=8)
    graph = random_graph(6)
    assert valid_model("dcrnn_time") and valid_model("dcrnn_k1_perm_time")
    assert not valid_model("dcrnn_banana_time")
    assert check_trial("dcrnn_k0_time") == "dcrnn_k0_time"
    with pytest.raises(argparse.ArgumentTypeError):
        check_trial("dcrnn_banana")
    plain = model_for("dcrnn", config, graph)
    timed = model_for("dcrnn_time", config, graph)
    assert isinstance(timed, DCRNN) and timed.extra == 3 and plain.extra == 0
    assert model_for("dcrnn_h4_time", config, graph).hidden == 4
    assert model_for("dcrnn_k1_time", config, graph).diffusion_steps == 1
    assert model_for("dcrnn_fwd", config, graph).mode == "forward"
    assert model_for("dcrnn_time", config, graph)(torch.randn(2, 12, 9)).shape == (
        2,
        12,
        6,
    )


def test_permuted_graph_keeps_the_edges_but_changes_who_is_connected_to_whom():
    from tracast.diffusion import permuted

    graph = random_graph(8, seed=7)
    shuffled = permuted(graph, seed=1)
    assert np.array_equal(np.sort(graph.flatten()), np.sort(shuffled.flatten()))
    assert np.allclose(
        np.sort(graph.sum(0)), np.sort(shuffled.sum(0))
    )  # same in-degrees, relabelled
    assert not np.array_equal(graph, shuffled)
    assert np.array_equal(shuffled, permuted(graph, seed=1))
    assert not np.array_equal(shuffled, permuted(graph, seed=2))
    from tracast.benchmark import BenchmarkConfig, model_for

    config = BenchmarkConfig(sensors=8, hidden=8)
    real = model_for("dcrnn_time", config, graph, seed=1)
    fake = model_for("dcrnn_perm_time", config, graph, seed=1)
    assert not torch.equal(real.powers, fake.powers)
    assert torch.equal(
        fake.powers, model_for("dcrnn_perm_time", config, graph, seed=1).powers
    )


def test_the_benchmark_fit_trains_a_diffusion_model_with_calendar_inputs(tmp_path):
    import pandas as pd

    from dataclasses import replace

    from tracast.benchmark import BenchmarkConfig, fit, make_loaders, model_for
    from tracast.data import Scaler, Windows

    rng = np.random.default_rng(0)
    index = pd.date_range("2012-03-05", periods=500, freq="5min")
    values = 50 + 10 * np.sin(np.arange(500)[:, None] / 25 + np.arange(5))
    frame = pd.DataFrame(
        (values + rng.normal(0, 1, (500, 5))).astype("float32"),
        index=index,
        columns=list("abcde"),
    )
    scaler = Scaler().fit(frame)
    datasets = {
        "train": Windows(frame, scaler, time_features=True),
        "val": Windows(frame, scaler, time_features=True),
    }
    config = replace(
        BenchmarkConfig(sensors=5),
        hidden=8,
        epochs=4,
        batch_size=32,
        threads=1,
        patience=None,
        scheduler="none",
    )
    loaders = make_loaders(datasets, config, torch.Generator().manual_seed(0))
    model = model_for("dcrnn_time", config, random_graph(5, 3))
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    out = fit(model, loaders, mean, std, config, "t", tmp_path)
    maes = [h["validation"]["mae"] for h in out["history"]]
    assert len(maes) == 4 and maes[-1] < maes[0]
