"""Held-out sensors: who is new, what a new sensor may use, and the vector fixes."""

import numpy as np
import pytest
import torch

from tracast import fedgraph as F
from tracast import inductive as I

from test_fedgraph import synthetic


def config(**changes):
    values = dict(
        sensors=16,
        rows=1500,
        hidden=4,
        embedding=2,
        rounds=1,
        local_stride=2,
        batch_size=16,
        few_shot_days=1,
        fit_epochs=3,
        threads=1,
        seeds=[1],
    )
    values.update(changes)
    cfg = I.InductiveConfig(**values)
    cfg.validate()
    return cfg


def test_config_rejects_invalid_settings():
    for changes in (
        {"holdout": "alphabetical"},
        {"holdout_fraction": 0.0},
        {"holdout_fraction": 0.6},
        {"sensors": 8},
        {"few_shot_days": 0},
        {"few_shot_days": 30, "rows": 1500},
        {"fit_lr": float("nan")},
        {"seeds": []},
        {"seeds": [1, 1]},
        {"ridge_alpha": -1.0},
    ):
        with pytest.raises(ValueError):
            config(**changes)


def test_config_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"sensors": 16, "rows": 1500, "banana": 1}')
    with pytest.raises(ValueError):
        I.load_config(path)


# --- who is new -----------------------------------------------------------------------


def locations(count=16):
    # four tight groups of four sensors, far apart
    return {
        str(i): {
            "latitude": 34.0 + 0.5 * (i // 4) + 0.001 * (i % 4),
            "longitude": -118.0 + 0.5 * ((i // 4) % 2),
        }
        for i in range(count)
    }


def test_random_holdout_has_the_asked_size_and_is_seeded():
    ids = [str(i) for i in range(16)]
    cfg = config(holdout="random", holdout_fraction=0.25, holdout_seed=3)
    first = I.choose_holdout(ids, locations(), cfg)
    assert len(first) == 4 and list(first) == sorted(first)
    assert np.array_equal(first, I.choose_holdout(ids, locations(), cfg))
    other = I.choose_holdout(ids, locations(), config(holdout="random", holdout_seed=4))
    assert not np.array_equal(first, other)


def test_region_holdout_is_one_whole_area():
    ids = [str(i) for i in range(16)]
    held = I.choose_holdout(ids, locations(), config(holdout="region"))
    assert len(held) == 4
    assert len({i // 4 for i in held}) == 1  # all four sit in the same group


def test_holdout_choice_never_reads_a_speed():
    ids = [str(i) for i in range(16)]
    # nothing but ids and coordinates go in, so no reading can influence the choice
    import inspect

    assert list(inspect.signature(I.choose_holdout).parameters) == [
        "ids",
        "locations",
        "config",
    ]
    assert len(I.choose_holdout(ids, locations(), config())) >= 1


def test_mixed_scaler_gives_new_sensors_only_their_first_days():
    frame, _ = synthetic(rows=1500, sensors=8)
    held = np.array([2, 5])
    split = int(len(frame) * 0.7)
    base = I.mixed_scaler(frame, np.setdiff1d(np.arange(8), held), held, 1, split)
    changed = frame.copy()
    changed.iloc[288:, held] = 500.0  # the new sensors' later rows, even in training
    other = I.mixed_scaler(changed, np.setdiff1d(np.arange(8), held), held, 1, split)
    assert np.array_equal(base.mean, other.mean) and np.array_equal(base.std, other.std)
    # trained sensors use their whole training period
    rewritten = frame.copy()
    rewritten.iloc[800:split, 0] = 5.0
    third = I.mixed_scaler(rewritten, np.setdiff1d(np.arange(8), held), held, 1, split)
    assert third.mean[0] != base.mean[0]


def test_descriptors_use_only_the_first_days_and_the_location():
    frame, _ = synthetic(rows=1500, sensors=8)
    place = {str(i): {"latitude": 34.0 + i, "longitude": -118.0 - i} for i in range(8)}
    base = I.descriptors(frame, place, 1)
    changed = frame.copy()
    changed.iloc[300:] = 1.0
    assert np.array_equal(base, I.descriptors(changed, place, 1))
    assert base.shape == (8, 5) and base[3, 3] == 37.0 and base[3, 4] == -121.0


# --- the ridge fix --------------------------------------------------------------------


def test_ridge_recovers_a_linear_map_and_leaves_the_intercept_alone():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    true = np.array([[1.0, -2.0], [0.5, 0.0], [0.0, 3.0]])
    y = x @ true + np.array([10.0, -5.0])
    new = rng.normal(size=(5, 3))
    got = I.ridge_predict(x, y, new, alpha=1e-9)
    assert np.allclose(got, new @ true + np.array([10.0, -5.0]), atol=1e-4)
    # a huge penalty shrinks the slopes to nothing but keeps the average
    flat = I.ridge_predict(x, y, new, alpha=1e12)
    assert np.allclose(flat, np.tile(y.mean(axis=0), (5, 1)), atol=1e-3)


def test_ridge_copes_with_a_constant_descriptor():
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    y = np.arange(10.0)[:, None] * 2
    got = I.ridge_predict(x, y, np.array([[1.0, 4.0]]), alpha=1e-9)
    assert np.isfinite(got).all() and got[0, 0] == pytest.approx(8.0, abs=1e-3)


# --- the vector fit -------------------------------------------------------------------


def make_setup(rows=1500):
    frame, weights = synthetic(rows=rows, sensors=8)
    cfg = config(rows=rows)
    fed = cfg.fed("sgru_id", 1)
    held = np.array([2, 5])
    trained = np.setdiff1d(np.arange(8), held)
    scaler = I.mixed_scaler(frame, trained, held, 1, int(rows * 0.7))
    data = F.ClientData(frame, weights, fed, scaler=scaler)
    model = F.build_model(fed, weights)
    torch.manual_seed(0)
    with torch.no_grad():
        model.node.normal_(0, 0.5)
    rows_t = model.node[trained].detach().clone()
    return cfg, fed, data, model, trained, held, rows_t, frame, weights


def test_the_fit_moves_only_the_new_sensors_rows():
    cfg, fed, data, model, trained, held, rows_t, *_ = make_setup()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    log = lambda *a: None  # noqa: E731
    out = I.fit_vectors(model, rows_t, trained, held, data, cfg, fed, 1, log)
    after = dict(model.named_parameters())
    for name, value in before.items():
        if name != "node":
            assert torch.equal(value, after[name]), name  # the network is frozen
    assert torch.equal(before["node"][trained], after["node"][trained])
    assert not torch.equal(before["node"][held], after["node"][held])
    assert all(p.requires_grad for p in model.parameters())  # restored afterwards
    assert len(out["validation_mae"]) == 3
    assert out["best_epoch"] == int(np.argmin(out["validation_mae"])) + 1


def test_the_kept_vector_is_the_best_validation_epoch():
    cfg, fed, data, model, trained, held, rows_t, *_ = make_setup()
    out = I.fit_vectors(
        model, rows_t, trained, held, data, cfg, fed, 1, lambda *a: None
    )
    again = F.pool([F.evaluate(model, data, "val", fed, False, only=held)])["mae"]
    assert again == pytest.approx(min(out["validation_mae"]), rel=1e-5)


def test_the_fit_never_reads_the_new_sensors_later_training_rows():
    cfg, fed, data, model, trained, held, rows_t, frame, weights = make_setup()
    start = {k: v.clone() for k, v in model.state_dict().items()}
    first = I.fit_vectors(
        model, rows_t, trained, held, data, cfg, fed, 1, lambda *a: None
    )
    vectors = model.node.detach().clone()

    changed = frame.copy()
    changed.iloc[288 : int(len(frame) * 0.7), held] = 1.0  # after the first day
    scaler = I.mixed_scaler(changed, trained, held, 1, int(len(frame) * 0.7))
    data2 = F.ClientData(changed, weights, fed, scaler=scaler)
    model2 = F.build_model(fed, weights)
    model2.load_state_dict(start)
    second = I.fit_vectors(
        model2, rows_t, trained, held, data2, cfg, fed, 1, lambda *a: None
    )
    assert first["validation_mae"] == second["validation_mae"]
    assert torch.equal(vectors, model2.node.detach())


def test_with_vectors_puts_every_row_where_it_belongs():
    cfg, fed, data, model, trained, held, rows_t, *_ = make_setup()
    params = F.parameter_dict(model)
    params["node"] = torch.arange(len(trained) * 2, dtype=torch.float32).reshape(-1, 2)
    fresh = F.build_model(fed, data.weights)
    new = torch.full((len(held), 2), -1.0)
    I.with_vectors(fresh, params, trained, held, new)
    assert torch.equal(fresh.node[trained], params["node"])
    assert torch.equal(fresh.node[held], new)
    assert torch.equal(fresh.rnn.weight_ih_l0, params["rnn.weight_ih_l0"])


# --- scoring --------------------------------------------------------------------------


def test_held_out_scores_count_only_the_held_out_targets():
    cfg, fed, data, model, trained, held, *_ = make_setup()
    scored = I.score_held(model, data, held, fed)
    everything = F.pool([F.evaluate(model, data, "test", fed, False)])
    one = F.pool([F.evaluate(model, data, "test", fed, False, only=[2])])
    assert (
        scored["test"]["per_horizon"][0]["count"]
        < everything["per_horizon"][0]["count"]
    )
    assert one["per_horizon"][0]["count"] < scored["test"]["per_horizon"][0]["count"]
    assert sum(scored["daily"]["count"]) == sum(
        h["count"] for h in scored["test"]["per_horizon"]
    )


def test_evaluate_only_keeps_the_models_view_of_all_sensors():
    cfg, fed, data, model, trained, held, *_ = make_setup()
    plain = F.evaluate(model, data, "test", fed, False, by_sensor=True)
    restricted = F.evaluate(model, data, "test", fed, False, by_sensor=True, only=held)
    # the restricted cells are the full cells on those sensors and zero elsewhere
    full = np.array(plain["cells"]["ae"])
    part = np.array(restricted["cells"]["ae"])
    assert np.allclose(part[:, held], full[:, held])
    assert not part[:, trained].any()


def test_historical_average_for_new_sensors_uses_only_the_days_it_is_given():
    cfg, fed, data, model, trained, held, rows_t, frame, weights = make_setup()
    base = I.historical_average(frame, data, held, fed, days=1)
    changed = frame.copy()
    changed.iloc[288 : int(len(frame) * 0.7), held] = 1.0
    data2 = F.ClientData(changed, weights, fed, scaler=data.scaler)
    other = I.historical_average(changed, data2, held, fed, days=1)
    assert base["test"]["mae"] == other["test"]["mae"]
    # with every training row, the same rewrite changes the table
    full = I.historical_average(changed, data2, held, fed)
    assert full["test"]["mae"] != base["test"]["mae"]


def test_pooled_only_models_cannot_be_federated(tmp_path):
    cfg = config()
    fed = cfg.fed("sgru_id", 1)
    frame, weights = synthetic(rows=1500, sensors=8)
    prep = {
        "frame": frame,
        "ids": list(frame.columns),
        "weights": weights,
        "groups": [np.arange(4), np.arange(4, 8)],
        "graph_meta": {},
    }
    clients = F.build_clients(prep, fed)
    with pytest.raises(ValueError):
        F.run_federated(fed, clients, F.initial_parameters(fed), log=lambda *a: None)
    # but pooled training works, with one vector row per sensor
    start = F.initial_parameters(fed, sensors=8)
    assert start["node"].shape == (8, 2)
    data = F.ClientData(frame, weights, fed)
    result, params = F.run_pooled(
        fed, data, start, "p", [np.arange(8)], lambda *a: None, return_params=True
    )
    assert params["node"].shape == (8, 2) and result["test"]["mae"] > 0
