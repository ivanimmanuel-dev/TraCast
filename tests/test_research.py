"""Constructed arrays check the research additions; real-data results live in reports."""

from dataclasses import replace
import io
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest
import torch

from tracast.baselines import HistoricalAverage, score_table, week_slots
from tracast.data import Scaler, Windows, calendar_features, split
from tracast.model import GRU, DailyMetrics, LinearAR, Metrics
from tracast.provenance import git_state
from tracast.recurrent import LSTM
from tracast.stats import day_bootstrap


@pytest.fixture
def frame():
    index = pd.date_range("2012-03-05", periods=1200, freq="5min")  # a Monday
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": 40 + 10 * np.sin(np.arange(1200) / 40) + rng.normal(0, 1, 1200),
            "b": 60 + rng.normal(0, 1, 1200),
        },
        index=index,
    ).astype("float32")


# --- time features -------------------------------------------------------


def test_time_features_cycle_and_weekend():
    index = pd.DatetimeIndex(
        ["2012-03-05 00:00", "2012-03-05 06:00", "2012-03-10 12:00"]
    )  # Monday midnight, Monday 06:00, Saturday noon
    features = calendar_features(index)
    assert features.shape == (3, 3) and features.dtype == np.float32
    np.testing.assert_allclose(features[0, :2], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(features[1, :2], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(features[:, 0] ** 2 + features[:, 1] ** 2, 1, atol=1e-6)
    assert list(features[:, 2]) == [0.0, 0.0, 1.0]


def test_time_channels_extend_inputs_without_touching_targets(frame):
    scaler = Scaler().fit(frame)
    plain = Windows(frame, scaler)
    timed = Windows(frame, scaler, time_features=True)
    assert plain.extra == 0 and timed.extra == 3
    x, y, mask = timed[5]
    px, py, pmask = plain[5]
    assert x.shape == (12, 5) and px.shape == (12, 2)
    torch.testing.assert_close(x[:, :2], px)
    assert torch.equal(y, py) and torch.equal(mask, pmask)


def test_origin_days_follow_first_target_row(frame):
    windows = Windows(frame, Scaler().fit(frame), history=12, horizon=12)
    codes, labels = windows.origin_days()
    assert len(codes) == len(windows)
    assert labels[0] == "2012-03-05"
    # Row 288 is midnight of the second day; its window starts 12 rows earlier.
    assert codes[288 - 12 - 1] == 0 and codes[288 - 12] == 1


# --- model plumbing ------------------------------------------------------


def test_extra_channels_default_to_the_published_parameter_counts():
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    assert count(GRU(16, 32, 12)) == 11136
    assert count(LSTM(16, 32, 12)) == 12736


def test_models_ignore_channels_they_do_not_declare():
    torch.manual_seed(1)
    x = torch.randn(3, 12, 5)
    for build in (
        lambda: GRU(2, 4, 3),
        lambda: LSTM(2, 4, 3),
        lambda: LinearAR(2, 12, 3),
    ):
        torch.manual_seed(2)
        model = build()
        assert torch.equal(model(x), model(x[..., :2]))


def test_models_use_declared_extra_channels():
    x = torch.randn(3, 12, 5)
    for build in (
        lambda: GRU(2, 4, 3, extra=3),
        lambda: LSTM(2, 4, 3, extra=3),
        lambda: LinearAR(2, 12, 3, extra=3),
    ):
        model = build()
        shifted = x.clone()
        shifted[..., 2:] += 1
        assert model(x).shape == (3, 3, 2)
        assert not torch.allclose(model(x), model(shifted))
        model(x).square().mean().backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )


def test_graph_gru_time_channels():
    from tracast.graph_model import GraphGRU

    graph = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    plain = GraphGRU(graph, hidden=4, horizon=3)
    timed = GraphGRU(graph, hidden=4, horizon=3, extra=3)
    x = torch.randn(2, 12, 5)
    assert plain(x).shape == timed(x).shape == (2, 3, 2)
    shifted = x.clone()
    shifted[..., 2:] += 1
    assert torch.equal(plain(x), plain(shifted))
    assert not torch.allclose(timed(x), timed(shifted))
    assert (
        sum(p.numel() for p in timed.parameters())
        == sum(p.numel() for p in plain.parameters()) + 3 * 4 * 3
    )  # three gate blocks of hidden width 4 times three new inputs


# --- daily metrics -------------------------------------------------------


def test_daily_metrics_partition_the_totals():
    codes = np.array([0, 0, 1, 2])
    daily = DailyMetrics(codes, ["d0", "d1", "d2"])
    totals = Metrics(2)
    prediction = torch.tensor(np.arange(4 * 2 * 1, dtype=np.float32).reshape(4, 2, 1))
    target = torch.zeros(4, 2, 1)
    mask = torch.ones(4, 2, 1, dtype=torch.bool)
    mask[3] = False
    for start, stop in ((0, 3), (3, 4)):
        daily.update(
            prediction[start:stop], target[start:stop], mask[start:stop], start
        )
        totals.update(prediction[start:stop], target[start:stop], mask[start:stop])
    result = daily.result()
    assert result["days"] == ["d0", "d1", "d2"]
    assert result["count"] == [4, 2, 0]
    assert result["abs"] == [0 + 1 + 2 + 3, 4 + 5, 0]
    assert sum(result["abs"]) == pytest.approx(totals.absolute.sum().item())
    assert sum(result["sq"]) == pytest.approx(totals.squared.sum().item())


def test_evaluate_fills_daily_without_changing_scores(frame):
    from tracast.experiment import evaluate

    scaler = Scaler().fit(frame)
    windows = Windows(frame, scaler)
    loader = torch.utils.data.DataLoader(windows, batch_size=37)
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    plain = evaluate(None, loader, mean, std, 12)
    codes, labels = windows.origin_days()
    daily = DailyMetrics(codes, labels)
    again = evaluate(None, loader, mean, std, 12, daily=daily)
    assert plain == again
    result = daily.result()
    assert sum(result["abs"]) / sum(result["count"]) == pytest.approx(plain["mae"])


def test_persistence_ignores_time_channels(frame):
    from tracast.experiment import evaluate

    scaler = Scaler().fit(frame)
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    plain = Windows(frame, scaler)
    timed = Windows(frame, scaler, time_features=True)
    first = evaluate(
        None, torch.utils.data.DataLoader(plain, batch_size=64), mean, std, 12
    )
    second = evaluate(
        None, torch.utils.data.DataLoader(timed, batch_size=64), mean, std, 12
    )
    assert first == second


# --- historical average --------------------------------------------------


def weekly_frame(weeks, noise_week=None):
    index = pd.date_range("2012-03-05", periods=weeks * 2016, freq="5min")
    slot = week_slots(index)
    pattern = 30 + slot / 100.0  # distinct value for every weekly slot
    values = np.stack([pattern, pattern + 5], axis=1).astype(np.float32)
    if noise_week is not None:
        values[noise_week * 2016 : (noise_week + 1) * 2016] = 99999
    return index, values


def test_week_slots_are_monday_first_five_minute_bins():
    index = pd.DatetimeIndex(
        ["2012-03-05 00:00", "2012-03-05 00:05", "2012-03-11 23:55"]
    )
    assert list(week_slots(index)) == [0, 1, 2015]


def test_historical_average_recovers_a_weekly_pattern():
    index, values = weekly_frame(3)
    model = HistoricalAverage().fit(index, values, np.ones_like(values, dtype=bool))
    later = pd.date_range("2012-03-26", periods=2016, freq="5min")
    expected = np.stack([30 + week_slots(later) / 100.0] * 2, axis=1)
    expected[:, 1] += 5
    np.testing.assert_allclose(model.predict(later), expected, rtol=1e-5)


def test_historical_average_uses_only_observed_values_and_falls_back():
    index, values = weekly_frame(3)
    mask = np.ones_like(values, dtype=bool)
    values[: 2016 * 2, 0] = 0  # sensor 0 unobserved in weeks 1 and 2
    mask[: 2016 * 2, 0] = False
    mask[100, 1] = False  # one slot of sensor 1 has no week-1 reading
    values[100, 1] = 0
    model = HistoricalAverage().fit(index, values, mask)
    slot = week_slots(index[[100]])[0]
    # Week 1 slot 100 is missing for sensor 1, but weeks 2 and 3 still agree.
    assert model.table[slot, 1] == pytest.approx(30 + slot / 100.0 + 5, rel=1e-5)
    assert model.table[slot, 0] == pytest.approx(30 + slot / 100.0, rel=1e-5)
    mask[:, 0] = False
    with pytest.raises(ValueError, match="no observed"):
        HistoricalAverage().fit(index, values, mask)


def test_historical_average_slot_fallback_to_time_of_day():
    index = pd.date_range("2012-03-05", periods=288 * 2, freq="5min")  # Mon, Tue
    values = np.full((len(index), 1), 50.0, dtype=np.float32)
    values[288:] = 70.0
    model = HistoricalAverage().fit(index, values, np.ones_like(values, dtype=bool))
    wednesday = pd.DatetimeIndex(["2012-03-07 12:00"])
    # Wednesday was never observed: pooled time-of-day mean across seen weekdays.
    assert model.predict(wednesday)[0, 0] == pytest.approx(60.0)


def test_score_table_matches_direct_computation(frame):
    scaler = Scaler().fit(frame)
    windows = Windows(frame, scaler)
    table = windows.y + 2.0
    result = score_table(windows, table, chunk=100)
    # Errors are exactly 2 mph wherever the target was observed.
    assert result["mae"] == pytest.approx(2.0, abs=1e-5)
    assert result["per_horizon"][0]["count"] == int(
        windows.mask[12 : 12 + len(windows)].sum()
    )
    with pytest.raises(ValueError):
        score_table(windows, table[:-1])


# --- statistics ----------------------------------------------------------


def daily(abs_error, count=100):
    n = len(abs_error)
    return {
        "days": [f"d{i}" for i in range(n)],
        "abs": list(map(float, abs_error)),
        "count": [count] * n,
    }


def test_day_bootstrap_separates_a_consistent_gap():
    rng = np.random.default_rng(1)
    base = 300 + rng.normal(0, 20, 24)
    worse = day_bootstrap([daily(base + 30)], [daily(base)], resamples=2000, seed=3)
    assert worse["estimate"] == pytest.approx(0.30)
    assert worse["ci95"][0] > 0 and worse["share_below_zero"] == 0
    tie = day_bootstrap([daily(base)], [daily(base)], resamples=2000)
    assert tie["estimate"] == 0 and tie["ci95"] == [0.0, 0.0]


def test_day_bootstrap_reports_wide_interval_for_inconsistent_gap():
    gap = np.array([-40, 40] * 12, dtype=float)
    result = day_bootstrap(
        [daily(300 + gap)], [daily(np.full(24, 300.0))], resamples=2000
    )
    assert result["ci95"][0] < 0 < result["ci95"][1]


def test_day_bootstrap_is_repeatable_and_averages_runs():
    runs = [daily(np.full(10, 300.0) + k) for k in (0, 10)]
    first = day_bootstrap(runs, resamples=500, seed=9)
    assert first == day_bootstrap(runs, resamples=500, seed=9)
    assert first["estimate"] == pytest.approx(3.05)
    assert first["days"] == 10


@pytest.mark.parametrize(
    "candidate,reference",
    [
        (daily([1, 2, 3]), daily([1, 2])),
        (daily([1, 2, 3], count=10), daily([1, 2, 3], count=11)),
        (daily([1]), daily([1])),
    ],
)
def test_day_bootstrap_rejects_mismatched_or_tiny_inputs(candidate, reference):
    with pytest.raises(ValueError):
        day_bootstrap([candidate], [reference], resamples=200)


# --- provenance ----------------------------------------------------------


def test_git_state_is_unknown_outside_a_checkout(tmp_path):
    assert git_state(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_git_state_reads_the_given_directory_not_the_cwd():
    state = git_state()
    assert state["git_commit"] is None or len(state["git_commit"]) == 40
    assert state["git_dirty"] in (None, True, False)


# --- download retry ------------------------------------------------------


def test_download_retries_transient_network_errors(tmp_path, monkeypatch):
    import tracast.data as data

    calls = []

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.URLError("temporary")
        return io.BytesIO(b"payload")

    monkeypatch.setattr(data.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    target = tmp_path / "file.partial"
    data.download("https://example.invalid/f", target, retries=3)
    assert target.read_bytes() == b"payload" and len(calls) == 3


def test_download_gives_up_after_the_retry_budget(tmp_path, monkeypatch):
    import tracast.data as data

    def down(url, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(data.urllib.request, "urlopen", down)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    with pytest.raises(urllib.error.URLError):
        data.download("https://example.invalid/f", tmp_path / "f.partial", retries=2)


# --- benchmark additions -------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"models": ["gru", "gru_time", "bogus_time"]},
        {"patience": 0},
        {"scheduler": "cosine"},
        {"scheduler_patience": 0},
        {"models": ["ha", "ha"]},
    ],
)
def test_new_benchmark_options_validate(changes):
    from tracast.benchmark import BenchmarkConfig

    with pytest.raises(ValueError):
        replace(BenchmarkConfig(), **changes).validate()


def test_new_model_names_build_and_time_models_take_more_inputs():
    from tracast.benchmark import BenchmarkConfig, model_for

    config = BenchmarkConfig(sensors=3)
    graph = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    assert model_for("persistence", config, graph) is None
    assert model_for("ha", config, graph) is None
    for name in ("gru", "lstm", "graph_gru", "graph_self", "linear"):
        plain = model_for(name, config, graph)
        timed = model_for(name + "_time", config, graph)
        assert sum(p.numel() for p in timed.parameters()) > sum(
            p.numel() for p in plain.parameters()
        )
        x = torch.randn(2, 12, 6)
        assert plain(x).shape == timed(x).shape == (2, 12, 3)
    with pytest.raises(ValueError):
        model_for("nonsense", config, graph)


def make_prepared(frame, config, timed):
    from tracast.data import missing_report

    parts = split(frame, config.history, config.horizon)
    scaler = Scaler().fit(parts["train"])
    datasets = {
        name: Windows(part, scaler, config.history, config.horizon, time_features=timed)
        for name, part in parts.items()
    }
    protocol = {
        "graph": {},
        "missing": {name: missing_report(part) for name, part in parts.items()},
        "scaler": {},
        "splits": {},
    }
    return datasets, scaler, np.zeros((2, 2), dtype=np.float32), protocol


def run_trial(frame, tmp_path, monkeypatch, name, **changes):
    import tracast.benchmark as benchmark

    settings = {"epochs": 4, **changes}
    config = replace(
        benchmark.BenchmarkConfig(),
        sensors=2,
        rows=len(frame),
        hidden=4,
        embedding=2,
        threads=1,
        # Persistence is always run by the driver, never listed as a model.
        models=["gru" if name == "persistence" else name],
        **settings,
    )
    monkeypatch.setattr(
        benchmark, "prepared", lambda *args: make_prepared(frame, config, args[-1])
    )
    return benchmark.trial(config, "unused", "unused", name, 7, tmp_path / name)


@pytest.mark.parametrize("name", ["persistence", "ha", "linear", "gru_time", "lstm"])
def test_trials_report_the_same_fields_for_every_model(
    frame, tmp_path, monkeypatch, name
):
    result = run_trial(frame, tmp_path, monkeypatch, name)
    assert result["model"] == name
    assert result["test"]["mae"] > 0
    assert sum(result["daily"]["count"]) == sum(
        row["count"] for row in result["test"]["per_horizon"]
    )
    assert sum(result["daily"]["abs"]) / sum(result["daily"]["count"]) == pytest.approx(
        result["test"]["mae"]
    )
    assert result["inference"]["median_ms_per_window"] >= 0
    learned = name not in ("persistence", "ha")
    assert bool(result["history"]) == learned
    assert (result["parameter_count"] > 0) == learned
    status = json.loads((tmp_path / name / "status.json").read_text())
    assert status["status"] == "complete"


def test_historical_average_beats_persistence_on_a_periodic_series(
    tmp_path, monkeypatch
):
    index = pd.date_range("2012-03-05", periods=7 * 288 * 4, freq="5min")
    slot = week_slots(index)
    frame = pd.DataFrame(
        {
            "a": 40 + 25 * np.sin(2 * np.pi * slot / 288),
            "b": 55 + 20 * np.cos(2 * np.pi * slot / 288),
        },
        index=index,
    ).astype("float32")
    ha = run_trial(frame, tmp_path / "ha", monkeypatch, "ha")
    persistence = run_trial(frame, tmp_path / "p", monkeypatch, "persistence")
    assert ha["test"]["mae"] < 0.01 < persistence["test"]["mae"]


def test_early_stopping_and_plateau_scheduler(frame, tmp_path, monkeypatch):
    import tracast.benchmark as benchmark

    seen = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    real = benchmark.evaluate

    def worsening(model, loader, mean, std, horizon, daily=None):
        result = real(model, loader, mean, std, horizon, daily)
        if daily is None and model is not None:
            result = {**result, "mae": next(seen)}
        return result

    monkeypatch.setattr(benchmark, "evaluate", worsening)
    result = run_trial(
        frame,
        tmp_path,
        monkeypatch,
        "gru",
        epochs=6,
        patience=3,
        scheduler="plateau",
        scheduler_patience=1,
    )
    # Epoch 1 is best; epochs 2-4 fail to improve, so training stops at 4.
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3, 4]
    assert result["best_epoch"] == 1
    assert result["stopped_early"] is True
    # The rate halves after the second consecutive miss and is recorded as the
    # rate each epoch trained with.
    lrs = [row["lr"] for row in result["history"]]
    assert lrs[:3] == pytest.approx([0.001] * 3) and lrs[3] == pytest.approx(0.0005)


def test_default_training_runs_every_epoch(frame, tmp_path, monkeypatch):
    result = run_trial(frame, tmp_path, monkeypatch, "gru", epochs=3)
    assert len(result["history"]) == 3
    assert result["stopped_early"] is False
    assert len({row["lr"] for row in result["history"]}) == 1


def test_aggregate_adds_day_bootstrap_when_daily_records_exist(
    frame, tmp_path, monkeypatch
):
    import tracast.benchmark as benchmark

    results = []
    for name in ("gru", "linear", "ha", "persistence"):
        for seed_dir, seed in (("a", 7),):
            result = run_trial(frame, tmp_path / seed_dir, monkeypatch, name)
            results.append({**result, "seed": seed})
    summary = benchmark.aggregate(results)
    comparisons = summary["day_bootstrap"]
    assert "linear_vs_gru" in comparisons and "gru_vs_ha" in comparisons
    for value in comparisons.values():
        assert value["ci95"][0] <= value["estimate"] <= value["ci95"][1]
    assert "models" in summary and "paired_mae" in summary
    legacy = [{k: v for k, v in r.items() if k != "daily"} for r in results]
    assert "day_bootstrap" not in benchmark.aggregate(legacy)


def fake_result(model, seed, shift):
    learned = model not in ("ha", "persistence")
    row = {"minutes": 5, "mae": 3.0 + shift, "rmse": 6.0}
    return {
        "model": model,
        "seed": seed,
        "test": {"mae": 3.0 + shift, "rmse": 6.0, "per_horizon": [row]},
        "daily": {
            "days": [f"d{i}" for i in range(4)],
            "abs": [300 + 100 * shift * (i + 1) for i in range(4)],
            "sq": [900.0] * 4,
            "count": [100] * 4,
        },
        # Runs of different lengths, as early stopping produces.
        "history": [
            {"train_normalized_mae": 0.5, "validation": {"mae": 3.0}}
            for _ in range(seed + 1)
        ]
        if learned
        else [],
        "best_epoch": 1 if learned else None,
        "parameter_count": 10 if learned else 0,
        "train_seconds": 1.0,
        "inference": {"median_ms_per_window": 0.1},
        "peak_memory": {"bytes": 1024**2},
    }


def test_aggregate_covers_graph_effect_with_and_without_calendar():
    import tracast.benchmark as benchmark

    shifts = {
        "persistence": 1.0,
        "ha": 0.6,
        "gru": 0.3,
        "gru_time": 0.0,
        "graph_gru": 0.25,
        "graph_self": 0.26,
        "graph_gru_time": -0.02,
        "graph_self_time": -0.01,
    }
    results = [
        fake_result(model, seed, shift + 0.001 * seed)
        for model, shift in shifts.items()
        for seed in ((1, 2) if model not in ("ha", "persistence") else (1,))
    ]
    comparisons = benchmark.aggregate(results)["day_bootstrap"]
    for key in (
        "gru_time_vs_gru",
        "graph_gru_vs_graph_self",
        "graph_gru_time_vs_graph_self_time",
        "graph_gru_time_vs_gru_time",
        "graph_gru_time_vs_graph_gru",
        "gru_vs_ha",
        "ha_vs_persistence",
    ):
        assert key in comparisons, key
    assert comparisons["gru_time_vs_gru"]["estimate"] < 0
    assert comparisons["graph_gru_time_vs_graph_self_time"]["estimate"] < 0


def test_convergence_plots_accept_runs_of_different_lengths(tmp_path):
    import tracast.benchmark as benchmark

    results = [
        fake_result(model, seed, shift)
        for model, shift in (("persistence", 1.0), ("gru", 0.3), ("gru_time", 0.0))
        for seed in ((1,) if model == "persistence" else (1, 2, 3))
    ]
    protocol = {
        "graph": {
            "sensor_ids": ["a", "b"],
            "adjacency": [[0.0, 1.0], [0.0, 0.0]],
            "locations": {
                "a": {"latitude": 34.0, "longitude": -118.0},
                "b": {"latitude": 34.1, "longitude": -118.1},
            },
        }
    }
    benchmark.plots(tmp_path, results, benchmark.aggregate(results), protocol)
    for name in ("horizons.png", "convergence.png", "topology.png"):
        assert (tmp_path / name).stat().st_size > 0
