"""The interval experiment on a small constructed series."""

import numpy as np
import pandas as pd
import pytest

from tracast import uncertainty
from tracast.data import Scaler, Windows, split
from tracast.uncertainty import (
    UncertaintyConfig,
    average,
    gaussian_intervals,
    groups_for,
    residual_std,
)


def test_average_gives_mean_and_sd_over_matching_trees():
    trees = [
        {"a": 1.0, "b": [2.0, {"c": 10.0}], "name": "x", "gap": None},
        {"a": 3.0, "b": [4.0, {"c": 30.0}], "name": "x", "gap": None},
    ]
    mean, sd = average(trees)
    assert mean["a"] == 2.0 and sd["a"] == pytest.approx(np.sqrt(2))
    assert mean["b"][1]["c"] == 20.0 and mean["name"] == "x" and mean["gap"] is None
    assert average([{"a": 5.0}])[1]["a"] is None  # one seed has no spread


def test_residual_std_uses_observed_targets_only():
    residuals = np.array([[[1.0]], [[3.0]], [[100.0]]])
    mask = np.array([[[True]], [[True]], [[False]]])
    assert residual_std(residuals, mask)[0, 0] == pytest.approx(np.sqrt((1 + 9) / 2))


def test_gaussian_intervals_are_centered_and_scale_with_sigma():
    pred = np.zeros((2, 3, 4))
    sigma = np.full((3, 4), 2.0)
    low, high = gaussian_intervals(pred, sigma, 0.05)
    assert np.allclose(high, -low) and np.allclose(high, 1.959964 * 2.0, atol=1e-4)


@pytest.fixture
def series():
    index = pd.date_range("2012-03-05", periods=3000, freq="5min")
    rng = np.random.default_rng(0)
    base = 50 + 10 * np.sin(2 * np.pi * np.arange(3000) / 288)
    values = base[:, None] + rng.normal(0, 3, (3000, 3)) + np.arange(3)
    return pd.DataFrame(values.astype("float32"), index=index, columns=list("abc"))


def make_prepared(frame, config, calibration):
    parts = split(frame, config.history, config.horizon, calibration)
    scaler = Scaler().fit(parts["train"])
    datasets = {
        name: Windows(part, scaler, config.history, config.horizon, True)
        for name, part in parts.items()
    }
    return datasets, scaler, np.zeros((3, 3), dtype=np.float32), {}


def test_groups_only_use_what_is_known_at_forecast_time(series):
    scaler = Scaler().fit(series)
    windows = Windows(series, scaler)
    last = windows.x[11 : 11 + len(windows), :3] * scaler.std + scaler.mean
    target = windows.y[12 : 12 + len(windows)][:, None, :].repeat(12, axis=1)
    test = {"last": last.astype(np.float64), "target": target}
    cfg = UncertaintyConfig(benchmark={}, congestion_mph=50.0)
    groups = groups_for(test, windows, cfg)
    assert np.array_equal(groups["origin_congested"], ~groups["origin_free_flow"])
    assert np.array_equal(groups["rush_hour"], ~groups["off_peak"])
    # shifting future targets must not move the groups that are defined at the origin
    changed = {"last": test["last"], "target": test["target"] + 100.0}
    other = groups_for(changed, windows, cfg)
    for name in ("origin_congested", "origin_free_flow", "rush_hour", "off_peak"):
        assert np.array_equal(groups[name], other[name])
    assert not np.array_equal(
        groups["target_congested_diagnostic"], other["target_congested_diagnostic"]
    )


def test_a_full_seed_runs_every_method_and_keeps_calibration_out_of_training(
    series, tmp_path, monkeypatch
):
    cfg = UncertaintyConfig(
        benchmark={
            "sensors": 3,
            "rows": 3000,
            "hidden": 4,
            "epochs": 2,
            "threads": 1,
            "patience": None,
            "scheduler": "none",
        },
        model="gru_time",
        seeds=[7],
        reliability_alphas=[0.1, 0.3],
        rolling_days=1,
    )
    seen = {}

    def fake_prepared(config, data, graph, time_features, calibration):
        seen["calibration"] = calibration
        return make_prepared(series, config, calibration)

    monkeypatch.setattr(uncertainty, "prepared", fake_prepared)
    result = uncertainty.run_seed(cfg, "unused", "unused", 7, tmp_path / "seed")
    assert seen["calibration"] == 0.05
    expected = set(uncertainty.POINT_METHODS + uncertainty.QUANTILE_METHODS)
    assert set(result["methods"]) == expected
    for name, summary in result["methods"].items():
        assert (
            0 <= summary["overall"]["coverage"] <= 1 and summary["overall"]["width"] > 0
        )
        assert len(summary["per_horizon"]) == 12
        assert summary["overall"]["targets"] == result["test_targets"]
    # quantile methods only exist at the primary level
    assert set(result["reliability"]["cqr"]) == {"0.1"}
    assert set(result["reliability"]["conformal_pooled"]) == {"0.1", "0.3"}
    # a higher nominal level gives wider intervals
    pooled = result["reliability"]["conformal_pooled"]
    assert pooled["0.1"]["width"] > pooled["0.3"]["width"]
    assert result["calibration_windows"] == int(3000 * 0.8) - int(3000 * 0.75) - 23
