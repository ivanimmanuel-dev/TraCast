"""End to end: nothing from the held-out rows reaches training or model selection."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import tracast.benchmark as benchmark
from tracast.baselines import HistoricalAverage
from tracast.data import Scaler, Windows, calendar_features, split


def series(seed=0, rows=2400):
    index = pd.date_range("2012-03-05", periods=rows, freq="5min")
    rng = np.random.default_rng(seed)
    base = 50 + 12 * np.sin(2 * np.pi * np.arange(rows) / 288)
    values = base[:, None] + rng.normal(0, 3, (rows, 3)) + np.arange(3)
    values[rng.random(values.shape) < 0.03] = 0  # some missing readings
    return pd.DataFrame(values.astype("float32"), index=index, columns=list("abc"))


def make_prepared(frame, config, timed):
    parts = split(frame, config.history, config.horizon)
    scaler = Scaler().fit(parts["train"])
    datasets = {
        name: Windows(part, scaler, config.history, config.horizon, timed)
        for name, part in parts.items()
    }
    return datasets, scaler, np.zeros((3, 3), dtype=np.float32), {"graph": {}}


def run(frame, tmp_path, monkeypatch, name):
    config = replace(
        benchmark.BenchmarkConfig(),
        sensors=3,
        rows=len(frame),
        hidden=4,
        epochs=3,
        threads=1,
        models=[name],
        scheduler="none",
    )
    monkeypatch.setattr(
        benchmark, "prepared", lambda *args: make_prepared(frame, config, args[-1])
    )
    return benchmark.trial(config, "unused", "unused", name, 7, tmp_path)


@pytest.mark.parametrize("name", ["gru_time", "sgraph_time", "linear"])
def test_rewriting_the_test_period_changes_the_score_but_not_the_trained_model(
    name, tmp_path, monkeypatch
):
    original = series()
    test_start = int(len(original) * 0.8)
    altered = original.copy()
    altered.iloc[test_start:] = altered.iloc[test_start:] * 3 + 7
    first = run(original, tmp_path / "a", monkeypatch, name)
    second = run(altered, tmp_path / "b", monkeypatch, name)
    assert first["state_sha256"] == second["state_sha256"]  # same weights, bit for bit
    assert [h["validation"] for h in first["history"]] == [
        h["validation"] for h in second["history"]
    ]
    assert first["best_epoch"] == second["best_epoch"]
    assert first["test"]["mae"] != second["test"]["mae"]  # the test data did change


def test_rewriting_validation_and_test_leaves_everything_fitted_on_train_alone():
    original = series()
    n = len(original)
    altered = original.copy()
    altered.iloc[int(n * 0.7) :] = 99.0
    a, b = split(original), split(altered)
    sa, sb = Scaler().fit(a["train"]), Scaler().fit(b["train"])
    np.testing.assert_array_equal(sa.mean, sb.mean)
    np.testing.assert_array_equal(sa.std, sb.std)
    wa, wb = (
        Windows(a["train"], sa, time_features=True),
        Windows(b["train"], sb, time_features=True),
    )
    np.testing.assert_array_equal(wa.x, wb.x)
    np.testing.assert_array_equal(wa.y, wb.y)
    ha = [HistoricalAverage().fit(w.times, w.y, w.mask).table for w in (wa, wb)]
    np.testing.assert_array_equal(ha[0], ha[1])


def test_calendar_features_depend_on_the_clock_only():
    index = pd.date_range("2012-03-05", periods=500, freq="5min")
    np.testing.assert_array_equal(
        calendar_features(index), calendar_features(index.copy())
    )
    assert calendar_features(index[:100]).shape == (100, 3)
    # a feature at time t must not change when later timestamps are appended
    np.testing.assert_array_equal(
        calendar_features(index)[:100], calendar_features(index[:100])
    )


def test_the_calibration_partition_never_reaches_training_or_checkpoint_choice(
    tmp_path, monkeypatch
):
    original = series(rows=3000)
    n = len(original)
    altered = original.copy()
    cal_start, cal_stop = int(n * 0.75), int(n * 0.8)
    altered.iloc[cal_start:cal_stop] = altered.iloc[cal_start:cal_stop] * 5

    def run_with_calibration(frame, folder):
        config = replace(
            benchmark.BenchmarkConfig(),
            sensors=3,
            rows=n,
            hidden=4,
            epochs=3,
            threads=1,
            models=["gru_time"],
            scheduler="none",
        )

        def prepared(*args):
            parts = split(frame, config.history, config.horizon, 0.05)
            scaler = Scaler().fit(parts["train"])
            datasets = {
                k: Windows(v, scaler, config.history, config.horizon, True)
                for k, v in parts.items()
                if k != "cal"
            }
            return datasets, scaler, np.zeros((3, 3), dtype=np.float32), {"graph": {}}

        monkeypatch.setattr(benchmark, "prepared", prepared)
        return benchmark.trial(config, "unused", "unused", "gru_time", 7, folder)

    first = run_with_calibration(original, tmp_path / "a")
    second = run_with_calibration(altered, tmp_path / "b")
    assert first["state_sha256"] == second["state_sha256"]
    assert first["best_epoch"] == second["best_epoch"]
