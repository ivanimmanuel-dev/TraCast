"""The replay never uses an outcome that has not arrived, and the pieces add up."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from tracast import shift
from tracast.data import Scaler, Windows
from tracast.model import GRU


@pytest.fixture
def series():
    index = pd.date_range("2012-03-05", periods=1900, freq="5min")
    rng = np.random.default_rng(0)
    base = 50 + 10 * np.sin(2 * np.pi * np.arange(1900) / 288)
    values = base[:, None] + rng.normal(0, 3, (1900, 3)) + np.arange(3)
    return pd.DataFrame(values.astype("float32"), index=index, columns=list("abc"))


def test_a_window_is_known_exactly_when_its_last_target_row_has_been_seen(series):
    windows = Windows(series, Scaler().fit(series))
    for first in (50, 288, 700):
        known = shift.known_windows(first)
        last_input_row = first + windows.history - 1
        last_target = lambda j: j + windows.history + windows.horizon - 1  # noqa: E731
        assert last_target(known) <= last_input_row
        assert last_target(known + 1) > last_input_row


def test_day_ranges_cover_every_window_once_in_order(series):
    windows = Windows(series, Scaler().fit(series))
    idx = np.arange(100, 1000)
    ranges = shift.day_ranges(windows, idx)
    assert ranges[0][0] == 100 and ranges[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert len(ranges) == 4  # 900 windows across calendar days of 288 rows


def test_residual_average_removes_a_constant_error_using_only_arrived_outcomes():
    rng = np.random.default_rng(1)
    target = rng.normal(50, 1, (400, 4, 2))
    static = target - 3.0
    mask = np.ones(target.shape, dtype=bool)
    fixed = shift.bias_ema(static, target, mask, 0.1)
    assert np.abs(fixed - target)[200:].mean() < 0.3 * np.abs(static - target).mean()
    start = 150
    changed = target.copy()
    changed[start:] += 100.0
    other = shift.bias_ema(static, changed, mask, 0.1)
    for k in range(4):
        # step k of window i may only use windows up to i - 1 - k
        np.testing.assert_array_equal(
            fixed[: start + k + 1, k], other[: start + k + 1, k]
        )
    assert not np.array_equal(fixed[:, 0], other[:, 0])


def test_residual_average_skips_missing_targets():
    target = np.full((60, 2, 1), 50.0)
    static = target - 4.0
    mask = np.ones(target.shape, dtype=bool)
    mask[:30] = False
    fixed = shift.bias_ema(static, target, mask, 0.5)
    assert np.allclose(fixed[:30], static[:30])  # nothing to learn from yet
    assert np.abs(fixed[-1] - 50.0).max() < 0.1


class Config:
    head_days = 1
    head_epochs = 1
    retrain_every = 1
    retrain_epochs = 1
    retrain_lr = 0.001


@pytest.mark.parametrize("kind", ["head_daily", "retrain_weekly"])
def test_model_updates_never_train_on_targets_that_have_not_arrived(series, kind):
    torch.manual_seed(0)
    windows = Windows(series, Scaler().fit(series), time_features=True)
    scaler = Scaler().fit(series)
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    model = GRU(3, 4, 12, extra=3)
    idx = np.arange(700, 1800)
    ranges = shift.day_ranges(windows, idx)
    cutoff = 1000  # rows from here on are "the future" for early days
    broken = copy.copy(windows)
    broken.y = windows.y.copy()
    broken.y[cutoff:] += 1e3

    def run(data):
        return shift.replay(model, data, idx, mean, std, Config, kind, 5, lr=0.01)[0]

    clean, corrupted = run(windows), run(broken)
    safe = [(lo, hi) for lo, hi in ranges if lo + windows.history - 1 < cutoff]
    assert safe, "the test needs days that start before the corrupted rows"
    for lo, hi in safe:
        np.testing.assert_array_equal(
            clean[lo - 700 : hi - 700], corrupted[lo - 700 : hi - 700]
        )
    later = ranges[-1]
    assert not np.array_equal(
        clean[later[0] - 700 : later[1] - 700],
        corrupted[later[0] - 700 : later[1] - 700],
    )


def test_head_only_finetuning_leaves_the_recurrent_weights_alone(series):
    windows = Windows(series, Scaler().fit(series), time_features=True)
    scaler = Scaler().fit(series)
    mean, std = torch.from_numpy(scaler.mean), torch.from_numpy(scaler.std)
    model = GRU(3, 4, 12, extra=3)
    before = copy.deepcopy(model.state_dict())
    shift.finetune(
        model,
        windows,
        np.arange(100, 400),
        mean,
        std,
        1,
        0.01,
        torch.Generator().manual_seed(0),
        head_only=True,
    )
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("gru"))
    assert not torch.equal(before["head.weight"], after["head.weight"])
    assert all(p.requires_grad for p in model.parameters())  # unfrozen again


def test_divergence_is_zero_for_equal_and_one_bit_for_disjoint_histograms():
    assert shift.js_divergence(
        np.array([1.0, 2, 3]), np.array([2.0, 4, 6])
    ) == pytest.approx(0)
    assert shift.js_divergence(np.array([1.0, 0]), np.array([0.0, 1])) == pytest.approx(
        1
    )


def test_score_sums_to_the_overall_error():
    pred = np.array([[[1.0]], [[2.0]], [[3.0]]])
    target = pred + np.array([[[1.0]], [[-2.0]], [[4.0]]])
    mask = np.array([[[True]], [[True]], [[False]]])
    result = shift.score(pred, target, mask, np.array([0, 0, 1]))
    assert result["mae"] == pytest.approx(1.5)
    assert result["daily"] == {"days": [0, 1], "abs": [3.0, 0.0], "count": [2, 0]}


def test_the_whole_experiment_runs_on_a_small_series(series, tmp_path, monkeypatch):
    monkeypatch.setattr(shift, "load", lambda *args: series)
    monkeypatch.setattr(
        shift.topology,
        "load",
        lambda *args: (list(series.columns), np.zeros((3, 3), dtype=np.float32), {}),
    )
    cfg = shift.ShiftConfig(
        benchmark={
            "sensors": 3,
            "rows": 1900,
            "hidden": 4,
            "epochs": 2,
            "threads": 1,
            "patience": None,
            "scheduler": "none",
        },
        seeds=[7, 42],
        settings={"standard": 0.7, "stale": 0.4},
        etas=[0.05, 0.2],
        slot_etas=[0.3],
        head_lrs=[0.001],
        retrain_every=1,
        retrain_epochs=1,
    )
    summary = shift.run(cfg, "unused", "unused", tmp_path)
    assert set(summary["settings"]) == {"standard", "stale"}
    for s in summary["settings"].values():
        assert set(s["strategies"]) == set(shift.STRATEGIES)
        assert s["strategies"]["static"]["adapt_seconds"] == 0
        for kind in ("bias_ema", "bias_slot", "head_daily", "retrain_weekly"):
            low, high = s["strategies"][kind]["vs_static"]["ci95"]
            assert low <= s["strategies"][kind]["vs_static"]["estimate"] <= high
        assert {"least_shifted_third", "most_shifted_third"} <= set(s["regimes"])
    shift.figures(summary, tmp_path)
    for name in ("shift-drift.png", "shift-daily.png", "shift-gain.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_the_replay_refuses_window_sizes_it_was_not_written_for(series, monkeypatch):
    monkeypatch.setattr(shift, "load", lambda *args: series)
    cfg = shift.ShiftConfig(benchmark={"sensors": 3, "rows": 1900, "history": 6})
    with pytest.raises(ValueError, match="12 input steps"):
        shift.prepare(cfg, "unused", "unused", 0.7)


def test_slot_bias_learns_a_daily_pattern_the_blurred_average_cannot():
    steps = 40
    slots = (np.arange(400) % steps)[:, None] + np.zeros((400, 3), dtype=int)
    pattern = np.where(slots % 2 == 0, 4.0, -4.0)[:, :, None]  # alternates every slot
    target = np.full((400, 3, 2), 50.0)
    static = target - pattern
    mask = np.ones(target.shape, dtype=bool)
    learned = shift.bias_slot(static, target, mask, slots, 0.5)
    blurred = shift.bias_ema(static, target, mask, 0.05)
    assert np.abs(learned - target)[200:].mean() < 0.3
    assert np.abs(blurred - target)[200:].mean() > 3.0
    start = 150
    changed = target.copy()
    changed[start:] += 100.0
    other = shift.bias_slot(static, changed, mask, slots, 0.5)
    for k in range(3):
        np.testing.assert_array_equal(
            learned[: start + k + 1, k], other[: start + k + 1, k]
        )
    assert not np.array_equal(learned[:, 0], other[:, 0])


def test_target_slots_follow_the_clock(series):
    windows = Windows(series, Scaler().fit(series))
    slots = shift.target_slots(windows, np.array([0, 1]))
    assert slots.shape == (2, 12)
    assert (
        slots[0, 0] == 12 and slots[0, 1] == 13 and slots[1, 0] == 13
    )  # first target is row 12
    # a weekend target lands in the second block of 288 slots
    index = pd.date_range("2012-03-05", periods=1900, freq="5min")  # starts on a Monday
    weekend_window = int(np.flatnonzero(index.dayofweek >= 5)[0]) - windows.history
    assert shift.target_slots(windows, np.array([weekend_window]))[0, 0] >= 288
