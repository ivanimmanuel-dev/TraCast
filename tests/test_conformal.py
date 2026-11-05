"""Conformal intervals on constructed noise, where the right answer is known."""

import numpy as np
import pytest

from tracast.conformal import (
    abs_residuals,
    adaptive_intervals,
    conformal_quantile,
    cqr_intervals,
    cqr_scores,
    evaluate_intervals,
    pooled_quantiles,
    rolling_intervals,
    sensor_quantiles,
    symmetric,
)


def test_the_quantile_is_the_ceil_n_plus_one_rank():
    scores = np.arange(1, 11)
    assert conformal_quantile(scores, 0.1) == 10  # ceil(11 * 0.9) = 10
    assert conformal_quantile(scores, 0.5) == 6  # ceil(11 * 0.5) = 6
    assert conformal_quantile(scores[::-1], 0.5) == 6
    assert conformal_quantile(scores[:3], 0.05) == np.inf  # too few points
    assert conformal_quantile([], 0.1) == np.inf


def draw(rng, windows, scale):
    """Targets around a zero forecast, scale (horizon, sensors)."""
    horizons, sensors = scale.shape
    return rng.normal(0, 1, (windows, horizons, sensors)) * scale


def test_split_conformal_covers_fresh_exchangeable_data():
    rng = np.random.default_rng(0)
    scale = np.tile(np.linspace(1, 3, 6)[:, None], (1, 4))
    cal, test = draw(rng, 3000, scale), draw(rng, 3000, scale)
    mask = np.ones(cal.shape, dtype=bool)
    q = pooled_quantiles(abs_residuals(0 * cal, cal), mask, 0.1)
    low, high = symmetric(0 * test, q)
    result = evaluate_intervals(low, high, test, mask, 0.1)
    assert result["overall"]["coverage"] == pytest.approx(0.9, abs=0.015)
    # Each horizon has its own noise level, and each gets about 90%.
    assert all(abs(h["coverage"] - 0.9) < 0.03 for h in result["per_horizon"])
    assert result["per_horizon"][-1]["width"] > 2 * result["per_horizon"][0]["width"]


def test_per_sensor_quantiles_fix_uneven_noise_and_fall_back_when_sparse():
    rng = np.random.default_rng(1)
    scale = np.array([[0.5, 1.0, 4.0]])
    cal, test = draw(rng, 4000, scale), draw(rng, 4000, scale)
    mask = np.ones(cal.shape, dtype=bool)
    residuals = abs_residuals(0 * cal, cal)
    pooled = evaluate_intervals(
        *symmetric(0 * test, pooled_quantiles(residuals, mask, 0.1)), test, mask, 0.1
    )
    per_sensor = evaluate_intervals(
        *symmetric(0 * test, sensor_quantiles(residuals, mask, 0.1)), test, mask, 0.1
    )
    assert pooled["sensor_coverage"]["min"] < 0.8  # the noisy sensor is under-covered
    assert per_sensor["sensor_coverage"]["min"] > 0.87
    sparse = mask.copy()
    sparse[:, :, 2] = False
    sparse[:10, :, 2] = True  # only 10 calibration points for sensor 2
    quantile = sensor_quantiles(residuals, sparse, 0.1, minimum=50)
    assert quantile[0, 2] == pooled_quantiles(residuals, sparse, 0.1)[0]


def test_masked_targets_are_not_calibration_points_or_scores():
    residuals = np.ones((20, 1, 1))
    residuals[:5] = 100.0
    mask = np.ones((20, 1, 1), dtype=bool)
    mask[:5] = False
    assert pooled_quantiles(residuals, mask, 0.2)[0] == 1.0
    target = np.zeros((20, 1, 1))
    target[:5] = 500.0  # missing readings stored as junk
    result = evaluate_intervals(target - 1, target + 1, target, mask, 0.1)
    assert result["overall"]["targets"] == 15 and result["overall"]["coverage"] == 1.0


def test_cqr_corrects_a_band_that_is_too_narrow_and_negative_scores_mean_inside():
    rng = np.random.default_rng(2)
    scale = np.full((2, 3), 2.0)
    cal, test = draw(rng, 3000, scale), draw(rng, 3000, scale)
    mask = np.ones(cal.shape, dtype=bool)
    narrow = lambda y: (np.full_like(y, -0.5), np.full_like(y, 0.5))  # noqa: E731
    scores = cqr_scores(*narrow(cal), cal)
    assert scores[np.abs(cal) < 0.5].max() < 0
    q = pooled_quantiles(scores, mask, 0.1)
    raw = evaluate_intervals(*narrow(test), test, mask, 0.1)
    fixed = evaluate_intervals(*cqr_intervals(*narrow(test), q), test, mask, 0.1)
    assert raw["overall"]["coverage"] < 0.3
    assert fixed["overall"]["coverage"] == pytest.approx(0.9, abs=0.02)


def shifted_stream(rng, windows=4000, jump=3.0):
    scale = np.ones((3, 2))
    first = draw(rng, windows // 2, scale)
    second = draw(rng, windows - windows // 2, scale * jump)
    return np.concatenate([first, second])


def test_adaptive_intervals_follow_a_jump_in_the_noise_that_split_conformal_misses():
    rng = np.random.default_rng(3)
    cal = draw(rng, 3000, np.ones((3, 2)))
    test = shifted_stream(rng)
    mask = np.ones(test.shape, dtype=bool)
    cal_mask = np.ones(cal.shape, dtype=bool)
    residuals = abs_residuals(0 * cal, cal)
    static = evaluate_intervals(
        *symmetric(0 * test, pooled_quantiles(residuals, cal_mask, 0.1)),
        test,
        mask,
        0.1,
    )
    low, high = adaptive_intervals(
        0 * test, test, mask, residuals, cal_mask, 0.1, gamma=0.02
    )
    adaptive = evaluate_intervals(low, high, test, mask, 0.1)
    assert static["overall"]["coverage"] < 0.75
    assert adaptive["overall"]["coverage"] > 0.84
    assert adaptive["overall"]["coverage"] > static["overall"]["coverage"] + 0.1


def corrupted(target, start):
    changed = target.copy()
    changed[start:] += 1e3
    return changed


@pytest.mark.parametrize("h", [0, 4, 11])
def test_adaptive_intervals_never_peek_at_outcomes_that_have_not_arrived(h):
    rng = np.random.default_rng(4)
    cal = draw(rng, 500, np.ones((12, 2)))
    test = draw(rng, 400, np.ones((12, 2)))
    mask = np.ones(test.shape, dtype=bool)
    cal_mask = np.ones(cal.shape, dtype=bool)
    residuals = abs_residuals(0 * cal, cal)
    start = 150
    original = adaptive_intervals(0 * test, test, mask, residuals, cal_mask, 0.1)
    other = adaptive_intervals(
        0 * test, corrupted(test, start), mask, residuals, cal_mask, 0.1
    )
    # The step-h target of window j arrives before window j + 1 + h is forecast,
    # so windows up to start + h cannot have seen the change.
    np.testing.assert_array_equal(
        original[0][: start + h + 1, h], other[0][: start + h + 1, h]
    )
    assert not np.array_equal(original[0][:, h], other[0][:, h])


@pytest.mark.parametrize("h", [0, 5, 11])
def test_rolling_intervals_only_use_known_residuals(h):
    rng = np.random.default_rng(5)
    steps = 40
    cal = draw(rng, 600, np.ones((12, 2)))
    test = draw(rng, 480, np.ones((12, 2)))
    days = np.arange(480) // steps
    mask = np.ones(test.shape, dtype=bool)
    cal_mask = np.ones(cal.shape, dtype=bool)
    residuals = abs_residuals(0 * cal, cal)
    kwargs = {"span_days": 3, "steps_per_day": steps}
    original = rolling_intervals(
        0 * test, test, mask, days, residuals, cal_mask, 0.1, **kwargs
    )
    start = 200
    other = rolling_intervals(
        0 * test, corrupted(test, start), mask, days, residuals, cal_mask, 0.1, **kwargs
    )
    # A day starting at window i0 uses windows below i0 - h, so days starting at or
    # before start + h are untouched.
    untouched = [d for d in range(12) if d * steps <= start + h]
    rows = np.isin(days, untouched)
    np.testing.assert_array_equal(original[0][rows, h], other[0][rows, h])
    assert not np.array_equal(original[0][:, h], other[0][:, h])


def test_evaluate_intervals_by_hand():
    target = np.array([[[0.0, 5.0]], [[1.0, 2.0]]])  # (windows=2, horizon=1, sensors=2)
    low, high = np.full(target.shape, -1.0), np.full(target.shape, 1.0)
    mask = np.array([[[True, True]], [[True, False]]])
    groups = {"first_sensor": np.broadcast_to(np.array([True, False]), target.shape)}
    result = evaluate_intervals(
        low, high, target, mask, 0.1, groups=groups, days=np.array([0, 1])
    )
    # Observed targets: 0 (inside), 5 (misses by 4), 1 (inside, on the edge).
    assert result["overall"]["coverage"] == pytest.approx(2 / 3)
    assert result["overall"]["width"] == pytest.approx(2.0)
    assert result["overall"]["interval_score"] == pytest.approx(2.0 + (2 / 0.1) * 4 / 3)
    assert result["first_sensor"]["coverage"] == 1.0
    assert [d["targets"] for d in result["per_day"]] == [2, 1]
    assert result["sensor_coverage"]["min"] == 0.0


def test_calibration_split_keeps_train_and_test_and_never_overlaps():
    import pandas as pd

    from tracast.data import split

    frame = pd.DataFrame(
        {"a": np.arange(2000, dtype=float) + 1},
        index=pd.date_range("2012-03-01", periods=2000, freq="5min"),
    )
    plain = split(frame)
    parts = split(frame, calibration=0.05)
    assert list(parts) == ["train", "val", "cal", "test"]
    assert parts["train"].equals(plain["train"]) and parts["test"].equals(plain["test"])
    # validation and calibration together are the old validation block, in order
    assert pd.concat([parts["val"], parts["cal"]]).equals(plain["val"])
    assert parts["val"].index[-1] < parts["cal"].index[0] < parts["test"].index[0]
    assert len(parts["cal"]) == int(2000 * 0.8) - int(2000 * 0.75)
    for bad in (-0.01, 0.1, 0.5):
        with pytest.raises(ValueError):
            split(frame, calibration=bad)
