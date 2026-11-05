"""Conformal prediction intervals for multi-horizon forecasts.

Everything here works on arrays shaped (windows, horizon, sensors): point forecasts,
targets and an observed mask. Zero and missing targets never count, neither as
calibration points nor in the scores.

Split conformal needs data the model never trained or selected on. That is what the
"cal" partition is for (see data.split). The guarantee, coverage of at least
1 - alpha, holds when calibration and test windows are exchangeable. Traffic is not:
neighbouring windows overlap, the daily pattern repeats, and the test period can
drift away from calibration. So the coverage here is what you measure, not what
you are promised, and the online methods exist to react to drift.
"""

import numpy as np


def conformal_quantile(scores, alpha):
    """The finite-sample (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th smallest score.

    Returns infinity when there are too few scores to reach the level.
    """
    scores = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return np.inf
    return float(scores[k - 1])


def abs_residuals(prediction, target):
    return np.abs(
        np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    )


def pooled_quantiles(residuals, mask, alpha):
    """One quantile per horizon, pooling every window and sensor."""
    return np.array(
        [
            conformal_quantile(residuals[:, h][mask[:, h]], alpha)
            for h in range(residuals.shape[1])
        ]
    )


def sensor_quantiles(residuals, mask, alpha, minimum=50):
    """One quantile per (horizon, sensor), also called Mondrian conformal.

    A sensor with fewer than `minimum` calibration points falls back to the pooled
    quantile for its horizon.
    """
    horizons, sensors = residuals.shape[1], residuals.shape[2]
    pooled = pooled_quantiles(residuals, mask, alpha)
    out = np.empty((horizons, sensors))
    for h in range(horizons):
        for s in range(sensors):
            keep = mask[:, h, s]
            out[h, s] = (
                conformal_quantile(residuals[:, h, s][keep], alpha)
                if keep.sum() >= minimum
                else pooled[h]
            )
    return out


def symmetric(prediction, quantile):
    """prediction +- quantile, where quantile is (horizon,) or (horizon, sensors)."""
    q = np.asarray(quantile)
    q = q[None, :, None] if q.ndim == 1 else q[None]
    return prediction - q, prediction + q


def cqr_scores(lower, upper, target):
    """How far the target fell outside the quantile band (negative when inside),
    as in conformalized quantile regression."""
    return np.maximum(lower - target, target - upper)


def cqr_intervals(lower, upper, quantile):
    q = np.asarray(quantile)
    q = q[None, :, None] if q.ndim == 1 else q[None]
    return lower - q, upper + q


def adaptive_intervals(
    prediction, target, mask, cal_residuals, cal_mask, alpha, gamma=0.005
):
    """Adaptive conformal inference (Gibbs and Candes, 2021), one level per horizon.

    The half-width is the (1 - a) quantile of the fixed calibration residuals. After
    each window the level moves, a <- a + gamma * (alpha - miss), where miss is the
    share of that window's observed targets outside the interval. A forecast of
    horizon step k is made when the input ends, but its target only arrives k + 1
    steps later, so at window i the newest feedback is from window i - 1 - k.
    Nothing later is ever read.

    Returns lower and upper arrays shaped like prediction.
    """
    windows, horizons, _ = prediction.shape
    lower = np.empty(prediction.shape, dtype=np.float64)
    upper = np.empty(prediction.shape, dtype=np.float64)
    for h in range(horizons):
        pool = np.sort(cal_residuals[:, h][cal_mask[:, h]])
        n = len(pool)
        level = alpha
        miss = np.full(windows, np.nan)
        for i in range(windows):
            feedback = i - 1 - h
            if feedback >= 0 and not np.isnan(miss[feedback]):
                level += gamma * (alpha - miss[feedback])
            rank = int(np.ceil((n + 1) * (1 - min(max(level, 1e-4), 1 - 1e-4))))
            width = pool[min(max(rank, 1), n) - 1] if n else np.inf
            lower[i, h] = prediction[i, h] - width
            upper[i, h] = prediction[i, h] + width
            seen = mask[i, h]
            if seen.any():
                outside = (target[i, h][seen] < lower[i, h][seen]) | (
                    target[i, h][seen] > upper[i, h][seen]
                )
                miss[i] = outside.mean()
    return lower, upper


def rolling_intervals(
    prediction,
    target,
    mask,
    days,
    cal_residuals,
    cal_mask,
    alpha,
    span_days=7,
    steps_per_day=288,
):
    """Recalibrate at the start of every day from residuals that are already known.

    `days` is each window's day code (0, 1, 2, ...). At the first window i0 of a day,
    horizon step k may use residuals of windows up to i0 - 1 - k (their step-k target
    has arrived), taking the most recent `span_days` days of them. If there are fewer
    than that, the calibration residuals fill in, so day 0 is plain split conformal.
    """
    horizons = prediction.shape[1]
    residuals = abs_residuals(prediction, target)
    lower = np.empty(prediction.shape, dtype=np.float64)
    upper = np.empty(prediction.shape, dtype=np.float64)
    span = span_days * steps_per_day
    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        start, stop = int(rows[0]), int(rows[-1]) + 1
        for h in range(horizons):
            known = max(start - h, 0)  # windows 0..known-1 have their step-h target
            low = max(0, known - span)
            parts = [residuals[low:known, h][mask[low:known, h]]]
            if known - low < span:
                parts.append(cal_residuals[:, h][cal_mask[:, h]])
            width = conformal_quantile(np.concatenate(parts), alpha)
            lower[start:stop, h] = prediction[start:stop, h] - width
            upper[start:stop, h] = prediction[start:stop, h] + width
    return lower, upper


def evaluate_intervals(lower, upper, target, mask, alpha, groups=None, days=None):
    """Coverage, width and interval score over observed targets.

    groups maps a name to a boolean array shaped like target (which targets to
    include); days is an integer code per window for day-by-day coverage. The
    interval score (Gneiting and Raftery) is the width plus 2 / alpha times how far
    a target missed, so narrow intervals only win if they hold.
    """
    width = upper - lower
    miss_low = np.maximum(lower - target, 0.0)
    miss_high = np.maximum(target - upper, 0.0)
    covered = (target >= lower) & (target <= upper)
    score = width + (2 / alpha) * (miss_low + miss_high)

    def summarize(keep):
        keep = keep & mask
        if not keep.any():
            return None
        return {
            "coverage": float(covered[keep].mean()),
            "width": float(width[keep].mean()),
            "interval_score": float(score[keep].mean()),
            "targets": int(keep.sum()),
        }

    everything = np.ones(mask.shape, dtype=bool)
    result = {"overall": summarize(everything)}
    result["per_horizon"] = []
    for h in range(target.shape[1]):
        keep = np.zeros(mask.shape, dtype=bool)
        keep[:, h] = True
        result["per_horizon"].append(summarize(keep))
    for name, keep in (groups or {}).items():
        result[name] = summarize(keep)
    if days is not None:
        result["per_day"] = [
            summarize(np.broadcast_to((days == d)[:, None, None], mask.shape))
            for d in np.unique(days)
        ]
    sensors = [
        float(covered[:, :, s][mask[:, :, s]].mean())
        for s in range(target.shape[2])
        if mask[:, :, s].any()
    ]
    result["sensor_coverage"] = {
        "min": float(np.min(sensors)),
        "p10": float(np.percentile(sensors, 10)),
        "median": float(np.median(sensors)),
    }
    return result
