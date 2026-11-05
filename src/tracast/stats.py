"""Bootstrap over test days.

Seed SDs only show how much training varies. These helpers resample whole test
days (the same days for both models in a pair) to see how much a score, or a
gap between two models, depends on which dates were held out.
"""

import numpy as np


def _daily(records):
    """Average per-day error sums over the runs of one model (usually seeds)."""
    first = records[0]
    if any(r["days"] != first["days"] for r in records):
        raise ValueError("Daily records cover different days")
    if any(r["count"] != first["count"] for r in records):
        raise ValueError("Daily records use different observed targets")
    return {
        "days": first["days"],
        "count": np.asarray(first["count"], dtype=np.float64),
        "abs": np.mean([r["abs"] for r in records], axis=0),
    }


def day_bootstrap(candidate, reference=None, resamples=10000, seed=0):
    """Interval for a model's MAE, or for candidate minus reference MAE.

    Negative gaps favor the candidate. Both arguments are lists of per-run daily
    records.
    """
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    first = _daily(candidate)
    second = _daily(reference) if reference is not None else None
    if second is not None and (
        first["days"] != second["days"]
        or not np.array_equal(first["count"], second["count"])
    ):
        raise ValueError("Paired models must be scored on identical windows")
    days = len(first["days"])
    if days < 2:
        raise ValueError("Bootstrapping needs at least two test days")
    draws = np.random.default_rng(seed).integers(0, days, size=(resamples, days))
    count = first["count"][draws].sum(axis=1)
    if (count == 0).any():
        raise ValueError("A resample contained no observed targets")
    estimate = first["abs"][draws].sum(axis=1) / count
    point = first["abs"].sum() / first["count"].sum()
    if second is not None:
        estimate = estimate - second["abs"][draws].sum(axis=1) / count
        point -= second["abs"].sum() / second["count"].sum()
    low, high = np.percentile(estimate, [2.5, 97.5])
    return {
        "estimate": float(point),
        "ci95": [float(low), float(high)],
        "days": days,
        "resamples": resamples,
        "resampled_unit": "test day (forecast origin); training runs averaged first",
        "share_below_zero": float((estimate < 0).mean()),
    }
