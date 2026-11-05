"""Report pages for the scaling, uncertainty and drift experiments.

Every number in the text is read from the saved summaries, so a page can't drift away
from the results it describes.
"""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(rel):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x, places=0):
    return f"{100 * x:.{places}f}%"


# --- uncertainty ---------------------------------------------------------------------

UNC_LABELS = {
    "gaussian_train": "Gaussian, training residuals",
    "gaussian_cal": "Gaussian, calibration residuals",
    "conformal_pooled": "Split conformal",
    "conformal_sensor": "Split conformal, per sensor",
    "conformal_aci": "Adaptive conformal (ACI)",
    "conformal_rolling": "Rolling conformal (7 days)",
    "quantile_raw": "Quantile network",
    "cqr": "Conformalized quantile (CQR)",
}


def unc_facts(summary):
    m, sd = summary["mean"]["methods"], summary["sd"]["methods"]
    rel = summary["mean"]["reliability"]
    total = m["conformal_pooled"]["overall"]["targets"]
    return m, sd, rel, total


def unc_table(summary):
    m, sd, _, _ = unc_facts(summary)
    rows = []
    for name in UNC_LABELS:
        v = m[name]
        rows.append(
            f"| {UNC_LABELS[name]} | {v['overall']['coverage']:.3f} "
            f"| {v['overall']['width']:.1f} | {v['overall']['interval_score']:.1f} "
            f"| {v['origin_free_flow']['coverage']:.3f} | {v['origin_congested']['coverage']:.3f} "
            f"| {v['rush_hour']['coverage']:.3f} | {v['sensor_coverage']['min']:.3f} |"
        )
    return "\n".join(rows)


def unc_findings(summary):
    """The headline bullets, numbers read from the summary."""
    m, _, rel, total = unc_facts(summary)
    alpha = summary["alpha"]
    pooled, cqr = m["conformal_pooled"], m["cqr"]
    share = pooled["origin_congested"]["targets"] / total
    return [
        f"- **Split conformal hits the coverage it was asked for, on average.** At a "
        f"nominal {pct(1 - alpha)} it covers {pct(pooled['overall']['coverage'], 1)} of the "
        f"test speeds, while the Gaussian baselines overshoot it at "
        f"{pct(m['gaussian_train']['overall']['coverage'], 1)} and "
        f"{pct(m['gaussian_cal']['overall']['coverage'], 1)} (see the reliability plot).",
        f"- **But on average hides congestion.** When the last speed was below 40 mph "
        f"({pct(share)} of the targets) the same intervals cover only "
        f"{pct(pooled['origin_congested']['coverage'])}, against "
        f"{pct(pooled['origin_free_flow']['coverage'])} in free flow.",
        f"- **A quantile network plus a conformal correction fixes that** at the same "
        f"average width ({cqr['overall']['width']:.1f} vs {pooled['overall']['width']:.1f} mph): "
        f"{pct(cqr['origin_congested']['coverage'])} in congestion, and an interval score of "
        f"{cqr['overall']['interval_score']:.1f} against {pooled['overall']['interval_score']:.1f}.",
    ]


UNCERTAINTY = """# Prediction intervals

A forecast of "47 mph" doesn't say how sure the model is. I wanted intervals that hold up: if I ask for {nominal} coverage, about {nominal} of the true speeds should land inside, and that should be true when traffic is bad too, not just on average.

## Setup

The rows are split in time. Training is the first 70% ({train_dates}), validation is the next {val_pct} ({val_dates}) and picks the checkpoint, calibration is the {cal_pct} after that ({cal_dates}), and the last 20% ({test_dates}, {days} days) is the test set. The calibration part is never used for training or for choosing a checkpoint, and a test checks that rewriting it doesn't change the trained model.

The point forecaster is the dense GRU with calendar inputs on the 16-sensor subset, trained with the same recipe as the other experiments (3 seeds). Its test MAE is {mae:.3f} mph (SD {mae_sd:.3f}). Every interval method is built from it:

- **Gaussian, training residuals**: the forecast plus or minus 1.645 standard deviations of the training residuals, per horizon and sensor. Training residuals are optimistic, so this is the naive baseline.
- **Gaussian, calibration residuals**: the same, with the standard deviation measured on the calibration part.
- **Split conformal**: the (1 - alpha) quantile of the absolute calibration residuals, one per horizon (pooled over sensors) or one per horizon and sensor.
- **Adaptive conformal (ACI)**: the same quantile, but the level is nudged after every window depending on whether the interval just missed (Gibbs and Candes, 2021). A forecast k steps ahead only learns its outcome k + 1 steps later, so the feedback is delayed accordingly.
- **Rolling conformal**: recalibrated every day from the most recent 7 days of residuals that have already arrived.
- **Quantile network**: a second network trained with the pinball loss to output the 5% and 95% quantiles, used as is, and **CQR**, the same band widened or narrowed by a conformal correction measured on the calibration part (Romano et al., 2019).

Coverage is the share of observed targets inside the interval, width is in mph, and the interval score (Gneiting and Raftery, 2007) is the width plus {penalty:.0f} times how far a target missed, so a narrow interval only scores well if it holds. I split the results by what is known when the forecast is made: the last observed speed below or above 40 mph, and weekday rush hours (7 to 10 and 16 to 19).

## Results

Nominal coverage {nominal}, averaged over the 3 seeds (the SD of the overall coverage is under {max_sd:.3f} for every method). "Congested" and "free flow" use the last observed speed at the forecast origin.

| Method | Coverage | Width (mph) | Interval score | Free flow | Congested | Rush hour | Weakest sensor |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

{findings}

![Reliability](uncertainty-reliability.png)

The Gaussian intervals look fine at 90% only because they are {wider_train} wider than split conformal, and they stop being calibrated as soon as the level moves: at a nominal 50% they cover {gauss_half}. Residuals of traffic forecasts are far from Gaussian (most are small and a few are huge), so a standard deviation is the wrong summary. The conformal lines follow the diagonal at every level.

![Coverage by horizon](uncertainty-horizon.png)

Split conformal is calibrated across horizons, with coverage between {hmin:.3f} and {hmax:.3f} for 5 to 60 minutes ahead, while the width grows from {w5:.1f} to {w60:.1f} mph.

![Coverage by condition](uncertainty-conditions.png)

## Why plain conformal fails in congestion

Split conformal guarantees coverage on average over windows that look alike. Here it uses one quantile for everything, but the errors are not alike: when traffic is already congested the next hour is much harder to predict, so the residuals are bigger. One quantile that is right on average is too small in congestion and too big in free flow, which is exactly what the table shows ({cong_cov} against {free_cov}). Doing it per sensor doesn't help in congestion ({sensor_cong}) because the problem is the traffic state, not the sensor, although it does lift the weakest sensor from {weak_pooled} to {weak_sensor}. ACI and rolling recalibration react to drift in time, which isn't what is wrong here: they leave the congestion gap in place ({aci_cong} and {roll_cong}).

The quantile network learns the state-dependent width itself. In congestion its intervals are {cqr_cong_w:.0f} mph wide on average against {free_w:.0f} in free flow, and the conformal correction then fixes the overall level. At the same average width, that moves congestion coverage from {cong_cov} to {cqr_cong}.

![Coverage on each test day](uncertainty-daily.png)

Day by day, coverage for split conformal ranges from {dmin:.3f} to {dmax:.3f}. The ACI and rolling versions have nearly the same range ({aci_range}), so there is no strong drift in coverage over these {days} days for them to correct. CQR's range is {cqr_range}.

## Limits

- Conformal coverage is guaranteed when calibration and test windows are exchangeable. Here neighbouring windows overlap, the daily pattern repeats and the test period may drift, so coverage is measured, not guaranteed. The calibration part is only {cal_days} days.
- One point model, 16 sensors that all sit on one highway corridor, one test period of {days} days.
- Intervals are per target. They don't say anything about errors being correlated across horizons or neighbouring sensors, so they are not joint prediction regions.
- "Congested" is defined from the last observed speed with a 40 mph threshold, which is a choice. Defining it from the realized speed gives the diagnostic number below, and every method undercovers there because selecting on the outcome always does: split conformal covers {diag_p}, CQR {diag_c}.
- The quantile network is trained for one level (90%). Other levels need another network, which is why CQR is missing from the reliability plot except at 90%.
- Missing readings are skipped everywhere, so coverage only describes observed targets.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli uncertainty --config configs/uncertainty.json --out runs/uncertainty
```

Files: [summary.json](summary.json), [config.json](config.json), per-seed results in `seed-7.json`, `seed-42.json` and `seed-2026.json`. Method code is in `src/tracast/conformal.py` and `src/tracast/uncertainty.py`; the tests build noise with a known answer and check that the online methods never read an outcome that hasn't arrived.
"""


def uncertainty_page(summary, config):
    m, sd, rel, total = unc_facts(summary)
    bench = config["benchmark"]
    n, cal = bench["rows"], config["calibration"]
    a, c, b = int(n * 0.7), int(n * (0.8 - cal)), int(n * 0.8)
    start = pd.Timestamp("2012-03-01")
    day = lambda row: str((start + pd.Timedelta(minutes=5 * row)).date())  # noqa: E731
    pooled, cqr = m["conformal_pooled"], m["cqr"]
    horizons = [h["coverage"] for h in pooled["per_horizon"]]
    widths = [h["width"] for h in pooled["per_horizon"]]
    daily = lambda name: [d["coverage"] for d in m[name]["per_day"]]  # noqa: E731
    rng = lambda name: f"{min(daily(name)):.3f} to {max(daily(name)):.3f}"  # noqa: E731
    alpha = summary["alpha"]
    gaussian_train = (
        m["gaussian_train"]["overall"]["width"] / pooled["overall"]["width"] - 1
    )
    mae, mae_sd = summary["point_test_mae"]
    return UNCERTAINTY.format(
        nominal=pct(1 - alpha),
        train_dates=f"{day(0)} to {day(a - 1)}",
        val_dates=f"{day(a)} to {day(c - 1)}",
        cal_dates=f"{day(c)} to {day(b - 1)}",
        test_dates=f"{day(b)} to {day(n - 1)}",
        val_pct=pct(0.1 - cal),
        cal_pct=pct(cal),
        days=len(pooled["per_day"]),
        cal_days=round((b - c) / 288),
        mae=mae,
        mae_sd=mae_sd,
        penalty=2 / alpha,
        max_sd=max(sd[k]["overall"]["coverage"] for k in m) + 0.0005,
        table=unc_table(summary),
        findings="\n".join(unc_findings(summary)),
        wider_train=pct(gaussian_train),
        gauss_half=pct(rel["gaussian_train"]["0.5"]["coverage"]),
        hmin=min(horizons),
        hmax=max(horizons),
        w5=widths[0],
        w60=widths[-1],
        cong_cov=pct(pooled["origin_congested"]["coverage"]),
        sensor_cong=pct(m["conformal_sensor"]["origin_congested"]["coverage"]),
        weak_pooled=f"{pooled['sensor_coverage']['min']:.3f}",
        weak_sensor=f"{m['conformal_sensor']['sensor_coverage']['min']:.3f}",
        free_cov=pct(pooled["origin_free_flow"]["coverage"]),
        aci_cong=pct(m["conformal_aci"]["origin_congested"]["coverage"]),
        roll_cong=pct(m["conformal_rolling"]["origin_congested"]["coverage"]),
        cqr_cong_w=cqr["origin_congested"]["width"],
        free_w=cqr["origin_free_flow"]["width"],
        cqr_cong=pct(cqr["origin_congested"]["coverage"]),
        dmin=min(daily("conformal_pooled")),
        dmax=max(daily("conformal_pooled")),
        aci_range=f"{rng('conformal_aci')} and {rng('conformal_rolling')}",
        cqr_range=rng("cqr"),
        diag_p=pct(pooled["target_congested_diagnostic"]["coverage"]),
        diag_c=pct(cqr["target_congested_diagnostic"]["coverage"]),
    )


def build_uncertainty():
    summary = load("reports/uncertainty/summary.json")
    config = load("reports/uncertainty/config.json")
    if summary is None:
        return None
    (REPORTS / "uncertainty/report.md").write_bytes(
        uncertainty_page(summary, config).encode()
    )
    return summary


if __name__ == "__main__":
    build_uncertainty()
