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


# --- scaling -------------------------------------------------------------------------

SCALE_LABELS = {
    "persistence": "Persistence",
    "ha": "Historical average",
    "linear_time": "Linear + calendar (dense)",
    "gru_time": "GRU + calendar (dense)",
    "graph_gru_time": "Graph-GRU + calendar (dense)",
    "sgru_time": "Shared GRU + calendar",
    "sgraph_time": "Shared Graph-GRU + calendar",
}


def cell(entry, key, places=3, sd=True):
    value = entry[key]
    text = f"{value['mean']:.{places}f}"
    if sd and value.get("sd") is not None:
        text += f" ({value['sd']:.{places}f})"
    return text


def size_table(summary, key, places=3, sd=True):
    sizes = summary["sizes"]
    rows = []
    for name, label in SCALE_LABELS.items():
        by = summary["models"].get(name)
        if by is None:
            continue
        cells = [
            cell(by[str(k)], key, places, sd) if str(k) in by else "" for k in sizes
        ]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    head = "| Model | " + " | ".join(str(k) for k in sizes) + " |"
    rule = "|---|" + "---:|" * len(sizes)
    return "\n".join([head, rule] + rows)


def epoch_seconds(entry):
    return entry["train_seconds"]["mean"] / entry["epochs_run"]["mean"]


def scale_findings(summary, cost):

    sizes = summary["sizes"]
    small, large = sizes[0], sizes[-1]
    models = summary["models"]
    gaps = summary["day_bootstrap"]
    graph = [gaps[str(k)]["sgraph_time_vs_sgru_time"] for k in sizes if str(k) in gaps]
    helps = sum(g["ci95"][1] < 0 for g in graph)
    dense = [
        gaps[str(k)]["graph_gru_time_vs_gru_time"] for k in sizes if str(k) in gaps
    ]
    out = [
        f"- **In the shared models the graph helps, at {helps} of {len(graph)} sizes.** Adding the "
        f"neighbor averages changes MAE by {min(g['estimate'] for g in graph):+.3f} to "
        f"{max(g['estimate'] for g in graph):+.3f} mph. Without them a shared model sees nothing "
        f"about its neighbors, so this is a clean test of the graph, which the dense models can't give.",
        f"- **In the dense models it barely shows.** The same graph features move the dense GRU by "
        f"{min(d['estimate'] for d in dense):+.3f} to {max(d['estimate'] for d in dense):+.3f} mph, "
        f"because a dense GRU already reads every sensor.",
    ]
    names = ("gru_time", "graph_gru_time", "sgru_time", "sgraph_time")
    change = {
        n: models[n][str(large)]["core16_mae"]["mean"]
        - models[n][str(small)]["core16_mae"]["mean"]
        for n in names
    }
    word = "worse" if min(change.values()) > 0 else "different"
    out.append(
        f"- **More sensors did not make the first {small} better.** Trained on all {large} sensors, "
        f"the models are {min(change.values()):+.3f} to {max(change.values()):+.3f} mph away from where "
        f"they were when trained on {small} (same targets, so positive means {word}). With the same "
        f"capacity and the same number of updates, a bigger and more varied network is harder to fit."
    )
    source = cost or summary
    ratio = {
        n: epoch_seconds(source["models"][n][str(large)])
        / epoch_seconds(source["models"][n][str(small)])
        for n in names
    }
    params = models["graph_gru_time"][str(large)]["parameters"]
    shared = models["sgraph_time"][str(large)]["parameters"]
    out.append(
        f"- **The shared model's size doesn't grow, its compute does.** It stays at {shared:,} parameters "
        f"while the dense Graph-GRU needs {params:,} at {large} sensors. But the time per epoch grows "
        f"x{ratio['sgraph_time']:.1f} from {small} to {large} sensors for the shared Graph-GRU (there are "
        f"x{large / small:.0f} more sensors) against x{ratio['graph_gru_time']:.1f} for the dense one."
    )
    best = min(names, key=lambda n: models[n][str(large)]["mae"]["mean"])
    skill = (
        1
        - models[best][str(large)]["mae"]["mean"]
        / models["ha"][str(large)]["mae"]["mean"]
    )
    out.append(
        f"- **On all {large} sensors the best model is {SCALE_LABELS[best]}** at "
        f"{models[best][str(large)]['mae']['mean']:.3f} mph, {skill:.0%} lower than the historical "
        f"average ({models['ha'][str(large)]['mae']['mean']:.3f})."
    )
    return out


SCALING = """# Scaling to the whole network

Everything so far used 16 sensors that happen to sit on one highway. METR-LA has 207, so the obvious question is what changes when the model sees all of them, and whether the road graph helps more when there is more of it.

![Nested sensor subsets](scaling-subsets.png)

## Setup

The sensors are prefixes of one ordering of the whole graph (a breadth-first walk from the first sensor, then the one sensor the walk can't reach). So the first 16 are exactly the sensors from the earlier experiments and every bigger set contains the smaller ones. The sizes are {sizes}. Splits, windows, masking and training-only normalization are the same as before, and a test checks that the 16-sensor results didn't move.

Models, all with 32 hidden units and the calendar inputs:

- **Dense GRU and dense Graph-GRU**: the earlier models. They read all sensors at once, so their input and output layers grow with the sensor count.
- **Shared GRU and shared Graph-GRU**: one small GRU run on every sensor separately with the same weights, so the size doesn't depend on the sensor count. The shared Graph-GRU also gets the average speed of each sensor's upstream and downstream neighbors; the shared GRU gets nothing but the sensor's own history.
- A linear model, the historical average and persistence as baselines.

![Shared Graph-GRU, schematic](../figures/shared-architecture.png)

Every model gets the same number of updates at every size: {epochs} epochs that each use a random third of the training windows (so {full} full passes), with the learning rate halved when validation stalls, and the best epoch on validation is kept. Seeds: {seeds}. That is a budget, not early stopping, so a model that needs longer is at a disadvantage; the earlier experiments showed this matters, and it is one of the limits below.

## Two kinds of comparison

The sets of sensors differ between sizes, so a score on "all the sensors" isn't comparable across sizes: the added sensors are simply harder (persistence goes from {p16} to {plarge} mph and the historical average from {h16} to {hlarge}). I report two things:

- **Own population**: each model scored on all the sensors it was trained on. Fine for comparing models at one size.
- **Same 16 sensors**: every model scored on the first 16 sensors only, so the targets are identical at every size. This is the fair way to ask whether more sensors help.

## Results

Test MAE in mph on the same first 16 sensors, mean (SD over seeds):

{core_table}

Test MAE in mph on all the sensors each model was trained on (not comparable across columns):

{own_table}

{findings}

Gaps between models at each size, on identical windows (first minus second, negative favors the first; 95% interval from resampling test days, seeds averaged first):

| Size | Shared graph - shared plain | Shared plain - dense plain | Shared graph - dense graph | Dense graph - dense plain |
|---|---:|---:|---:|---:|
{gap_rows}

![Accuracy by sensor count](scaling-accuracy.png)

## Cost

{cost_intro}

| Model | Parameters at {large} | Seconds per epoch, {sizes_short} | Peak memory (MiB), {sizes_short} | Inference ms per window, {sizes_short} |
|---|---:|---:|---:|---:|
{cost_rows}

![Cost by sensor count](scaling-cost.png)

## Limits

- Equal update budgets, not equal convergence. A model that is still improving at the end is undersold, and the shared models are slower to train per update.
- Two seeds per cell. The paired test-day intervals say how much a gap depends on which days were held out, but the seed SDs here come from only two runs.
- The sensor ordering is one breadth-first walk. A different walk (or a random subset) would give different nested sets, and the first 16 are a single highway segment, so "the first 16" is not a typical set of sensors.
- All 207 sensors share one network, one time period and one city. Nothing here says how it behaves in another network.
- The models have 32 hidden units at every size. Giving the bigger sets more capacity is the first thing to try next, since the same-16 scores got worse as the population grew.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli scaling --config configs/scaling.json --out runs/scaling
.\\.venv\\Scripts\\python.exe -m tracast.cli scaling --config configs/scaling-cost.json --out runs/scaling-cost
```

The first is the accuracy run ({epochs} epochs, {nseeds} seeds). The second repeats the setup for 3 epochs with one seed and nothing else running on the machine, and the cost numbers come from it. The accuracy run's own timings were taken while other jobs shared the machine, so they are in `scaling-summary.json` but not used. Files: [scaling-summary.json](scaling-summary.json), [cost-summary.json](cost-summary.json), [scaling-config.json](scaling-config.json).
"""


def scaling_page(summary, cost):
    sizes = summary["sizes"]
    large = sizes[-1]
    source = cost or summary
    gaps = summary["day_bootstrap"]

    def gap(size, key):
        g = gaps.get(str(size), {}).get(key)
        if g is None:
            return ""
        return f"{g['estimate']:+.3f} ({g['ci95'][0]:+.3f} to {g['ci95'][1]:+.3f})"

    gap_rows = "\n".join(
        f"| {k} | {gap(k, 'sgraph_time_vs_sgru_time')} | {gap(k, 'sgru_time_vs_gru_time')} "
        f"| {gap(k, 'sgraph_time_vs_graph_gru_time')} | {gap(k, 'graph_gru_time_vs_gru_time')} |"
        for k in sizes
    )

    def series(name, getter, places):
        by = source["models"][name]
        return " / ".join(
            f"{getter(by[str(k)]):.{places}f}" for k in sizes if str(k) in by
        )

    cost_rows = "\n".join(
        f"| {SCALE_LABELS[name]} | {summary['models'][name][str(large)]['parameters']:,} "
        f"| {series(name, epoch_seconds, 1)} "
        f"| {series(name, lambda e: e['peak_memory_mib']['mean'], 0)} "
        f"| {series(name, lambda e: e['inference_ms_per_window']['mean'], 3)} |"
        for name in (
            "linear_time",
            "gru_time",
            "graph_gru_time",
            "sgru_time",
            "sgraph_time",
        )
    )
    config = json.loads((ROOT / "reports/scaling/scaling-config.json").read_text())[
        "benchmark"
    ]
    persistence, ha = summary["models"]["persistence"], summary["models"]["ha"]
    return SCALING.format(
        sizes=", ".join(map(str, sizes)),
        sizes_short="/".join(map(str, sizes)),
        large=large,
        epochs=config["epochs"],
        full=round(config["epochs"] / config["train_stride"]),
        seeds=", ".join(map(str, config["seeds"])),
        nseeds=len(config["seeds"]),
        p16=f"{persistence[str(sizes[0])]['mae']['mean']:.2f}",
        plarge=f"{persistence[str(large)]['mae']['mean']:.2f}",
        h16=f"{ha[str(sizes[0])]['mae']['mean']:.2f}",
        hlarge=f"{ha[str(large)]['mae']['mean']:.2f}",
        core_table=size_table(summary, "core16_mae"),
        own_table=size_table(summary, "mae"),
        findings="\n".join(scale_findings(summary, cost)),
        gap_rows=gap_rows,
        cost_intro=(
            "Measured in a separate short run with nothing else running on the machine "
            "(8 threads, 3 epochs of a random third of the windows, one seed)."
            if cost
            else "Measured in the main run, which shared the machine with other jobs, so treat these as rough."
        ),
        cost_rows=cost_rows,
    )


def build_scaling():
    summary = load("reports/scaling/scaling-summary.json")
    if summary is None:
        return None
    cost = load("reports/scaling/cost-summary.json")
    (REPORTS / "scaling/report.md").write_bytes(scaling_page(summary, cost).encode())
    return summary


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
    build_scaling()
    build_uncertainty()
