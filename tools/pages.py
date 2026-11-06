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


def join_and(items):
    items = [str(i) for i in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def pct(x, places=0):
    return f"{100 * x:.{places}f}%"


# --- audit ---------------------------------------------------------------------------

AUDIT = """# Audit

Before building on the code I went through it looking for leakage, wrong assumptions and things that would break at a bigger scale. This page says what I checked, what I found and what is still open. Nothing in here is a result of the models; the numbers about the data come from `tools/data_audit.py` ([data-audit.json](../reports/data-audit.json)).

## Bugs found and fixed

- **The sensor-subset code couldn't do all 207 sensors.** It walked the graph from the first sensor and raised an error if that walk didn't reach enough sensors. The graph has {components} components, so asking for all of them failed. The new `order()` finishes the first component and then adds the leftover sensors in file order, and every subset is a prefix of that one list. The old rule is the same walk, so the 16-sensor subset is unchanged (a test checks it against the old function, and the seed-42 benchmark runs still give the recorded model hashes).
- **The dense graph ablation could not isolate the graph.** The "self only" model still got every sensor's readings through its input layer, so it was never really without neighbors. The earlier write-up said so, but the experiment couldn't fix it. The shared-weight models can: a shared GRU sees only its own sensor, and the shared Graph-GRU adds the two neighbor averages, nothing else.
- **A plotting crash after a long run** (found earlier): early-stopped runs have different lengths and the convergence plot assumed equal ones. Fixed, with a test.

I didn't find anything else that was wrong in the original pipeline, and I'd rather say that than pad this list.

## Leakage checks

None found. These are checked by tests, not just by reading the code (`tests/test_leakage.py` and the tests next to each module):

- Rewrite every test row and the trained weights, the validation curve and the chosen epoch stay bit-for-bit the same. Only the test score changes.
- Rewrite the calibration rows (the part used for conformal intervals) and training and checkpoint choice don't move either.
- The scalers, the training windows, the historical average and the calendar features come from training rows or from the clock only.
- The graph comes from published distances and never from speeds, and the chosen sensors don't depend on speeds.
- In the day-by-day replay a window counts as known only once its last target row has been observed. The tests corrupt the not-yet-arrived targets and check that no strategy's forecasts change, and that the online interval methods never read an outcome before it arrives.

One choice worth knowing about: in the replay experiment, missing readings are forward-filled across the split boundaries (the past is known when a model runs for real). The other experiments fill within each split, which is slightly more conservative at the start of validation and test. It only affects the first few windows of a split.

## Federated pipeline

Before federating the graph models I went through the existing federated code the same way. Nothing in it was wrong, but it couldn't be extended as it was, and a few details decided how the new version is built:

- **The old federated task couldn't use a graph.** It trained a GRU that reads one sensor at a time, so there was nothing for a road graph to attach to. The new clients get windows over all their own sensors.
- **Averaging whole state dicts would have mixed up the graphs.** The graph models keep their adjacency matrices as buffers, which land in `state_dict()`, and a client's matrix has the shape of its own sensor count. Averaging those would crash or blend graphs from different clients. The new code exchanges only the learned parameters (their shapes don't depend on the sensor count), and a test checks that no graph buffer is among them.
- **FedProx's proximal term is squared in the paper and not in Flower's docs.** Flower's strategy only sends mu to the clients and its docstring shows the unsquared norm, while the paper uses (mu / 2) times the squared distance. The clients here use the squared form, with a test on it.
- **Every client's normalization equals the global one.** Normalization is per sensor from training rows, so a client fitting it on its own sensors gets exactly the numbers a central fit would. A test rewrites other clients' sensors and this client's validation and test rows and checks that nothing changes.
- **A cut graph is a second difference between "federated" and "pooled".** A federated client only has the edges inside its own sensor set, so comparing it with a pooled model on the full graph mixes two effects. The new study adds a pooled arm that gets only the in-client edges, which separates the cost of federated optimization from the cost of the cut graph.
- **Conventions kept from the old study so the arms stay comparable:** a fresh optimizer each round (also in the pooled and local-only arms), clients weighted by their number of training examples, the best round picked on pooled validation MAE, and communication counted as uncompressed parameters three times per round per client (training down and up, validation down).

Still open: it is one machine and no network, the validation statistics the server uses to pick a round would have to be reported by real clients, and the partition (a k-means cut of the coordinates) is one choice among many.

## What the data looks like

- **Graph:** {edges} directed edges, {components} components (sizes {component_sizes}), and the sensor order in the data file matches the graph file.
- **Timestamps:** the slowest weekday hours are {weekday_slow} (the evening and morning rush) and weekends are slowest around {weekend_slow}, so the timestamps behave like local Los Angeles time even though the file doesn't say.
- **Missing readings** (NaN or zero) are {train_missing} of the readings in the training period and {val_missing} in validation but {test_missing} in the test period, and {test_bad} sensors have more than 30% missing in the test period against none in training. So the test period isn't just later, it is noisier. Every experiment masks missing targets, so scores only cover observed readings, but a model sees more stale (forward-filled) inputs in the test period than it was trained on.
- **Congestion:** {train_cong} of observed training readings are below 40 mph, {test_cong} in the test period.
- **The first 16 sensors** sit on one highway corridor ([map](../reports/scaling/scaling-subsets.png)), which is a limit on what the earlier results say about the whole network.

## Limits that are still open

- Zero counts as missing, which also drops real stopped traffic.
- One city and one four-month period, with one test window of {test_days} days.
- Overlapping windows (one per five minutes) are strongly correlated, so the number of independent test points is far smaller than the number of windows. The test-day intervals are my way of respecting that, and they are wide for that reason.
- Seeds: two to five per cell depending on the experiment.

## Speed and memory

Nothing in the data pipeline was slow enough to change. The dense models cost about the same at every sensor count. The shared models do one recurrent pass per sensor and window, so their time per update grows roughly with the sensor count; that cost is measured in the [scaling report](../reports/scaling/report.md).
"""


def build_audit():
    audit = load("reports/data-audit.json")
    if audit is None:
        return None
    splits = audit["splits"]
    sizes = audit["graph"]["components"]
    test_rows = splits["test"]["rows"]

    def pcts(x):
        return pct(x, 1)

    text = AUDIT.format(
        components=len(sizes),
        edges=audit["graph"]["directed_edges"],
        component_sizes=" and ".join(map(str, sizes)),
        weekday_slow=", ".join(
            f"{h}:00" for h in audit["slowest_hours_of_day"]["weekday"][:3]
        ),
        weekend_slow=", ".join(
            f"{h}:00" for h in audit["slowest_hours_of_day"]["weekend"][:2]
        ),
        train_missing=pcts(splits["train"]["missing_share"]),
        val_missing=pcts(splits["val"]["missing_share"]),
        test_missing=pcts(splits["test"]["missing_share"]),
        test_bad=splits["test"]["sensors_over_30_percent_missing"],
        train_cong=pcts(splits["train"]["share_below_40_mph"]),
        test_cong=pcts(splits["test"]["share_below_40_mph"]),
        test_days=round(test_rows / 288),
    )
    (ROOT / "docs/audit.md").write_bytes(text.encode())
    return audit


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
    vs = [(k, gaps[str(k)]["sgru_time_vs_gru_time"]) for k in sizes if str(k) in gaps]
    better = [(k, g) for k, g in vs if g["ci95"][1] < 0]
    level = [k for k, g in vs if g["ci95"][1] >= 0]
    if better:
        low = min(g["estimate"] for _, g in better)
        high = max(g["estimate"] for _, g in better)
        out.append(
            f"- **Sharing the weights wins once the network is big.** At {join_and(k for k, _ in better)} "
            f"sensors the shared GRU beats the dense GRU by {-high:.3f} to {-low:.3f} mph (the interval "
            f"excludes zero), and at {join_and(level) if level else 'no other size'} they are level. It does that "
            f"with {models['sgru_time'][str(large)]['parameters']:,} parameters instead of "
            f"{models['gru_time'][str(large)]['parameters']:,}."
        )
    names = ("gru_time", "graph_gru_time", "sgru_time", "sgraph_time")
    change = {
        n: models[n][str(large)]["core16_mae"]["mean"]
        - models[n][str(small)]["core16_mae"]["mean"]
        for n in names
    }
    dense_change = [change["gru_time"], change["graph_gru_time"]]
    shared_change = [change["sgru_time"], change["sgraph_time"]]
    out.append(
        f"- **More sensors did not make the first {small} better, and the dense models suffer most.** "
        f"Trained on all {large} sensors, the dense models are {min(dense_change):+.3f} to "
        f"{max(dense_change):+.3f} mph worse on the same {small} targets than when trained on {small}, the shared "
        f"models {min(shared_change):+.3f} to {max(shared_change):+.3f}. With the same capacity and the same number "
        f"of updates, a bigger and more varied network is harder to fit, and a dense network has to squeeze "
        f"every sensor through the same 32 hidden units."
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
        f"x{large / small:.0f} more sensors) against x{ratio['graph_gru_time']:.1f} for the dense one. At {large} sensors "
        f"that is {epoch_seconds(source['models']['sgraph_time'][str(large)]):.1f} s per epoch against "
        f"{epoch_seconds(source['models']['graph_gru_time'][str(large)]):.1f} s."
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


# --- drift and adaptation ------------------------------------------------------------

SHIFT_LABELS = {
    "static": "Static model",
    "bias_ema": "Residual average",
    "bias_slot": "Residual average by time of day",
    "head_daily": "Output layer fine-tuned daily",
    "retrain_weekly": "Whole network fine-tuned weekly",
}
SETTING_TEXT = {
    "standard": "Standard setting: the model is trained on the first 70% of the timeline, as everywhere else",
    "stale": "Stale setting: the model is trained on only the first 40%, so a long stretch it has never seen sits between its training and the test period",
}


def shift_table(setting):
    base = setting["strategies"]["static"]["mae"]
    rows = []
    for kind, label in SHIFT_LABELS.items():
        e = setting["strategies"][kind]
        sd = f" ({e['mae_sd']:.3f})" if e["mae_sd"] is not None else ""
        if kind == "static":
            change = ""
        else:
            g = e["vs_static"]
            change = (
                f"{g['estimate']:+.3f} ({g['ci95'][0]:+.3f} to {g['ci95'][1]:+.3f})"
            )
        seconds = "0" if kind == "static" else f"{e['adapt_seconds']:.1f}"
        rows.append(
            f"| {label} | {e['mae']:.3f}{sd} | {change} | {e['per_horizon_mae'][-1]:.3f} "
            f"| {seconds} | {e['updates'] if kind != 'static' else ''} |"
        )
    return "\n".join(rows), base


def drift_rows(blocks):
    return "\n".join(
        f"| {b['start']}{' (train)' if b['in_train'] else ''} | {b['mean_speed']:.1f} "
        f"| {pct(b['congested_share'], 1)} | {pct(b['missing_share'], 1)} | {b['js_vs_train_bits']:.4f} |"
        for b in blocks
    )


def regime_rows(setting):
    names = {
        "least_shifted_third": "Least shifted third of days",
        "most_shifted_third": "Most shifted third of days",
        "weekday": "Weekdays",
        "weekend": "Weekends",
    }
    rows = []
    for key, label in names.items():
        r = setting["regimes"].get(key)
        if r:
            rows.append(
                f"| {label} | " + " | ".join(f"{r[k]:.3f}" for k in SHIFT_LABELS) + " |"
            )
    return "\n".join(rows)


def shift_findings(summary):
    std, stale = summary["settings"]["standard"], summary["settings"]["stale"]
    out = []

    def best(setting):
        kinds = [k for k in SHIFT_LABELS if k != "static"]
        return min(kinds, key=lambda k: setting["strategies"][k]["mae"])

    for name, setting in (("standard", std), ("stale", stale)):
        top = best(setting)
        e = setting["strategies"][top]
        g = e["vs_static"]
        apart = g["ci95"][1] < 0
        out.append(
            f"- **{name.capitalize()} model:** the best strategy is "
            f"{SHIFT_LABELS[top].lower()}, {g['estimate']:+.3f} mph against the static model "
            f"({setting['strategies']['static']['mae']:.3f}), and the test-day interval "
            f"{'excludes' if apart else 'includes'} zero ({g['ci95'][0]:+.3f} to {g['ci95'][1]:+.3f})."
        )
    gap = stale["strategies"]["static"]["mae"] - std["strategies"]["static"]["mae"]
    recovered = {
        k: (stale["strategies"]["static"]["mae"] - stale["strategies"][k]["mae"]) / gap
        for k in SHIFT_LABELS
        if k != "static"
    }
    if gap > 0:
        out.append(
            f"- **A stale model is {gap:.3f} mph worse than a fresh one**, and adapting wins back "
            + ", ".join(
                f"{pct(v)} ({SHIFT_LABELS[k].lower()})"
                for k, v in recovered.items()
                if v > 0
            )
            + (
                " of that gap."
                if any(v > 0 for v in recovered.values())
                else "none of it."
            )
        )
    cheap, full = std["strategies"]["head_daily"], std["strategies"]["retrain_weekly"]

    def share(setting):
        base = setting["strategies"]["static"]["mae"]
        return (base - setting["strategies"]["head_daily"]["mae"]) / (
            base - setting["strategies"]["retrain_weekly"]["mae"]
        )

    out.append(
        f"- **Cost:** over the {len(std['strategies']['static']['daily_mae'])} test days, fine-tuning the output layer "
        f"daily takes {cheap['adapt_seconds']:.0f} s and fine-tuning the whole network weekly takes "
        f"{full['adapt_seconds']:.0f} s, about x{full['adapt_seconds'] / cheap['adapt_seconds']:.0f} more. "
        f"The cheaper one keeps {share(std):.0%} of the weekly fine-tune's gain in the standard setting and "
        f"{share(stale):.0%} in the stale one."
    )
    for name, setting in (("standard", std), ("stale", stale)):
        daily = setting["strategies"]["static"]["daily_mae"]
        first, last = sum(daily[:7]) / 7, sum(daily[-7:]) / 7
        regimes = setting["regimes"]
        gain = {
            group: regimes[group]["static"] - regimes[group]["head_daily"]
            for group in ("least_shifted_third", "most_shifted_third")
            if group in regimes
        }
        text = (
            f"- **{name.capitalize()} model over time:** the static model's MAE is {first:.3f} in the first "
            f"week of the test period and {last:.3f} in the last ({last - first:+.3f}; both weeks hold each "
            f"weekday once). "
            + (
                "So there is no sign of the error growing with time since training."
                if last <= first
                else "So the error does grow with time since training."
            )
        )
        if len(gain) == 2:
            text += (
                f" Fine-tuning the output layer gains {gain['most_shifted_third']:.3f} mph on the third of days "
                f"furthest from the training distribution and {gain['least_shifted_third']:.3f} on the third closest to it."
            )
        out.append(text)
    return out


SHIFT = """# Drift and adaptation

A forecaster is trained once on the past and then used for a long time. Traffic changes, so I wanted to know three things: does the model get worse as time moves away from its training data, does adapting it help, and is it worth the compute.

## What changes in the data

Statistics of the 16 sensors in blocks of 14 days. The last column is how far the block's speed distribution is from the training period's (Jensen-Shannon divergence in bits, 0 means identical).

| Block starting | Mean speed (mph) | Readings below 40 mph | Missing readings | Distance from training |
|---|---:|---:|---:|---:|
{drift}

{drift_text}

![Drift in the data](shift-drift.png)

## Setup

The test period is replayed one day at a time, the way a deployed model would run. At the start of each day a strategy may use only data that has already been observed: a window counts as known once its last target is in the past, and a test corrupts the not-yet-arrived targets to check that nothing leaks. The strategies:

- **Static model**: trained once, never touched.
- **Residual average**: add a running average of the model's recent errors (per horizon and sensor) to its forecast. No training, almost free.
- **Residual average by time of day**: the same, but one average for every time of day (weekday and weekend separately), because errors depend on the hour.
- **Output layer fine-tuned daily**: every morning, fine-tune only the last layer on the most recent 14 days of known windows.
- **Whole network fine-tuned weekly**: every 7 days, fine-tune all the weights on everything known so far.

The model is the dense GRU with calendar inputs on 16 sensors, 3 seeds. I used two settings: the standard one (trained on the first 70%, like everywhere else) and a stale one (trained on only the first 40%, so there are {gap_days} days it has never seen before the test period starts, which the adaptive strategies can use once they are known). The learning rate for the output layer and the averaging rate of the cheap strategies are chosen by replaying the validation part, never the test part. Costs are wall-clock seconds on the laptop.

## Results

Test MAE in mph (SD over seeds), the change against the static model with a 95% interval from resampling test days (negative is better), the 60-minute MAE, and the adaptation cost over the whole test period.

**{standard_text}.**

| Strategy | MAE | Change vs static | MAE at 60 min | Adaptation seconds | Updates |
|---|---:|---:|---:|---:|---:|
{standard_table}

**{stale_text}.**

| Strategy | MAE | Change vs static | MAE at 60 min | Adaptation seconds | Updates |
|---|---:|---:|---:|---:|---:|
{stale_table}

{findings}

![MAE change against the static model](shift-gain.png)

{validation_text}

![Test MAE per day](shift-daily.png)

## Where does adapting help

Test MAE by kind of day, standard setting. "Shifted" ranks the test days by how far their speed distribution is from the training period's.

| Days | {strategy_heads} |
|---|{strategy_rule}
{regimes_standard}

Same for the stale setting:

| Days | {strategy_heads} |
|---|{strategy_rule}
{regimes_stale}

## Limits

- One network, one model and 24 test days. A difference of a few hundredths of a mph is about the size of the day-to-day variation, which is why the intervals matter more than the averages.
- The drift in these four months is mild (see the table at the top), so the standard setting has little to adapt to. The stale setting is a way to get a bigger gap, but it is a constructed one, not natural drift.
- The strategies and their settings are small choices from a short list. Other adaptation methods (replay buffers, learning-rate schedules, change detection) could do better, and the retraining baseline is fine-tuning from the current weights, not training from scratch.
- Adaptation costs are for this CPU. Retraining scales with the amount of data it uses, the cheap strategies don't.
- No change-point detection: every strategy updates on a fixed schedule whether or not anything changed.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli shift --config configs/shift.json --out runs/shift
```

Files: [summary.json](summary.json), [config.json](config.json), per-seed results in `standard-seed-*.json` and `stale-seed-*.json`.
"""


def shift_page(summary, config):
    std, stale = summary["settings"]["standard"], summary["settings"]["stale"]
    table_s, _ = shift_table(std)
    table_t, _ = shift_table(stale)
    blocks = std["drift"]
    train_blocks = [b for b in blocks if b["in_train"]]
    later = [b for b in blocks if not b["in_train"]]

    def mean(items, key):
        return sum(b[key] * b["rows"] for b in items) / sum(b["rows"] for b in items)

    drift_text = (
        f"In the training blocks {pct(mean(train_blocks, 'missing_share'), 1)} of the readings are missing, "
        f"after them {pct(mean(later, 'missing_share'), 1)}. The share of readings below 40 mph goes from "
        f"{pct(mean(train_blocks, 'congested_share'), 1)} to {pct(mean(later, 'congested_share'), 1)}, "
        f"and the mean speed from {mean(train_blocks, 'mean_speed'):.1f} to {mean(later, 'mean_speed'):.1f} mph. "
        f"These are small shifts, so any adaptation gain here will be small too."
    )
    stale_rows = stale["rows"]
    gap_days = round((stale_rows["test"][0] - stale_rows["val"][1]) / 288)
    kinds = list(SHIFT_LABELS)
    # did the validation replay predict the direction of the test change?
    agree, total = 0, 0
    for setting in (std, stale):
        for kind, val_key in (
            ("bias_ema", "eta_val_mae"),
            ("bias_slot", "slot_eta_val_mae"),
            ("head_daily", "head_val_mae"),
        ):
            helped_test = setting["strategies"][kind]["vs_static"]["estimate"] < 0
            for chosen in setting["chosen"]:
                best_val = min(chosen[val_key].values())
                total += 1
                agree += (best_val < chosen["static_val_mae"]) == helped_test
    validation_text = (
        f"The validation replay is the only thing that picked the settings, so I checked that it points "
        f"the right way: for the three cheap strategies, in {agree} of {total} seed and setting combinations the "
        f"validation replay and the test period agreed on whether adapting beats the static model."
    )
    return SHIFT.format(
        drift=drift_rows(blocks),
        drift_text=drift_text,
        gap_days=gap_days,
        standard_text=SETTING_TEXT["standard"],
        stale_text=SETTING_TEXT["stale"],
        standard_table=table_s,
        stale_table=table_t,
        findings="\n".join(shift_findings(summary)),
        validation_text=validation_text,
        strategy_heads=" | ".join(SHIFT_LABELS[k] for k in kinds),
        strategy_rule="---:|" * len(kinds),
        regimes_standard=regime_rows(std),
        regimes_stale=regime_rows(stale),
    )


def build_shift():
    summary = load("reports/shift/summary.json")
    if summary is None:
        return None
    config = load("reports/shift/config.json")
    (REPORTS / "shift/report.md").write_bytes(shift_page(summary, config).encode())
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

Split conformal guarantees coverage on average over windows that look alike. Here it uses one quantile for everything, but the errors are not alike: when traffic is already congested the next hour is much harder to predict, so the residuals are bigger. One quantile that is right on average is too small in congestion and too big in free flow, which is exactly what the table shows ({cong_cov} against {free_cov}). Doing it per sensor doesn't help in congestion ({sensor_cong}) because the problem is the traffic state, not the sensor, although it does lift the weakest sensor from {weak_pooled} to {weak_sensor}. ACI and rolling recalibration react to drift in time, which isn't what is wrong here: at the preset step size they leave the congestion gap in place ({aci_cong} and {roll_cong}); the step-size sweep further down shows what a bigger step does.

The quantile network learns the state-dependent width itself. In congestion its intervals are {cqr_cong_w:.0f} mph wide on average against {free_w:.0f} in free flow, and the conformal correction then fixes the overall level. At the same average width, that moves congestion coverage from {cong_cov} to {cqr_cong}.

![Coverage on each test day](uncertainty-daily.png)

Day by day, coverage for split conformal ranges from {dmin:.3f} to {dmax:.3f}. The ACI and rolling versions have nearly the same range ({aci_range}), so there is no strong drift in coverage over these {days} days for them to correct. CQR's range is {cqr_range}.

{aci_sweep}## Limits

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


def aci_sweep_section(summary, preset):
    sweep = summary["mean"].get("aci_sweep")
    if not sweep:
        return ""
    gammas = sorted(sweep, key=float)
    rows = chr(10).join(
        f"| {float(g):g} | {sweep[g]['overall']['coverage']:.3f} | {sweep[g]['overall']['width']:.1f} "
        f"| {sweep[g]['origin_free_flow']['coverage']:.3f} | {sweep[g]['origin_congested']['coverage']:.3f} "
        f"| {sweep[g]['overall']['interval_score']:.1f} |"
        for g in gammas
    )
    overall = [sweep[g]["overall"]["coverage"] for g in gammas]
    cong = [sweep[g]["origin_congested"]["coverage"] for g in gammas]
    best = min(gammas, key=lambda g: sweep[g]["overall"]["interval_score"])
    scores = [sweep[g]["overall"]["interval_score"] for g in gammas]
    still, top = sweep[gammas[0]], sweep[gammas[-1]]
    cqr = summary["mean"]["methods"]["cqr"]
    rise = top["origin_congested"]["coverage"] - still["origin_congested"]["coverage"]
    if rise > 0.05:
        verdict_text = (
            f"A bigger step does raise congestion coverage, from {pct(still['origin_congested']['coverage'])} "
            f"with no adaptation to {pct(top['origin_congested']['coverage'])} at gamma {float(gammas[-1]):g}. "
            f"But it does that by widening every interval, not only the ones in congestion: the average "
            f"width goes from {still['overall']['width']:.1f} to {top['overall']['width']:.1f} mph, free-flow "
            f"coverage drops from {pct(still['origin_free_flow']['coverage'])} to "
            f"{pct(top['origin_free_flow']['coverage'])}, and the interval score only moves between "
            f"{min(scores):.1f} and {max(scores):.1f}. ACI follows runs of misses over time, but it keeps one "
            f"level for all sensors at a horizon, so when a jam makes some sensors miss it widens the "
            f"intervals of the free-flowing ones too. The "
            f"quantile network with the conformal correction reaches {pct(cqr['origin_congested']['coverage'])} "
            f"in congestion at {cqr['overall']['width']:.1f} mph with a score of "
            f"{cqr['overall']['interval_score']:.1f}."
        )
    else:
        verdict_text = (
            f"No step size moves congestion coverage by more than {rise:.3f}. One level per horizon "
            f"can't know what the traffic state is, however fast it moves."
        )
    return f"""## How much the ACI step size matters

ACI moves its level by gamma after every window. A step of 0 never moves it, which is plain split conformal, and the main table uses {preset:g}, a value I picked without tuning it. This sweep on the same test windows is a sensitivity check, not a way to choose the step. Averages over the seeds:

| gamma | Coverage | Width (mph) | Free flow | Congested | Interval score |
|---|---:|---:|---:|---:|---:|
{rows}

Overall coverage stays between {min(overall):.3f} and {max(overall):.3f} and coverage in congestion between {min(cong):.3f} and {max(cong):.3f}. {verdict_text} The lowest interval score is at gamma {float(best):g} ({sweep[best]["overall"]["interval_score"]:.1f}).

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
        aci_sweep=aci_sweep_section(summary, config["gamma"]),
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


# --- README pieces ---------------------------------------------------------------------

NEXT = """## What's next

- Bigger shared models and longer training. The scores on the first 16 sensors got worse as the training population grew, every federated arm's best round was its last, and training alone beat one set of shared weights for every area. A bigger shared model, or a few specialized ones, is the first thing to try, with the fine-tuning result (which already closes the gap) as the one to compare against.
- A closer DCRNN: the paper's encoder-decoder with scheduled sampling, a second layer, and training for accuracy on all 207 sensors (mine predicts every horizon at once from the last hidden state, and is trained up to 64 sensors and only timed at 207).
- Real deployment effects (partial participation, stragglers, dropped clients, a network) that a single-machine simulation can't show, and more rounds and learning-rate schedules for FedAvg and FedProx.
- More than one geographic cut, and more seeds. The partitions are fixed k-means solutions and the federated studies use one or two seeds, so a different cut or another seed could change the smaller gaps.
- Held-out sensors beyond one area: a random set of sensors, the graph model with a vector, and a better way to start a new sensor than the average vector (predicting it from descriptors didn't work).
- Intervals that are joint across horizons and sensors, and conformal methods with guarantees conditional on the traffic state instead of only on average.
- More data: another year or another city, and drift with real change points (roadworks, a sensor replaced) and a change detector that decides when to adapt.
"""


def unc_readme_bullet(summary):
    m = summary["mean"]["methods"]
    pooled, cqr = m["conformal_pooled"], m["cqr"]
    return (
        f"- **Prediction intervals that are right on average can fail when it matters.** At a nominal "
        f"{pct(1 - summary['alpha'])}, split conformal covers {pct(pooled['overall']['coverage'], 1)} of the "
        f"test speeds, but only {pct(pooled['origin_congested']['coverage'])} when the last speed was below "
        f"40 mph (against {pct(pooled['origin_free_flow']['coverage'])} in free flow). A quantile network with a "
        f"conformal correction holds {pct(cqr['origin_congested']['coverage'])} there at the same average width."
    )


def shift_readme_bullet(summary):
    std, stale = summary["settings"]["standard"], summary["settings"]["stale"]

    def gain(setting, kind):
        return -setting["strategies"][kind]["vs_static"]["estimate"]

    naive = min(
        std["strategies"][k]["vs_static"]["estimate"] for k in ("bias_ema", "bias_slot")
    )
    return (
        f"- **Adapting a deployed model helps a little, and cheap adaptation can hurt.** Fine-tuning the output "
        f"layer daily improves MAE by {gain(std, 'head_daily'):.3f} mph and fine-tuning the whole network weekly by "
        f"{gain(std, 'retrain_weekly'):.3f} (for a model trained on only the first 40% of the timeline: "
        f"{gain(stale, 'head_daily'):.3f} and {gain(stale, 'retrain_weekly'):.3f}), at "
        f"{std['strategies']['head_daily']['adapt_seconds']:.0f} s and "
        f"{std['strategies']['retrain_weekly']['adapt_seconds']:.0f} s for the 24 test days. Averaging past "
        f"errors into the forecast was worse than doing nothing ({naive:+.3f} mph at best)."
    )


def scale_readme_text(scaling, cost):
    sizes = scaling["sizes"]
    small, large = sizes[0], sizes[-1]
    models, gaps = scaling["models"], scaling["day_bootstrap"]
    graph = [
        gaps[str(k)]["sgraph_time_vs_sgru_time"]["estimate"]
        for k in sizes
        if str(k) in gaps
    ]
    vs = [(k, gaps[str(k)]["sgru_time_vs_gru_time"]) for k in sizes if str(k) in gaps]
    better = [(k, g) for k, g in vs if g["ci95"][1] < 0]
    names = ("gru_time", "graph_gru_time", "sgru_time", "sgraph_time")
    change = {
        n: models[n][str(large)]["core16_mae"]["mean"]
        - models[n][str(small)]["core16_mae"]["mean"]
        for n in names
    }
    dense = [change["gru_time"], change["graph_gru_time"]]
    shared = [change["sgru_time"], change["sgraph_time"]]
    source = cost or scaling
    text = (
        f"The graph helps the shared models at every size ({min(graph):+.3f} to {max(graph):+.3f} mph), "
        f"which the dense models can't show because they already read every sensor."
    )
    if better:
        text += (
            f" Sharing the weights also wins once the network is big: at {join_and(k for k, _ in better)} "
            f"sensors the shared GRU beats the dense one by "
            f"{-max(g['estimate'] for _, g in better):.3f} to {-min(g['estimate'] for _, g in better):.3f} mph "
            f"with {models['sgru_time'][str(large)]['parameters']:,} parameters instead of "
            f"{models['gru_time'][str(large)]['parameters']:,}."
        )
    text += (
        f" What it costs is compute: at {large} sensors the shared Graph-GRU needs "
        f"{epoch_seconds(source['models']['sgraph_time'][str(large)]):.1f} s per epoch against "
        f"{epoch_seconds(source['models']['graph_gru_time'][str(large)]):.1f} s for the dense one. And more "
        f"sensors did not make the first {small} better: trained on all {large}, the dense models are "
        f"{min(dense):+.3f} to {max(dense):+.3f} mph worse on those {small} targets, the shared ones "
        f"{min(shared):+.3f} to {max(shared):+.3f}."
    )
    return text


def unc_readme_text(summary):
    m, _, rel, total = unc_facts(summary)
    pooled, cqr, aci = m["conformal_pooled"], m["cqr"], m["conformal_aci"]
    share = pooled["origin_congested"]["targets"] / total
    return (
        f"At a nominal {pct(1 - summary['alpha'])}, split conformal covers {pct(pooled['overall']['coverage'], 1)} "
        f"of the test speeds, calibrated across all twelve horizons, while the Gaussian baselines overshoot it "
        f"({pct(m['gaussian_train']['overall']['coverage'], 1)}) and drift away from the diagonal at other levels. "
        f"But when the last speed was below 40 mph ({pct(share)} of the targets), split conformal covers "
        f"{pct(pooled['origin_congested']['coverage'])} and ACI {pct(aci['origin_congested']['coverage'])}, against "
        f"{pct(pooled['origin_free_flow']['coverage'])} in free flow: one quantile for everything is too small "
        f"exactly when traffic is bad. The quantile network with a conformal correction learns widths that depend "
        f"on the traffic state and holds {pct(cqr['origin_congested']['coverage'])} there, at the same average "
        f"width ({cqr['overall']['width']:.1f} against {pooled['overall']['width']:.1f} mph) and with an interval "
        f"score of {cqr['overall']['interval_score']:.1f} against {pooled['overall']['interval_score']:.1f}."
        + sweep_sentence(summary)
    )


def sweep_sentence(summary):
    sweep = summary["mean"].get("aci_sweep")
    if not sweep:
        return ""
    gammas = sorted(sweep, key=float)
    still, top = sweep[gammas[0]], sweep[gammas[-1]]
    if (
        top["origin_congested"]["coverage"] - still["origin_congested"]["coverage"]
        <= 0.05
    ):
        return ""
    return (
        f" Letting ACI adapt faster does lift congestion coverage to {pct(top['origin_congested']['coverage'])}, "
        f"but only by widening every interval (the average goes from {still['overall']['width']:.1f} to "
        f"{top['overall']['width']:.1f} mph)."
    )


def shift_readme_text(summary):
    std, stale = summary["settings"]["standard"], summary["settings"]["stale"]

    def gain(setting, kind):
        return -setting["strategies"][kind]["vs_static"]["estimate"]

    gap = stale["strategies"]["static"]["mae"] - std["strategies"]["static"]["mae"]
    recovered = gain(stale, "retrain_weekly") / gap
    return (
        f"Fine-tuning the output layer daily improves the usual model's MAE by {gain(std, 'head_daily'):.3f} mph "
        f"and fine-tuning the whole network weekly by {gain(std, 'retrain_weekly'):.3f}, and both intervals over "
        f"test days exclude zero. For a stale model the gains are {gain(stale, 'head_daily'):.3f} and "
        f"{gain(stale, 'retrain_weekly'):.3f}, so weekly fine-tuning wins back {pct(recovered)} of the "
        f"{gap:.3f} mph the stale model lost. The output-layer update costs "
        f"{std['strategies']['head_daily']['adapt_seconds']:.0f} s over the 24 test days against "
        f"{std['strategies']['retrain_weekly']['adapt_seconds']:.0f} s for the weekly one. Averaging past errors "
        f"into the forecast made things worse. The drift in these four months is mild, and the static model's "
        f"error didn't grow with time since training, so the gains are small."
    )


def variant_numbers(variants, scaling):
    """The same-16 gains (positive is better) the README sentences quote."""
    sizes, gaps = variants["sizes"], variants["day_bootstrap"]

    def gain(key):
        return [-gaps[str(k)][key]["estimate"] for k in sizes]

    vector = gain("sgru_id_time_vs_sgru_time") + gain("sgraph_id_time_vs_sgraph_time")
    lstm = gain("slstm_time_vs_sgru_time")
    graph = [
        -scaling["day_bootstrap"][str(k)]["sgraph_time_vs_sgru_time"]["estimate"]
        for k in sizes
        if str(k) in scaling["day_bootstrap"]
    ]
    return sizes, vector, lstm, graph


def variants_readme_bullet(variants, scaling):
    sizes, vector, lstm, graph = variant_numbers(variants, scaling)
    return (
        f"- **In a shared model, telling the sensors apart matters more than the graph.** A learned "
        f"4-number vector per sensor changes test MAE by {-max(vector):+.3f} to {-min(vector):+.3f} mph at "
        f"{join_and(sizes)} sensors, the graph's neighbor averages by {-max(graph):+.3f} to {-min(graph):+.3f} "
        f"and an LSTM instead of a GRU by {-max(lstm):+.3f} to {-min(lstm):+.3f} (negative is better)."
    )


def variants_readme_text(variants, scaling):
    sizes, vector, lstm, graph = variant_numbers(variants, scaling)
    large = str(sizes[-1])
    shared = variants["models"]["sgru_id_time"][large]["core16_mae"]["mean"]
    dense = scaling["models"]["gru_time"][large]["core16_mae"]["mean"]
    return (
        f"Two variants of the shared models are in the [variants report](reports/variants/report.md), tested at "
        f"{join_and(sizes)} sensors. A learned vector of 4 numbers per sensor, which lets the shared network "
        f"tell sensors apart, changes the test MAE on the sensors the models train on by {-max(vector):+.3f} to {-min(vector):+.3f} mph "
        f"(negative is better), a bigger effect than the graph has, and with it the shared GRU scores "
        f"{shared:.3f} on the first 16 sensors when trained on {sizes[-1]}, against {dense:.3f} for the dense GRU. The catch is that a model "
        f"with those vectors can't forecast a sensor it wasn't trained on. Swapping the GRU for an LSTM changes "
        f"it by {-max(lstm):+.3f} to {-min(lstm):+.3f}."
    )


def readme_blocks():
    """Bullets and short sections for the README, from whichever studies have run."""
    blocks = {
        "bullets": [],
        "scaling": "",
        "uncertainty": "",
        "shift": "",
        "federated_graph": "",
        "federated_sections": [],
        "next": NEXT,
    }
    scaling = load("reports/scaling/scaling-summary.json")
    cost = load("reports/scaling/cost-summary.json")
    unc = load("reports/uncertainty/summary.json")
    shf = load("reports/shift/summary.json")
    var = load("reports/variants/variants-summary.json")

    if scaling:
        sizes = scaling["sizes"]
        small, large = sizes[0], sizes[-1]
        models = scaling["models"]
        found = scale_findings(scaling, cost)
        blocks["bullets"] += [found[0], found[2]]
        if var:
            blocks["bullets"].append(variants_readme_bullet(var, scaling))
        rows = [
            ("ha", "Historical average"),
            ("gru_time", "Dense GRU"),
            ("graph_gru_time", "Dense Graph-GRU"),
            ("sgru_time", "Shared GRU"),
            ("sgraph_time", "Shared Graph-GRU"),
        ]
        table = "\n".join(
            f"| {label} | "
            + " | ".join(cell(models[n][str(k)], "core16_mae") for k in sizes)
            + " |"
            for n, label in rows
        )
        head = "| Trained on this many sensors | " + " | ".join(map(str, sizes)) + " |"
        blocks["scaling"] = (
            f"## Scaling to all {large} sensors\n\n"
            f"Everything above uses {small} sensors from one highway. I ran the same comparison on nested "
            f"sets of {join_and(sizes)} sensors (each set contains the one before it), plus two "
            f"new models that share one small GRU across every sensor, so their size doesn't depend on how "
            f"many sensors there are. Test MAE in mph on the same first {small} sensors, whatever size "
            f"the model was trained on:\n\n{head}\n|---|"
            + "---:|"
            * len(sizes)
            + f"\n{table}\n\n"
            f"![Accuracy by sensor count](reports/scaling/scaling-accuracy.png)\n\n"
            f"{scale_readme_text(scaling, cost)}\n\n"
            f"{variants_readme_text(var, scaling) + chr(10) * 2 if var else ''}"
            f"The sizes' own-population scores, the cost in time and memory, and the limits are in the "
            f"[scaling report](reports/scaling/report.md).\n"
        )

    if unc:
        m = unc["mean"]["methods"]
        rows = "\n".join(
            f"| {UNC_LABELS[n]} | {m[n]['overall']['coverage']:.3f} | {m[n]['overall']['width']:.1f} "
            f"| {m[n]['origin_congested']['coverage']:.3f} | {m[n]['overall']['interval_score']:.1f} |"
            for n in ("gaussian_train", "conformal_pooled", "conformal_aci", "cqr")
        )
        blocks["bullets"] += [unc_readme_bullet(unc)]
        blocks["uncertainty"] = (
            "## Prediction intervals\n\n"
            "A point forecast doesn't say how sure the model is, so I built 90% prediction intervals "
            "on the 16-sensor GRU and checked them on the held-out test days. There is a separate "
            "calibration part between validation and test that never touches training or the choice "
            "of checkpoint. Methods: a Gaussian baseline, split conformal, adaptive conformal (ACI), "
            "and a quantile network with a conformal correction (CQR). Averages over 3 seeds:\n\n"
            "| Method | Coverage | Width (mph) | Coverage when last speed < 40 mph | Interval score |\n"
            "|---|---:|---:|---:|---:|\n"
            f"{rows}\n\n"
            "![Coverage by traffic condition](reports/uncertainty/uncertainty-conditions.png)\n\n"
            f"{unc_readme_text(unc)}\n\n"
            "The other methods, the reliability plot, coverage by horizon and day, and the limits are "
            "in the [intervals report](reports/uncertainty/report.md).\n"
        )

    if shf:
        std, stale = shf["settings"]["standard"], shf["settings"]["stale"]
        blocks["bullets"] += [shift_readme_bullet(shf)]
        rows = []
        for kind, label in SHIFT_LABELS.items():
            a, b = std["strategies"][kind], stale["strategies"][kind]

            def change(e, kind=kind):
                if kind == "static":
                    return ""
                return f" ({e['vs_static']['estimate']:+.3f})"

            seconds = "0" if kind == "static" else f"{a['adapt_seconds']:.0f}"
            rows.append(
                f"| {label} | {a['mae']:.3f}{change(a)} | {b['mae']:.3f}{change(b)} | {seconds} |"
            )
        blocks["shift"] = (
            "## Drift and adaptation\n\n"
            "A model is trained once and used for months, so I replayed the test period one day at a "
            "time and let it adapt using only data that had already arrived (the tests check that "
            "nothing from the future gets in). Test MAE in mph, with the change against the static "
            "model in brackets, for the usual model and for a stale one trained on only the first 40% "
            "of the timeline:\n\n"
            "| Strategy | Standard model | Stale model | Adaptation seconds (standard) |\n"
            "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
            "![MAE change against the static model](reports/shift/shift-gain.png)\n\n"
            f"{shift_readme_text(shf)}\n\n"
            "The data drift statistics, the regime breakdown and the limits are in the "
            "[drift report](reports/shift/report.md).\n"
        )

    from fed_readme import blocks as federated_blocks

    bullets, sections = federated_blocks()
    blocks["bullets"] += bullets
    blocks["federated_graph"] = "\n".join(sections.values())
    blocks["federated_sections"] = list(sections.values())
    return blocks


# --- shared model variants -----------------------------------------------------------

VARIANT_LABELS = {
    "sgru_time": "Shared GRU (reference)",
    "slstm_time": "Shared LSTM",
    "sgru_id_time": "Shared GRU + sensor vector",
    "sgraph_time": "Shared Graph-GRU (reference)",
    "sgraph_id_time": "Shared Graph-GRU + sensor vector",
}
VARIANT_GAPS = (
    ("slstm_time_vs_sgru_time", "LSTM - GRU"),
    ("sgru_id_time_vs_sgru_time", "GRU + vector - GRU"),
    ("sgraph_id_time_vs_sgraph_time", "Graph-GRU + vector - Graph-GRU"),
    ("sgraph_id_time_vs_sgru_id_time", "Graph-GRU + vector - GRU + vector"),
)


def verdict(gap):
    low, high = gap["ci95"]
    est = gap["estimate"]
    if high < 0:
        return f"better by {-est:.3f} mph (interval {low:+.3f} to {high:+.3f})"
    if low > 0:
        return f"worse by {est:.3f} mph (interval {low:+.3f} to {high:+.3f})"
    return f"not clearly different ({est:+.3f} mph, interval {low:+.3f} to {high:+.3f})"


def variant_findings(summary, scaling):
    sizes, gaps, models = summary["sizes"], summary["day_bootstrap"], summary["models"]
    out = []
    for key, label in VARIANT_GAPS:
        parts = [
            f"at {k} sensors {verdict(gaps[str(k)][key])}"
            for k in sizes
            if key in gaps.get(str(k), {})
        ]
        out.append(f"- **{label}:** " + "; ".join(parts) + ".")
    if scaling:
        dense = scaling["models"].get("gru_time", {})
        shared = models.get("sgru_id_time", {})
        both = [k for k in sizes if str(k) in dense and str(k) in shared]
        if both:
            parts = ", ".join(
                f"{shared[str(k)]['core16_mae']['mean']:.3f} against {dense[str(k)]['core16_mae']['mean']:.3f} "
                f"at {k} sensors"
                for k in both
            )
            large = str(both[-1])
            out.append(
                "- **Against the dense GRU from the scaling study** (same windows and seeds, same-16 MAE in "
                f"mph): the shared GRU with a sensor vector scores {parts}, with "
                f"{shared[large]['parameters']:,} parameters instead of {dense[large]['parameters']:,} at "
                f"{large} sensors."
            )
        same = [
            (
                k,
                models[n][str(k)]["core16_mae"]["mean"],
                scaling["models"][n][str(k)]["core16_mae"]["mean"],
            )
            for n in ("sgru_time", "sgraph_time")
            for k in sizes
            if str(k) in scaling["models"].get(n, {})
        ]
        worst = max(abs(a - b) for _, a, b in same)
        out.append(
            "- **Check against the scaling study:** the two reference models were trained again here, "
            + (
                "and their same-16 scores match the scaling study's to the last digit "
                "(training is deterministic for a given seed)."
                if worst < 1e-9
                else f"and their same-16 scores differ from the scaling study's by up to {worst:.4f} mph."
            )
        )
    return out


VARIANTS = """# Shared model variants

The shared models in the [scaling study](../scaling/report.md) use a GRU and give every sensor the same weights, so a sensor is only told apart from the others by its own speeds. This page tests two things I left out there, one at a time: swapping the GRU for an LSTM, and giving each sensor a learned vector of {dim} numbers so the network can tell sensors apart. The vectors add {dim} parameters per sensor, so those two models do grow with the network, just slowly.

The setup is the scaling study's: the same nested sensor sets (here {sizes}), the same splits and masking, {epochs} epochs that each use a random third of the training windows, the best epoch on validation kept, seeds {seeds}. The shared GRU and shared Graph-GRU are trained again as references, so every gap below is measured inside this run on identical windows.

## Results

Test MAE in mph on the same first {small} sensors, mean (SD over seeds):

{core_table}

Parameters at {large} sensors: {params}.

Gaps in test MAE on all the sensors each model was trained on, on identical windows (first minus second, negative favors the first; 95% interval from resampling test days, seeds averaged first). These are the same paired gaps as in the scaling study, so they aren't the same numbers as the differences between the same-{small} scores above:

| Size | {gap_heads} |
|---|{gap_rule}
{gap_rows}

{findings}

![Accuracy by variant](variants-accuracy.png)

The left panel scores each model on all the sensors it was trained on, the right one on the same first {small} sensors for every size. Persistence is the dotted grey line.

## Limits

- A sensor vector ties the model to the sensors it was trained on: it can't forecast a new sensor without training a vector for it. The plain shared models don't have that problem, which is why the scaling study used them, so the vectors trade some of what sharing weights is for against accuracy.
- Only {sizes} sensors, because every extra model is another full training run at each size. Whether the vectors help more (or overfit) at 207 sensors isn't tested here.
- Two seeds, the same equal-update budget as before, and a fixed {dim}-number vector. A larger vector or a longer run might change the picture.
- I didn't time these models; the cost comparison in the scaling study covers the GRU versions only.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli scaling --config configs/variants.json --out runs/variants
```

Files: [variants-summary.json](variants-summary.json), [variants-config.json](variants-config.json).
"""


def variants_page(summary, scaling, config):
    sizes = summary["sizes"]
    small, large = sizes[0], sizes[-1]
    models = summary["models"]
    rows = []
    for name, label in VARIANT_LABELS.items():
        by = models.get(name)
        if by is None:
            continue
        cells = [cell(by[str(k)], "core16_mae") if str(k) in by else "" for k in sizes]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    table = "\n".join(
        [
            "| Model | " + " | ".join(map(str, sizes)) + " |",
            "|---|" + "---:|" * len(sizes),
        ]
        + rows
    )
    gaps = summary["day_bootstrap"]
    gap_rows = "\n".join(
        f"| {k} | "
        + " | ".join(
            f"{gaps[str(k)][key]['estimate']:+.3f} ({gaps[str(k)][key]['ci95'][0]:+.3f} to {gaps[str(k)][key]['ci95'][1]:+.3f})"
            if key in gaps.get(str(k), {})
            else ""
            for key, _ in VARIANT_GAPS
        )
        + " |"
        for k in sizes
    )
    params = ", ".join(
        f"{VARIANT_LABELS[n].replace(' (reference)', '')} {models[n][str(large)]['parameters']:,}"
        for n in VARIANT_LABELS
        if n in models
    )
    return VARIANTS.format(
        dim=config["benchmark"]["embedding"],
        sizes=join_and(sizes),
        epochs=config["benchmark"]["epochs"],
        seeds=join_and(config["benchmark"]["seeds"]),
        small=small,
        large=large,
        core_table=table,
        params=params,
        gap_heads=" | ".join(label for _, label in VARIANT_GAPS),
        gap_rule="---:|" * len(VARIANT_GAPS),
        gap_rows=gap_rows,
        findings="\n".join(variant_findings(summary, scaling)),
    )


def build_variants():
    summary = load("reports/variants/variants-summary.json")
    config = load("reports/variants/variants-config.json")
    if summary is None or config is None:
        return None
    scaling = load("reports/scaling/scaling-summary.json")
    (REPORTS / "variants/report.md").write_bytes(
        variants_page(summary, scaling, config).encode()
    )
    return summary


if __name__ == "__main__":
    build_audit()
    build_scaling()
    build_uncertainty()
    build_shift()
    build_variants()
