"""Report page for the held-out sensor experiment.

Every number in the text comes from the saved summary, and a comparison is only called
better or worse when the interval over test days excludes zero.
"""

import shutil
import sys

from fed_pages import ROOT, gap, load, mae, plural, seed_phrase, versus

sys.path.insert(0, str(ROOT / "src"))

from tracast.inductive import PAIRS, TARGETS  # noqa: E402

ORDER = [
    "persistence",
    "ha_first_days",
    "ha_all_train",
    "sgru_zero_shot",
    "sgraph_zero_shot",
    "sgru_id_zero",
    "sgru_id_mean",
    "sgru_id_ridge",
    "sgru_id_fitted",
    "oracle_sgru",
    "oracle_sgraph",
    "oracle_sgru_id",
]


def table(summary):
    rows = []
    for name in ORDER:
        m = summary["methods"].get(name)
        if m is None:
            continue
        rows.append(f"| {m['label']} | {mae(m)} | {m['rmse']['mean']:.3f} |")
    return "\n".join(["| Method | MAE (SD) | RMSE |", "|---|---:|---:|"] + rows)


def findings(summary):
    """Three short paragraphs; every comparison is also in the gap table below them."""

    def get(first, second):
        return summary["paired_day_bootstrap"].get(f"{first}_vs_{second}")

    def sentence(first, second, subject, other):
        g = get(first, second)
        return versus(g, subject, other, interval=False) + "." if g else ""

    zero_shot = [
        sentence(
            "sgraph_zero_shot",
            "oracle_sgraph",
            "On a sensor it never saw, the shared Graph-GRU",
            "the same model trained on that sensor",
        ),
        sentence(
            "sgru_zero_shot",
            "oracle_sgru",
            "The shared GRU",
            "the same model trained on that sensor",
        ),
        sentence(
            "sgraph_zero_shot",
            "sgru_zero_shot",
            "For a sensor the model never saw, the Graph-GRU",
            "the GRU without a graph",
        ),
    ]
    vector = [
        sentence(
            "sgru_id_fitted",
            "sgru_id_zero",
            "A vector fitted on the first week",
            "a zero vector",
        ),
        sentence(
            "sgru_id_fitted",
            "sgru_id_mean",
            "It",
            "the average vector",
        ),
        sentence(
            "sgru_id_ridge",
            "sgru_id_mean",
            "A vector predicted from descriptors",
            "the average vector",
        ),
        sentence(
            "sgru_id_fitted",
            "sgraph_zero_shot",
            "The fitted-vector GRU",
            "the zero-shot shared Graph-GRU",
        ),
        sentence(
            "sgru_id_fitted",
            "oracle_sgru_id",
            "It",
            "the same model with a vector learned from all the training data",
        ),
    ]
    reference = [
        sentence(
            "sgraph_zero_shot",
            "ha_first_days",
            "The zero-shot shared Graph-GRU",
            "a historical average from the first week",
        ),
        sentence(
            "sgraph_zero_shot",
            "persistence",
            "It",
            "persistence",
        ),
    ]
    paragraphs = [
        " ".join(s for s in group if s) for group in (zero_shot, vector, reference)
    ]
    return "\n\n".join(p for p in paragraphs if p)


def gap_table(summary):
    rows = []
    for first, second in PAIRS:
        g = summary["paired_day_bootstrap"].get(f"{first}_vs_{second}")
        if g is None:
            continue
        rows.append(f"| {TARGETS[first]} - {TARGETS[second]} | {gap(g)} |")
    return "\n".join(["| Comparison | Gap in MAE (mph) |", "|---|---:|"] + rows)


PAGE = """# Sensors the model never trained on

The [variants study](../variants/report.md) found that giving each sensor a small learned vector improves the shared GRU a lot. The catch is that the vector is a row in a table with one row per training sensor, so a sensor that isn't in the table has nothing to look up. The plain shared models have no per-sensor parameters, so a new sensor can simply be fed through. Here I test all of them on sensors they never trained on, and try to fix the vector models.

## Setup

- **Sensors.** The first {sensors} sensors of the graph ordering. One whole geographic area of {held_count} sensors ({holdout_rule} holdout, chosen from coordinates only) is held out; the other {trained} sensors train the models. For the held-out sensors the training period is untouched except for the first {days} days, which stand in for what a newly installed sensor has before it goes live.
- **What a new sensor is allowed to use.** Its first {days} days of readings, its location, and (for the graph model) its neighbors' current readings at forecast time. The first days set its input scaling and, for the fitted and predicted vectors, its vector. The fitted vector also picks its stopping epoch on the sensor's own validation rows; its test rows are only ever scored. A test checks that rewriting the sensor's later training rows changes none of this.
- **Models,** all with calendar inputs and the same training recipe as the earlier studies ({rounds} epochs that each use a random third of the training windows, best epoch on validation kept; {seeds_text}):
  - the shared GRU and the shared Graph-GRU, run zero-shot;
  - the sensor-vector GRU ({embedding} numbers per sensor) with a **zero vector**, the **average trained vector**, a vector **fitted** on the new sensors' first days with the network frozen (Adam, {fit_epochs} epochs), and a vector **predicted** by ridge regression from the sensor's speeds in those days (mean, SD, share below 40 mph) and its latitude and longitude.
- **References.** The same three models trained on every sensor (so they did see the held-out sensors' training rows), persistence, and the historical average, once from the first {days} days only (what a new sensor would have) and once from all its training rows.
- **Scoring.** Test MAE in mph on the held-out sensors only, over the same test windows for every method. {held_count} sensors, {days_test} test days.

## Results

{table}

{findings}

Gaps on identical windows (first minus second, negative favors the first; 95% interval from resampling test days, seeds averaged first):

{gap_table}

![Held-out MAE](holdout-accuracy.png)

![By horizon](holdout-horizons.png)

For comparison, the shared GRU, the shared Graph-GRU and the sensor-vector GRU score {trained_text} on the {trained} sensors they trained on.

## Limits

- {nseeds_text}, one held-out area ({held_count} sensors) and one city. A different area or a random set of sensors (the code supports `holdout: random`) would give different numbers, but I only ran the area.
- The fitted vector uses the new sensor's validation rows to pick its stopping epoch. A real deployment would have to carve those out of the first {days} days; the effect is small, but it does use data after the first {days} days.
- Only the GRU has a vector version here. The Graph-GRU with a vector exists in the benchmark, but I didn't run it on held-out sensors.
- The fitted vector is only as good as the first week: a sensor that is mostly missing then would get a poor one.
- Every model trained to an equal budget, not to convergence.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli holdout --config configs/holdout.json --out runs/holdout
```

Files: [summary.json](summary.json), [config.json](config.json) and one result file per seed.
"""


def trained_text(summary):
    values = [
        f"{summary['trained_sensors_mae'][kind]['mean']:.3f}"
        for kind in ("sgru", "sgraph", "sgru_id")
    ]
    return ", ".join(values[:-1]) + " and " + values[-1] + " mph"


def page(summary):
    cfg = summary["config"]
    held = summary["held_out"][0]
    return PAGE.format(
        sensors=cfg["sensors"],
        held_count=len(held),
        trained=cfg["sensors"] - len(held),
        holdout_rule=cfg["holdout"],
        days=cfg["few_shot_days"],
        rounds=cfg["rounds"],
        seeds_text=seed_phrase(summary["seeds"]),
        nseeds_text=plural(len(summary["seeds"]), "seed").capitalize(),
        embedding=cfg["embedding"],
        fit_epochs=cfg["fit_epochs"],
        days_test=next(iter(summary["paired_day_bootstrap"].values()))["days"],
        table=table(summary),
        findings=findings(summary),
        gap_table=gap_table(summary),
        trained_text=trained_text(summary),
    )


def build(source="runs/holdout", target="reports/holdout"):
    summary = load(ROOT / source / "summary.json")
    if summary is None:
        return None
    out = ROOT / target
    out.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "config.json", "launch.json"):
        if (ROOT / source / name).exists():
            shutil.copy2(ROOT / source / name, out / name)
    for path in (ROOT / source).glob("seed-*.json"):
        shutil.copy2(path, out / path.name)
    for png in (ROOT / source).glob("*.png"):
        shutil.copy2(png, out / png.name)
    (out / "report.md").write_bytes(page(summary).encode())
    return summary


if __name__ == "__main__":
    build()
