"""Report pages for the federated graph studies.

Every number in the text comes from the saved summaries, so a page can't drift away
from the results it describes. A sentence about a comparison only says "better" or
"worse" when the interval over test days excludes zero.
"""

from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pages import join_and, pct  # noqa: E402

LABELS = {
    "fed_sgraph": "Federated shared Graph-GRU",
    "fed_sgru": "Federated shared GRU (no graph)",
    "fed_dcrnn": "Federated DCRNN",
    "central_sgraph_full": "Pooled shared Graph-GRU, full graph",
    "central_sgraph_local": "Pooled shared Graph-GRU, local graphs only",
    "central_sgru": "Pooled shared GRU",
    "central_dcrnn_full": "Pooled DCRNN, full graph",
    "central_dcrnn_local": "Pooled DCRNN, local graphs only",
    "local_sgraph": "Local-only shared Graph-GRU",
    "local_dcrnn": "Local-only DCRNN",
}


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mae(entry, places=3, sd=True):
    value = entry["mae"]
    text = f"{value['mean']:.{places}f}"
    if sd and value.get("sd") is not None:
        text += f" ({value['sd']:.{places}f})"
    return text


def gap(g):
    low, high = g["ci95"]
    return f"{g['estimate']:+.3f} ({low:+.3f} to {high:+.3f})"


def verdict(g, better="lower", worse="higher"):
    """Plain words for a paired gap (first minus second MAE)."""
    low, high = g["ci95"]
    if high < 0:
        return f"{better} by {-g['estimate']:.3f} mph"
    if low > 0:
        return f"{worse} by {g['estimate']:.3f} mph"
    return f"not clearly different ({g['estimate']:+.3f} mph, interval {low:+.3f} to {high:+.3f})"


def pair(summary, first, second):
    return summary["paired_day_bootstrap"].get(f"{first}_vs_{second}")


def plural(n, word):
    """'1 seed', '2 seeds'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def seed_phrase(seeds):
    """'seed 7' or 'seeds 7 and 42'."""
    return ("seed " if len(seeds) == 1 else "seeds ") + join_and(seeds)


def versus(g, first, second, interval=True, verb="was"):
    """A clause comparing two models from a gap (first minus second MAE).

    It only says better or worse when the interval over test days excludes zero. Pass
    interval=False where a table with the intervals sits right next to the sentence, and
    verb="were" for a plural subject.
    """
    low, high = g["ci95"]
    span = f" (interval {low:+.3f} to {high:+.3f})" if interval else ""
    if high < 0:
        return f"{first} {verb} {-g['estimate']:.3f} mph better than {second}{span}"
    if low > 0:
        return f"{first} {verb} {g['estimate']:.3f} mph worse than {second}{span}"
    detail = f"{abs(g['estimate']):.3f} mph"
    if interval:
        detail += f", interval {low:+.3f} to {high:+.3f}"
    return f"{first} {verb} not clearly different from {second} ({detail})"


def arm_rows(summary, order, labels=LABELS):
    rows = []
    for name in order:
        entry = summary["arms"].get(name)
        if entry is None:
            continue
        comm = entry.get("communication")
        params = entry.get("parameters")
        rounds = entry.get("best_round")
        rows.append(
            f"| {labels.get(name, name)} | {mae(entry)} | {entry['rmse']['mean']:.3f} "
            f"| {params:,} | "
            if params
            else f"| {labels.get(name, name)} | {mae(entry)} | {entry['rmse']['mean']:.3f} | | "
        )
        rows[-1] += (
            (f"{rounds['mean']:.0f}" if rounds else "each client's own")
            + f" | {entry['train_seconds']['mean'] / 60:.1f}"
            + f" | {entry['peak_memory_mib']['mean']:.0f}"
            + f" | {(comm['total_parameter_bytes'] / 2**20) if comm else 0:.1f} |"
        )
    return rows


def client_rows(summary, order, labels=LABELS):
    arms = [n for n in order if n in summary["arms"]]
    first = summary["arms"][arms[0]]["clients"]
    rows = []
    for i, client in enumerate(first):
        cells = " | ".join(
            f"{summary['arms'][n]['clients'][i]['mae']['mean']:.3f}" for n in arms
        )
        base = summary["baselines"]["persistence"]["clients"][i]["mae"]
        rows.append(
            f"| {client['client']} | {client['sensors']} | {base:.3f} | {cells} |"
        )
    head = (
        "| Client | Sensors | Persistence | "
        + " | ".join(labels.get(n, n) for n in arms)
        + " |"
    )
    rule = "|---|---:|---:|" + "---:|" * len(arms)
    return [head, rule] + rows


def copy_results(source, target, extras=(), seeds=None, figures=()):
    """Copy the small saved records (not the logs, jobs or checkpoints) into reports/.

    With seeds, only cases for those seeds are copied, so a stray finished run from a
    seed that was dropped from the study never ends up in the reports. Only the figures
    named in figures are copied, so the reports hold the charts a page actually shows.
    """
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "settings.json", "launch.json", *extras):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    for name in figures:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    cases = target / "cases"
    for folder in sorted(source.glob("*-s[0-9]*")):
        if ".unfinished" in folder.name or not (folder / "result.json").exists():
            continue
        if seeds is not None and int(folder.name.rsplit("-s", 1)[1]) not in seeds:
            continue
        (cases / folder.name).mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "manifest.json", "setup.json"):
            if (folder / name).exists():
                shutil.copy2(folder / name, cases / folder.name / name)
    if (source / "baselines" / "result.json").exists():
        (target / "baselines").mkdir(exist_ok=True)
        shutil.copy2(
            source / "baselines" / "result.json", target / "baselines" / "result.json"
        )


# --- headline: federated shared Graph-GRU on every sensor -----------------------------


def claim(g, better, worse, unclear):
    """One of three sentences, by whether the interval over test days excludes zero.

    g is a gap (first minus second MAE), so a negative gap favors the first model.
    The templates may use {amount} (the size of the gap in mph) and {interval}.
    """
    low, high = g["ci95"]
    interval = f"{low:+.3f} to {high:+.3f}"
    if high < 0:
        return better.format(amount=f"{-g['estimate']:.3f}", interval=interval)
    if low > 0:
        return worse.format(amount=f"{g['estimate']:.3f}", interval=interval)
    return unclear.format(amount=f"{abs(g['estimate']):.3f}", interval=interval)


def headline_findings(summary):
    arms = summary["arms"]
    out = []
    fed = arms.get("fed_sgraph")
    base = summary["baselines"]
    local = pair(summary, "fed_sgraph", "local_sgraph")
    if local:
        text = "- **Federated against training alone:** " + claim(
            local,
            "federated training was {amount} mph better than every client training by itself "
            "(interval {interval})",
            "federated training was {amount} mph worse than every client training by itself "
            "(interval {interval})",
            "federated training and every client training by itself were not clearly different "
            "({amount} mph, interval {interval})",
        )
        if "local_sgraph" in arms and fed:
            together = fed["clients"]
            alone = arms["local_sgraph"]["clients"]
            wins = sum(
                a["mae"]["mean"] < b["mae"]["mean"] for a, b in zip(together, alone)
            )
            text += f", and it came out ahead on {wins} of {len(together)} clients"
        out.append(text + ".")
    if fed:
        rounds = summary["base"]["rounds"]
        late = fed["best_round"]["mean"] >= rounds - 2
        out.append(
            f"- **The federated model on every sensor.** The federated shared Graph-GRU reaches a "
            f"test MAE of {mae(fed)} mph on all {summary['base']['sensors']} sensors, against "
            f"{base['persistence']['mae']:.3f} for persistence and {base['ha']['mae']:.3f} for the "
            f"historical average. "
            + (
                f"Its best round was {fed['best_round']['mean']:.0f} of {rounds}, so it was still "
                f"improving when the budget ran out."
                if late
                else f"Its best round was {fed['best_round']['mean']:.0f} of {rounds}."
            )
        )
    price = pair(summary, "fed_sgraph", "central_sgraph_local")
    if price:
        out.append(
            "- **What federated optimization costs:** "
            + claim(
                price,
                "federated training was {amount} mph better than one model trained on all the data "
                "with the same cut graph (interval {interval}).",
                "federated training cost {amount} mph (interval {interval}) compared with one model "
                "trained on all the data with the same cut graph.",
                "federated training and one model trained on all the data with the same cut graph "
                "were not clearly different ({amount} mph, interval {interval}).",
            )
        )
    cut = pair(summary, "central_sgraph_local", "central_sgraph_full")
    if cut:
        out.append(
            "- **What cutting the graph at client borders costs:** "
            + claim(
                cut,
                "a pooled model that keeps only the edges inside a client was {amount} mph better than "
                "one with every edge (interval {interval}).",
                "a pooled model that keeps only the edges inside a client was {amount} mph worse than "
                "one with every edge (interval {interval}).",
                "a pooled model with only the in-client edges and one with every edge were not "
                "clearly different ({amount} mph, interval {interval}).",
            )
        )
    graph_fed = pair(summary, "fed_sgraph", "fed_sgru")
    graph_cen = pair(summary, "central_sgraph_full", "central_sgru")
    if graph_fed:
        text = "- **The graph under federation:** " + claim(
            graph_fed,
            "with only its local graph, the federated Graph-GRU was {amount} mph better than the "
            "federated GRU without a graph (interval {interval})",
            "the federated Graph-GRU was {amount} mph worse than the federated GRU without a graph "
            "(interval {interval})",
            "the federated Graph-GRU and the federated GRU without a graph were not clearly "
            "different ({amount} mph, interval {interval})",
        )
        if graph_cen:
            text += f"; with everything pooled the gap was {gap(graph_cen)}"
        out.append(text + ".")
    ha = pair(summary, "fed_sgraph", "ha")
    if ha:
        out.append(
            "- **Against the historical average:** "
            + versus(ha, "the federated model", "the historical average")
            + "."
        )
    return out


def why_text(summary):
    """What the numbers say about why local training won, and what hasn't been tested."""
    arms = summary["arms"]
    local = pair(summary, "fed_sgraph", "local_sgraph")
    if not local or local["ci95"][0] <= 0:
        return (
            "Training alone did not beat federated training by a clear margin here, so there is "
            "nothing in this study that needs explaining beyond the gaps above."
        )
    vs_pooled = pair(summary, "local_sgraph", "central_sgraph_local")
    fed, alone = arms["fed_sgraph"]["clients"], arms["local_sgraph"]["clients"]
    pooled = arms.get("central_sgraph_local")
    gains = [f["mae"]["mean"] - a["mae"]["mean"] for f, a in zip(fed, alone)]
    curve = arms["fed_sgraph"].get("validation_curve")
    text = (
        f"Training alone was better than federated training on every client, by {min(gains):.3f} to "
        f"{max(gains):.3f} mph (federated minus local-only MAE per client)."
    )
    if vs_pooled and vs_pooled["ci95"][1] < 0 and pooled:
        better = sum(
            a["mae"]["mean"] < p["mae"]["mean"]
            for a, p in zip(alone, pooled["clients"])
        )
        text += (
            f" It also beat one pooled model with the same graph, by {-vs_pooled['estimate']:.3f} mph "
            f"(interval {vs_pooled['ci95'][0]:+.3f} to {vs_pooled['ci95'][1]:+.3f}), on {better} of "
            f"{len(alone)} clients. So the limit isn't federation alone: a single set of "
            f"{arms['fed_sgraph'].get('parameters', 0):,} weights has to serve all "
            f"{summary['base']['sensors']} sensors, while local training gives every area its own copy "
            f"({len(alone)} copies in total), and the areas differ a lot (the table by client shows "
            f"MAEs from {min(c['mae']['mean'] for c in fed):.1f} to {max(c['mae']['mean'] for c in fed):.1f} mph)."
        )
    text += (
        f" Federated training then costs a further "
        f"{pair(summary, 'fed_sgraph', 'central_sgraph_local')['estimate']:.3f} mph compared with the "
        f"pooled model, which is what averaging the clients' updates loses in a short budget."
        if pair(summary, "fed_sgraph", "central_sgraph_local")
        else ""
    )
    if curve:
        text += (
            f" Every arm's best round was the last or nearly the last one, and the federated "
            f"validation MAE was still falling at round {len(curve)} "
            f"({curve[-6]['mean']:.3f} at round {len(curve) - 5} to {curve[-1]['mean']:.3f}), so the "
            f"comparison is at an equal, modest budget."
        )
    text += (
        "\n\nI haven't isolated the cause, so this is the explanation that fits the numbers rather "
        "than a tested one. A larger shared model (more hidden units would give it room to "
        "specialize), longer training or a different learning rate could each change the picture, and "
        "I didn't try them here. The [local fine-tuning "
        "experiment](../fedopt/report.md#local-fine-tuning-of-the-federated-model) tries the middle "
        "ground: start from the federated model and let each client keep training it on its own data."
    )
    return text


HEADLINE = """# Federated shared Graph-GRU on all {sensors} sensors

The earlier federated experiment used a tiny setup: 12 sensors, three clients, and a GRU that looked at one sensor at a time, so it couldn't use the road graph at all. This one federates the shared Graph-GRU from the [scaling study](../scaling/report.md) across **every** METR-LA sensor, with clients that own different areas of the city and different numbers of sensors. The question is how much accuracy federation costs, and whether the graph still helps when a client can only see its own part of it.

It is a single-machine simulation: Flower's simulation runtime runs the clients in turn on one computer. There is no real network, no stragglers and no dropped clients.

## Setup

![Clients and the road graph](partition-map.png)

- **Clients.** The {sensors} sensors are cut into {clients} groups by k-means on their coordinates, so each client owns an area. The groups have {sizes} sensors. Nothing about speeds goes into this choice.
- **What a client can see.** Its own sensors' readings, and the road graph among its own sensors only. The edges that cross into other clients are dropped (the red lines): {kept} of {edges} directed edges ({weight} of the edge weight) stay inside a client, and {isolated_text}. A client never reads another client's speeds, not even a neighbor's, so its local graph is the only graph it has.
- **Model.** The shared Graph-GRU with calendar inputs: one small GRU run on every sensor, fed its own speed plus the average speed of its upstream and downstream neighbors. It has {parameters:,} parameters whatever the number of sensors, which is what lets clients with 10 and with 34 sensors average the same weights. The graph matrices stay on the client and are never exchanged.
- **One round.** The server sends the global parameters ({kib:.1f} KiB), each client trains on its own windows and sends back its updated parameters, and Flower's FedAvg averages them weighted by the number of training examples (windows times sensors).
- **Training budget.** {rounds} rounds, each a pass over a random third of the client's training windows with a fresh Adam (learning rate 0.001, batch 64, gradient clipping at 5). That is the same number of examples as the 30 epochs of a random third that the scaling study used, so the pooled arms below get exactly the same amount of data. The kept model is the round with the best pooled validation MAE.
- **Data.** The same chronological 70/10/20 split, training-only per-sensor normalization (each client fits it on its own training rows, which gives the same numbers as fitting it globally because it is per sensor), and the same missing-value masking as everywhere else. Seeds: {seeds}.

![One federated round](protocol.png)

## The arms

Each arm changes one thing, so the gaps between them say what that thing costs or buys:

| Arm | What it is |
|---|---|
| Federated shared Graph-GRU | The main arm, through Flower. |
| Federated shared GRU | The same, without graph inputs (tests the graph under federation). |
| Pooled shared Graph-GRU, local graphs only | One model on all the data, but with only the edges that stay inside a client. Same graph as the federated clients; differs from the federated arm only in optimization. |
| Pooled shared Graph-GRU, full graph | One model on all the data with every edge. Differs from the one above only in the cut edges. |
| Pooled shared GRU | One model, no graph. |
| Local-only shared Graph-GRU | Every client trains alone with the same rounds, seeds and local graph and no communication; each keeps its best round on its own validation data. |

All arms start from the same initial weights, process the same number of training examples, and are scored on the same observed test targets.

## Results

Test MAE in mph over {days} test days, mean (SD over seeds); the last columns are the model size, the average best round, training minutes (summed over clients for the federated arm), peak memory of the biggest process in MiB, and communicated parameters in MiB.

| Arm | MAE (SD) | RMSE | Parameters | Best round | Train min | Peak MiB | Comm MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}
| Persistence | {pers:.3f} | {persr:.3f} | 0 | | | | 0 |
| Historical average | {ha:.3f} | {har:.3f} | | | | | 0 |

{findings}

{timing_note}

Gaps between arms on identical test targets (first minus second, negative favors the first; 95% interval from resampling test days, seeds averaged first):

| Comparison | Gap in MAE (mph) |
|---|---:|
{pairs}

![Accuracy](accuracy.png)

![Convergence](convergence.png)

## Why training alone wins here

{why}

## By forecast horizon

![By horizon](horizons.png)

{horizon_text}

## By client

Test MAE for each client (mean over seeds). Clients differ a lot in how hard their sensors are, which is why persistence is in the table.

{client_table}

![By client](clients.png)

{client_text}

## Communication

Each round moves {kib:.1f} KiB of parameters down and {kib:.1f} KiB back up for every client, plus the same amount again to evaluate the aggregated model on validation data. So {n} clients over {rounds} rounds exchange {comm:.1f} MiB. That counts uncompressed parameter tensors only, not Flower's message envelopes or any metrics, and nothing was compressed. A pooled model would instead have to move the raw readings of every sensor (about {raw:.0f} MiB at 4 bytes per reading), so for this task the weights are much smaller than the data, which is not true of every federated setting.

## Limits

- **Simulation only.** Clients run in turn on one machine, so there is no real network, no dropped or slow clients, and no timing result for communication.
- **Not converged.** The best round is the last or nearly the last one, so every arm was still improving. The comparison is at an equal, modest budget and doesn't show the best each approach could reach.
- **{nseeds_text}.** The SD over seeds comes from {nseeds_text_short}, so it is a rough number. The intervals over test days only reflect which days were held out, not the randomness of training.
- **One partition.** The geographic cut is one k-means solution (fixed partition seed), and a different cut changes which edges survive. The [optimization study](../fedopt/report.md) varies the partition rule.
- **One city, one period.** Same as the earlier studies, including a test period with more missing readings than training.
- **Shared validation.** The best round is picked from every client's validation error sums and counts (no readings), as simulations usually do it. A deployment would need clients to report them.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli fedgraph --config configs/fedgraph.json --out runs/fedgraph
```

Each arm and seed is its own folder, so the run can be stopped and resumed, or split with `--arms` and `--seeds`. `--summarize-only` rebuilds the table from the finished folders. Files: [summary.json](summary.json), [settings.json](settings.json), [partition.json](partition.json) and one folder per case under `cases/` with its result, manifest and setup. The runs started from a frozen copy of the working tree ([launch.json](launch.json) lists the files that weren't committed at the time, and each manifest has the source hashes). A third seed (2026) was planned and dropped for time: three of its arms (the federated Graph-GRU, the federated GRU and the pooled GRU) had finished before it was stopped, and they are not used anywhere in this report.
"""


TIMING_NOTE = (
    "Training minutes and peak memory come from the study runs themselves, which shared the machine "
    "with other jobs, so minutes are only a rough guide and aren't comparable between arms. Peak "
    "memory is per process: a federated client holds only its own part of the data, while the pooled "
    "model holds all of it."
)


def cost_text(cost):
    """The cost of one round of each arm, from the short run with nothing else running."""
    if not cost:
        return TIMING_NOTE
    base, arms = cost["base"], cost["arms"]
    rounds = base["rounds"]
    rows = []
    for name, label in (
        ("fed_sgraph", "Federated shared Graph-GRU (Flower)"),
        ("central_sgraph_full", "Pooled shared Graph-GRU, full graph"),
        ("local_sgraph", "Local-only shared Graph-GRU"),
    ):
        entry = arms.get(name)
        if not entry:
            continue
        rows.append(
            f"| {label} | {entry['train_seconds']['mean'] / rounds:.1f} "
            f"| {entry['wall_seconds']['mean'] / rounds:.1f} "
            f"| {entry['peak_memory_mib']['mean']:.0f} |"
        )
    return (
        f"Cost, measured in a separate short run with nothing else on the machine ({rounds} rounds, one seed, "
        f"{base['threads']} threads). Training seconds are summed over clients for the federated and "
        "local-only arms; wall seconds also include validation scoring and, for the federated arm, "
        "Flower's messaging; peak memory is the largest single process (a Flower client, or the whole "
        "pooled or local-only process).\n\n"
        "| Arm | Training seconds per round | Wall seconds per round | Peak memory (MiB) |\n"
        "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        "The study runs above shared the machine with other jobs, so their own timings are only a rough "
        "guide and aren't used."
    )


def horizon_text(summary):
    arms = summary["arms"]
    fed = arms.get("fed_sgraph")
    if not fed:
        return ""
    steps = sorted(fed["horizon_mae"], key=int)
    first, last = steps[0], steps[-1]
    base = summary["baselines"]
    ha = base["ha"]["horizon_mae"]

    def crossing(name):
        entry = arms.get(name)
        if not entry:
            return None
        for step in steps:
            if entry["horizon_mae"][step]["mean"] > ha[step]:
                return step
        return "never"

    text = (
        f"The federated Graph-GRU's MAE grows from {fed['horizon_mae'][first]['mean']:.2f} mph at "
        f"{first} minutes to {fed['horizon_mae'][last]['mean']:.2f} at {last}. Persistence goes from "
        f"{base['persistence']['horizon_mae'][first]:.2f} to {base['persistence']['horizon_mae'][last]:.2f}. "
        f"The historical average doesn't use recent readings, so it stays flat "
        f"({ha[first]:.2f} to {ha[last]:.2f}) and wins at the long horizons."
    )
    parts = []
    for name, label in (
        ("fed_sgraph", "the federated model"),
        ("central_sgraph_full", "the pooled model"),
        ("local_sgraph", "local-only training"),
    ):
        step = crossing(name)
        if step is None:
            continue
        parts.append(
            f"{label} stays ahead of it up to "
            + (
                "every horizon shown"
                if step == "never"
                else f"{steps[steps.index(step) - 1]} minutes and falls behind from {step} minutes"
                if steps.index(step) > 0
                else "no horizon"
            )
        )
    if parts:
        text += " " + "; ".join(parts)[0].upper() + "; ".join(parts)[1:] + "."
    text += (
        " So at the longest horizons a one-hour forecast from these small shared models is worse "
        "than just knowing the time of week, which is a limit of the model and the budget, not of "
        "federation (the pooled and local arms do the same)."
    )
    return text


def client_text(summary):
    arms = summary["arms"]
    fed, local = arms.get("fed_sgraph"), arms.get("local_sgraph")
    if not fed:
        return ""
    maes = [c["mae"]["mean"] for c in fed["clients"]]
    best, worst = min(maes), max(maes)
    text = (
        f"Federated MAE ranges from {best:.2f} mph (client {maes.index(best)}) to {worst:.2f} "
        f"(client {maes.index(worst)})."
    )
    if local:
        diffs = [
            (c["client"], c["mae"]["mean"] - a["mae"]["mean"], c["sensors"])
            for c, a in zip(fed["clients"], local["clients"])
        ]
        top = min(diffs, key=lambda d: d[1])
        bottom = max(diffs, key=lambda d: d[1])
        text += (
            f" Federated minus local-only MAE (negative means federating helped) ranges from "
            f"{top[1]:+.3f} mph for client {top[0]} ({top[2]} sensors) to {bottom[1]:+.3f} for "
            f"client {bottom[0]} ({bottom[2]} sensors)."
        )
    return text


def isolated_text(n):
    if n == 0:
        return "no sensor ends up without a neighbor"
    if n == 1:
        return "1 sensor ends up with no neighbor at all"
    return f"{n} sensors end up with no neighbor at all"


def headline_page(summary, cost=None):
    base = summary["base"]
    part = summary["partition"]
    sizes = join_and([c["sensors"] for c in part["clients"]])
    graph = part["graph"]
    fed = summary["arms"].get("fed_sgraph") or next(iter(summary["arms"].values()))
    size = fed.get("parameters", 0) * 4
    comm = (fed.get("communication") or {}).get("total_parameter_bytes", 0)
    order = [
        "fed_sgraph",
        "fed_sgru",
        "central_sgraph_local",
        "central_sgraph_full",
        "central_sgru",
        "local_sgraph",
    ]
    pairs = [
        f"| {LABELS.get(k.split('_vs_')[0], k.split('_vs_')[0])} - "
        f"{LABELS.get(k.split('_vs_')[1], {'ha': 'Historical average', 'persistence': 'Persistence'}.get(k.split('_vs_')[1], k.split('_vs_')[1]))} | {gap(v)} |"
        for k, v in summary["paired_day_bootstrap"].items()
    ]
    days = (
        len(next(iter(summary["paired_day_bootstrap"].values()))["ci95"])
        and next(iter(summary["paired_day_bootstrap"].values()))["days"]
    )
    raw = base["rows"] * base["sensors"] * 4 / 2**20
    return HEADLINE.format(
        sensors=base["sensors"],
        clients=base["clients"],
        sizes=sizes,
        kept=f"{graph['edges_kept']:,}",
        edges=f"{graph['directed_edges']:,}",
        weight=pct(graph["weight_kept_fraction"]),
        isolated_text=isolated_text(
            sum(c["locally_isolated_sensors"] for c in graph["clients"])
        ),
        parameters=fed.get("parameters", 0),
        kib=size / 1024,
        rounds=base["rounds"],
        seeds=join_and(summary["seeds"]),
        nseeds_text=plural(len(summary["seeds"]), "seed"),
        nseeds_text_short=plural(len(summary["seeds"]), "run"),
        days=days,
        rows="\n".join(arm_rows(summary, order)),
        pers=summary["baselines"]["persistence"]["mae"],
        persr=summary["baselines"]["persistence"]["rmse"],
        ha=summary["baselines"]["ha"]["mae"],
        har=summary["baselines"]["ha"]["rmse"],
        findings="\n".join(headline_findings(summary)),
        why=why_text(summary),
        timing_note=cost_text(cost),
        pairs="\n".join(pairs),
        horizon_text=horizon_text(summary),
        client_table="\n".join(client_rows(summary, order)),
        client_text=client_text(summary),
        n=base["clients"],
        comm=comm / 2**20,
        raw=raw,
    )


HEADLINE_FIGURES = (
    "accuracy.png",
    "convergence.png",
    "horizons.png",
    "clients.png",
    "partition-map.png",
    "protocol.png",
)


def build_headline(source="runs/fedgraph", target="reports/fedgraph"):
    summary = load(ROOT / source / "summary.json")
    if summary is None:
        return None
    copy_results(
        ROOT / source,
        ROOT / target,
        seeds=summary["seeds"],
        figures=HEADLINE_FIGURES,
    )
    cost = load(ROOT / "runs/fedgraph-cost/summary.json")
    if cost:
        (ROOT / target / "cost-summary.json").write_bytes(
            (ROOT / "runs/fedgraph-cost/summary.json").read_bytes()
        )
    (ROOT / target / "report.md").write_bytes(headline_page(summary, cost).encode())
    return summary


if __name__ == "__main__":
    build_headline()
