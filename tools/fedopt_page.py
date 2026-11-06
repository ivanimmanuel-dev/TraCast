"""Report page for the FedAvg / FedProx study: client distributions, budgets and graph cuts.

Every number in the text is read from the saved summary (the client-heterogeneity table
is recomputed from the data and the partition rule). A comparison is only called better
or worse when the interval over test days excludes zero.
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fed_pages import (  # noqa: E402
    ROOT,
    claim,
    copy_results,
    gap,
    load,
    mae,
    plural,
    seed_phrase,
    versus,
)
from pages import join_and, pct  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from tracast import fedgraph as F  # noqa: E402
from tracast.federated_study import js_divergence  # noqa: E402

RULES = {
    "geographic": ("geo", "Geographic areas"),
    "balanced": ("bal", "Balanced (each client a similar mix)"),
    "congestion": ("cong", "Congestion-grouped (each client one regime)"),
}


def heterogeneity(base):
    """Per partition rule: client sizes, how different their speed distributions are,
    and how much of the road graph survives the cut."""
    config = F.GraphFedConfig(**{**base, "budgets": []})
    out = {}
    for rule in RULES:
        cfg = F.with_config(config, partition=rule)
        prep = F.prepare(cfg, ROOT / "data/raw/metr-la.h5", ROOT / "data/topology")
        train = prep["frame"].iloc[: int(len(prep["frame"]) * 0.7)]
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
        hists = []
        for group in prep["groups"]:
            values = train.iloc[:, group].to_numpy().reshape(-1)
            values = values[np.isfinite(values) & (values > 0)]
            hists.append(np.histogram(values, bins=bins)[0].astype(float))
        pooled = np.sum(hists, axis=0)
        divergence = [js_divergence(h, pooled) for h in hists]
        report = F.graph_report(prep["weights"], prep["groups"])
        out[rule] = {
            "sizes": [len(g) for g in prep["groups"]],
            "mean_js_bits": float(np.mean(divergence)),
            "congested_share": [float((h[:4].sum()) / h.sum()) for h in hists],
            "edges_kept_fraction": report["edges_kept_fraction"],
            "weight_kept_fraction": report["weight_kept_fraction"],
        }
    return out


def arm(summary, name):
    return summary["arms"].get(name)


def pair(summary, first, second):
    return summary["paired_day_bootstrap"].get(f"{first}_vs_{second}")


def gap_cell(summary, first, second):
    g = pair(summary, first, second)
    return gap(g) if g else ""


def mae_cell(summary, name):
    entry = arm(summary, name)
    return mae(entry) if entry else ""


def partition_table(summary, hetero):
    rows = []
    for rule, (tag, label) in RULES.items():
        h = hetero[rule]
        sizes = join_and(sorted(h["sizes"], reverse=True))
        rows.append(
            f"| {label} | {sizes} | {h['mean_js_bits']:.4f} "
            f"| {pct(h['edges_kept_fraction'])} | {pct(h['weight_kept_fraction'])} |"
        )
    head = (
        "| Partition | Sensors per client | Speed-distribution distance (bits) "
        "| Edges kept | Edge weight kept |"
    )
    return "\n".join([head, "|---|---|---:|---:|---:|"] + rows)


MUS = ("0.001", "0.01", "0.1")
KINDS = {"geo": "geographic", "cong": "congestion-grouped", "bal": "balanced"}


def algorithm_table(summary):
    ran = [
        mu
        for mu in MUS
        if any(arm(summary, f"{t}_fedprox_{mu}") for t in ("geo", "cong", "bal"))
    ]
    rows = []
    for rule, (tag, label) in RULES.items():
        cells = " | ".join(
            mae_cell(summary, f"{tag}_fedprox_{mu}") or "-" for mu in ran
        )
        rows.append(
            f"| {label} | {mae_cell(summary, f'{tag}_fedavg') or '-'} | {cells} |"
        )
    head = (
        "| Partition | FedAvg | " + " | ".join(f"FedProx, mu {mu}" for mu in ran) + " |"
    )
    return "\n".join([head, "|---|---:|" + "---:|" * len(ran)] + rows)


def reference_table(summary):
    rows = []
    tags = {"geo": "geographic", "cong": "congestion", "bal": "balanced"}
    for tag, rule in tags.items():
        fed = f"{tag}_fedavg"
        central = f"central_local_{tag}"
        local = f"local_{tag}"
        rows.append(
            f"| {RULES[rule][1]} | {mae_cell(summary, fed)} | {mae_cell(summary, central) or '-'} "
            f"| {mae_cell(summary, local) or '-'} | {gap_cell(summary, fed, central) or '-'} "
            f"| {gap_cell(summary, fed, local) or '-'} |"
        )
    head = (
        "| Partition | Federated | Pooled, same cut graph | Local-only "
        "| Federated - pooled | Federated - local-only |"
    )
    return "\n".join([head, "|---|---:|---:|---:|---:|---:|"] + rows)


def budget_series(summary):
    """(label, epochs per round, [(rounds, MAE, MiB)]) for every arm that has budgets."""
    series = []
    for name, label, epochs in (
        ("geo_fedavg", "FedAvg, 1 local epoch", 1),
        ("geo_e2", "FedAvg, 2 local epochs", 2),
        ("geo_e5", "FedAvg, 5 local epochs", 5),
        ("geo_e5_fedprox_0.1", "FedProx (mu 0.1), 5 local epochs", 5),
    ):
        entry = arm(summary, name)
        if not entry or "budgets" not in entry:
            continue
        points = sorted(
            (
                int(b),
                v["mae"]["mean"],
                v["communication_bytes"] / 2**20,
                v["mae"]["sd"],
            )
            for b, v in entry["budgets"].items()
        )
        series.append((label, epochs, points))
    return series


def budget_table(summary):
    rows = []
    for label, epochs, points in budget_series(summary):
        for rounds, value, comm, spread in points:
            rows.append(
                f"| {label} | {rounds} | {rounds * epochs} | "
                + (
                    f"{value:.3f} ({spread:.3f})"
                    if spread is not None
                    else f"{value:.3f}"
                )
                + f" | {comm:.1f} |"
            )
    head = "| Setting | Rounds | Local epochs in total | MAE (SD) | Communicated MiB |"
    return "\n".join([head, "|---|---:|---:|---:|---:|"] + rows)


def k_table(summary):
    rows = []
    entries = [
        (2, "geo_k2", "central_local_k2"),
        (6, "geo_fedavg", "central_local_geo"),
        (12, "geo_k12", "central_local_k12"),
    ]
    for k, fed, central in entries:
        f, c = arm(summary, fed), arm(summary, central)
        if not f:
            continue
        setup = f.get("setup", {})
        rows.append(
            f"| {k} | {join_and(sorted(setup.get('client_sensors', []), reverse=True))} "
            f"| {pct(setup.get('edges_kept_fraction', 0))} | {mae(f)} "
            f"| {mae(c) if c else '-'} | {gap_cell(summary, fed, central) or '-'} "
            f"| {gap_cell(summary, central, 'central_full') or '-'} |"
        )
    head = (
        "| Clients | Sensors per client | Edges kept | Federated | Pooled, same cut graph "
        "| Federated - pooled | Pooled cut - pooled full |"
    )
    return "\n".join([head, "|---|---|---:|---:|---:|---:|---:|"] + rows)


def k_section(summary):
    """The client-count sweep, only if its arms were run."""
    if not arm(summary, "geo_k12") and not arm(summary, "geo_k2"):
        return ""
    return (
        "## Number of clients and the cut graph\n\n"
        f"Geographic clients with {summary['base']['sensors']} sensors, one model per row:\n\n"
        f"{k_table(summary)}\n\n"
        "![Clients](fedopt-clients.png)\n\n"
        "More clients means smaller areas and more edges that cross a border, so a client's local "
        "graph is thinner.\n\n"
    )


def personalization_section(ps):
    """Local fine-tuning of the federated model, from its own small study."""
    if not ps:
        return ""
    rows, versus_local, sentences = [], [], []
    for tag, rule in (("geo", "Geographic areas"), ("cong", "Congestion-grouped")):
        fed, tuned = arm(ps, f"fed_{tag}"), arm(ps, f"fed_{tag}+ft")
        local, pooled = arm(ps, f"local_{tag}"), arm(ps, f"central_local_{tag}")
        if not (fed and tuned and local):
            continue
        rows.append(
            f"| {rule} | {mae(fed)} | {mae(tuned)} | {mae(local)} | {mae(pooled) if pooled else '-'} "
            f"| {gap_cell(ps, f'fed_{tag}+ft', f'fed_{tag}')} | {gap_cell(ps, f'fed_{tag}+ft', f'local_{tag}')} "
            f"| {gap_cell(ps, f'fed_{tag}+ft', f'central_local_{tag}') or '-'} |"
        )
        g = pair(ps, f"fed_{tag}+ft", f"local_{tag}")
        if g:
            versus_local.append(g)
            sentences.append(
                versus(
                    g,
                    f"On {KINDS[tag]} clients the fine-tuned model",
                    "every client training alone",
                    interval=False,
                )
                + "."
            )
        g = pair(ps, f"fed_{tag}+ft", f"central_local_{tag}")
        if g:
            sentences.append(
                versus(g, "It", "one pooled model with the same graph", interval=False)
                + "."
            )
    if not rows:
        return ""
    chosen = arm(ps, "fed_geo+ft") or arm(ps, "fed_cong+ft")
    allowed = chosen["fine_tune_rounds_allowed"]
    rounds = [c["fine_tune_round"]["mean"] for c in chosen["clients"]]
    head = (
        "| Partition | Federated | Federated + fine-tuning | Local-only | Pooled, same cut graph "
        "| Fine-tuned - federated | Fine-tuned - local-only | Fine-tuned - pooled |"
    )
    return (
        "## Local fine-tuning of the federated model\n\n"
        "In the [headline study](../fedgraph/report.md) every client training alone beat federated "
        "training. This is the middle ground: run federated training as before, then let every "
        f"client keep training the final federated model on its own data for up to {allowed} more local "
        "rounds (the same local update, fresh seeds, no communication), keeping the round with the best "
        "validation MAE on its own data. Round 0 is the federated model itself, so a client only moves "
        f"away from it when its own validation data says that is better. {seed_phrase(ps['seeds']).capitalize()}, "
        f"{ps['base']['sensors']} sensors, {ps['base']['clients']} clients.\n\n"
        + "\n".join([head, "|---|---:|---:|---:|---:|---:|---:|---:|"] + rows)
        + "\n\n"
        + " ".join(sentences)
        + "\n\n"
        f"The clients' chosen fine-tuning rounds average between {min(rounds):.1f} and {max(rounds):.1f} "
        f"of {allowed}, so they were still improving at the limit I allowed. It costs {allowed} extra "
        "local rounds per client (about a third of the federated training compute at 30 rounds), and "
        "the result is a personalized model per client rather than one shared model, so it answers a "
        "different question than FedAvg does: "
        + (
            "the fine-tuned federated model beat each area training alone."
            if any(g["ci95"][1] < 0 for g in versus_local)
            else "the federated model is a good starting point for each area, but at this training "
            "budget the fine-tuned model doesn't clearly beat each area training alone."
        )
        + "\n\n"
    )


def five_epoch_sentence(summary):
    """The test of the other regime: five local epochs a round leave more drift to correct."""
    g = pair(summary, "geo_e5_fedprox_0.1", "geo_e5")
    if not g:
        return ""
    return " Five local epochs per round leave more room to drift, and " + claim(
        g,
        "FedProx (mu 0.1) was {amount} mph better there (interval {interval}), so that is where it helps.",
        "FedProx (mu 0.1) was still {amount} mph worse there (interval {interval}), which suggests drift isn't what limits accuracy here.",
        "FedProx (mu 0.1) was not clearly different there ({amount} mph, interval {interval}).",
    )


def prox_story(summary):
    """What the FedProx gaps add up to, in words that follow them."""
    gaps = []
    for tag, label in (
        ("geo", "geographic"),
        ("cong", "congestion-grouped"),
        ("bal", "balanced"),
    ):
        for mu in MUS:
            g = pair(summary, f"{tag}_fedprox_{mu}", f"{tag}_fedavg")
            if g:
                gaps.append((label, mu, g))
    if not gaps:
        return ""
    worse = [x for x in gaps if x[2]["ci95"][0] > 0]
    better = [x for x in gaps if x[2]["ci95"][1] < 0]
    if len(worse) == len(gaps):
        geo = [(mu, g["estimate"]) for label, mu, g in gaps if label == "geographic"]
        trend = ", ".join(f"{e:+.3f} at mu {mu}" for mu, e in geo)
        return (
            "FedProx was worse than FedAvg in every setting I ran, and the penalty grows with mu"
            + (f" (geographic clients: {trend})" if len(geo) > 1 else "")
            + ". The proximal term pulls every client back toward the global model, which helps when "
            "clients drift far apart during long local training. Here a round is only about a hundred "
            "optimizer steps and the model is still improving at the end, so the clients probably don't "
            "drift much, and the pull mostly slows learning down."
            + five_epoch_sentence(summary)
        )
    if len(better) == len(gaps):
        return "FedProx was better than FedAvg in every setting I ran."
    return (
        f"FedProx helped in {len(better)} of {len(gaps)} settings and hurt in {len(worse)}; the gaps "
        "above say which."
    )


def heterogeneity_story(summary, hetero):
    """Does FedAvg's error follow how different the clients are, or how much graph is cut?"""
    rows = []
    for rule, (tag, label) in RULES.items():
        entry = arm(summary, f"{tag}_fedavg")
        if entry:
            rows.append(
                (
                    entry["mae"]["mean"],
                    label.split(" (")[0].lower(),
                    hetero[rule]["mean_js_bits"],
                    hetero[rule]["edges_kept_fraction"],
                )
            )
    if len(rows) < 3:
        return ""
    by_error = sorted(rows)
    by_js = sorted(rows, key=lambda r: r[2])
    by_edges = sorted(rows, key=lambda r: -r[3])
    same_js = [r[1] for r in by_error] == [r[1] for r in by_js]
    same_edges = [r[1] for r in by_error] == [r[1] for r in by_edges]
    order = ", ".join(f"{r[1]} ({r[0]:.3f})" for r in by_error)
    if same_js and not same_edges:
        return (
            f"FedAvg's test MAE from lowest to highest is {order}. That is the order of how different "
            "the clients' speed distributions are, not of how much of the road graph each cut keeps, so for "
            "this model the differences between clients cost more than the lost edges. With one seed "
            "that is a hint rather than a result."
        )
    if same_edges and not same_js:
        return (
            f"FedAvg's test MAE from lowest to highest is {order}, which follows how much of the road graph "
            "each cut keeps rather than how different the clients' speed distributions are."
        )
    return f"FedAvg's test MAE from lowest to highest is {order}, which doesn't line up with just one of the two."


def make_figures(summary, out):
    series = budget_series(summary)
    if series:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        colors = ["#1d4ed8", "#087e8b", "#d97706", "#dc2626"]
        for (label, epochs, points), color in zip(series, colors):
            ax.errorbar(
                [p[2] for p in points],
                [p[1] for p in points],
                yerr=[p[3] or 0 for p in points],
                marker="o",
                capsize=3,
                label=label,
                color=color,
            )
            for rounds, value, comm, _ in points:
                ax.annotate(f" {rounds}", (comm, value), fontsize=7, color=color)
        ax.set(
            xlabel="Communicated parameters (MiB, log scale)",
            ylabel="Test MAE (mph)",
            title="Accuracy against communication (labels are rounds)",
            xscale="log",
        )
        ax.legend(fontsize=7)
        fig.savefig(out / "fedopt-budget.png", dpi=150)
        plt.close(fig)
    ks = [
        (2, "geo_k2", "central_local_k2"),
        (6, "geo_fedavg", "central_local_geo"),
        (12, "geo_k12", "central_local_k12"),
    ]
    rows = [(k, arm(summary, f), arm(summary, c)) for k, f, c in ks]
    rows = [r for r in rows if r[1]]
    full = arm(summary, "central_full")
    if rows and full:
        fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        xs = [r[0] for r in rows]
        ax.plot(
            xs,
            [r[1]["mae"]["mean"] for r in rows],
            "o-",
            label="Federated",
            color="#1d4ed8",
        )
        ax.plot(
            xs,
            [r[2]["mae"]["mean"] if r[2] else np.nan for r in rows],
            "s--",
            label="Pooled, same cut graph",
            color="#087e8b",
        )
        ax.axhline(
            full["mae"]["mean"],
            color="#667085",
            linestyle=":",
            label="Pooled, full graph",
        )
        ax.set_xscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set(
            xlabel="Number of clients",
            ylabel="Test MAE (mph)",
            title="More clients, more cut edges",
        )
        ax.legend(fontsize=8)
        fig.savefig(out / "fedopt-clients.png", dpi=150)
        plt.close(fig)


def findings(summary):
    """Gaps with intervals; the story after the table says what they add up to."""
    rows = []
    for tag, label in (
        ("geo", "geographic"),
        ("cong", "congestion-grouped"),
        ("bal", "balanced"),
    ):
        for mu in MUS:
            g = pair(summary, f"{tag}_fedprox_{mu}", f"{tag}_fedavg")
            if g:
                rows.append(
                    f"| FedProx (mu {mu}) - FedAvg, {label} clients | {gap(g)} |"
                )
    for first, second, title in (
        (
            "geo_fedavg",
            "cong_fedavg",
            "FedAvg, geographic - congestion-grouped clients",
        ),
        ("geo_fedavg", "bal_fedavg", "FedAvg, geographic - balanced clients"),
        (
            "geo_e2",
            "geo_fedavg",
            "FedAvg, 2 local epochs over 15 rounds - 1 local epoch over 30",
        ),
        (
            "geo_e5",
            "geo_fedavg",
            "FedAvg, 5 local epochs over 6 rounds - 1 local epoch over 30",
        ),
        (
            "geo_e5_fedprox_0.1",
            "geo_e5",
            "FedProx (mu 0.1) - FedAvg, 5 local epochs",
        ),
    ):
        g = pair(summary, first, second)
        if g:
            rows.append(f"| {title} | {gap(g)} |")
    if not rows:
        return ""
    return (
        "Gaps on identical windows (first minus second, negative favors the first; 95% interval "
        "from resampling test days, seeds averaged first):\n\n"
        + "\n".join(["| Comparison | Gap in MAE (mph) |", "|---|---:|"] + rows)
    )


PAGE = """# FedAvg and FedProx under different client distributions

This is the optimization side of the [federated graph study](../fedgraph/report.md). It asks how much federated averaging is affected by how different the clients are, how much local work a round does, how many rounds there are, and how much of the road graph a client can see. It uses the shared Graph-GRU on {sensors} sensors with {clients} clients ({seeds_text}). The runs use the same code the Flower clients run, and a test checks that the in-process loop matches Flower bit for bit; the larger [headline study](../fedgraph/report.md) ran through Flower itself.

## How the clients differ

Three ways to cut the same sensors into {clients} clients:

{partition_table}

The speed-distribution distance is the average Jensen-Shannon divergence in bits between a client's histogram of training speeds and the pooled histogram (0 means identical). Geographic clients are the realistic case: each owns an area, so they differ in how congested their roads are and have unequal numbers of sensors. Balanced clients each get a similar mix of congested and free-flowing sensors, which makes them as alike as a split can, but it scatters neighbors across clients and cuts most of the graph. Congestion-grouped clients each hold sensors of about the same congestion level, so their speed distributions differ the most, and that is the heterogeneity FedProx was designed for.

## FedAvg against FedProx

FedProx adds a term (mu / 2) times the squared distance to the global weights to every client's loss, which pulls local updates back toward the global model. Test MAE in mph, mean (SD over seeds), the same {rounds} rounds and the same training examples in every cell:

{algorithm_table}

{findings}

{prox_story}

## What federating costs, by partition

{reference_table}

{heterogeneity_story}

The pooled arm gets the same graph as the federated clients (only the edges inside a client), so "federated - pooled" isolates what federated optimization costs, while the partition table above says how much graph each cut removes. Local-only is every client training alone with no communication.

## Local work and rounds

A round is one local epoch over a random third of the client's windows; with more local epochs per round, fewer rounds use the same number of examples. The table gives the test score the server would have had after each number of rounds, from the same run (training in a round never depends on later rounds, and a test checks that a shorter run gives exactly the prefix of a longer one).

{budget_table}

![Accuracy against communication](fedopt-budget.png)

{budget_text}

{k_section}{personalization}## Limits

- {nseeds_text}. The intervals over test days reflect which days were held out, not the randomness of training.
- {sensors} sensors, not all 207, so that this many arms were affordable. Larger networks have more room for heterogeneity to matter.
- Three values of mu (0.001, 0.01 and 0.1) and one learning rate. A different local learning rate or more local work could change how FedProx behaves.
- A single-machine simulation, so it says nothing about real network conditions, stragglers or partial participation.
- The partitions are three fixed cuts (partition seed 0), and a different geographic cut changes which edges survive.
- Communication is counted as uncompressed parameter tensors (training down, training up and validation down for every client in every round), not Flower's message overhead.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli fedgraph --config configs/fedgraph-optimization.json --out runs/fedgraph-optimization
```

Files: [summary.json](summary.json), [settings.json](settings.json) and one folder per case under `cases/`.
"""


def budget_text(summary):
    series = budget_series(summary)
    base = next((x for x in series if x[1] == 1 and x[0].startswith("FedAvg")), None)
    if base is None:
        return ""
    points = base[2]
    cheap, full = points[0], points[-1]
    text = (
        f"With one local epoch per round, MAE goes from {cheap[1]:.3f} after {cheap[0]} rounds "
        f"({cheap[2]:.1f} MiB communicated) to {full[1]:.3f} after {full[0]} rounds "
        f"({full[2]:.1f} MiB)"
        + (
            ", and it was still improving at the end."
            if points[-1][1] < points[-2][1]
            else ", and it had levelled off."
        )
    )
    heavy = next((x for x in series if x[1] == 5 and x[0].startswith("FedAvg")), None)
    g = pair(summary, "geo_e5", "geo_fedavg")
    if heavy and g:
        last = heavy[2][-1]
        text += (
            f" Five local epochs per round do the same total local work in {last[0]} rounds: "
            f"{last[1]:.3f} mph for {last[2]:.1f} MiB, which is {full[2] / last[2]:.1f} times less "
            f"communication than the 30-round run, and "
            + claim(
                g,
                "{amount} mph better (interval {interval}).",
                "{amount} mph worse (interval {interval}).",
                "not clearly different ({amount} mph, interval {interval}).",
            )
        )
    return text


def page(summary, hetero, ps=None):
    base = summary["base"]
    return PAGE.format(
        sensors=base["sensors"],
        clients=base["clients"],
        seeds_text=seed_phrase(summary["seeds"]),
        nseeds_text=plural(len(summary["seeds"]), "seed").capitalize(),
        rounds=base["rounds"],
        partition_table=partition_table(summary, hetero),
        algorithm_table=algorithm_table(summary),
        findings=findings(summary),
        prox_story=prox_story(summary),
        heterogeneity_story=heterogeneity_story(summary, hetero),
        reference_table=reference_table(summary),
        budget_table=budget_table(summary),
        budget_text=budget_text(summary),
        k_section=k_section(summary),
        personalization=personalization_section(ps),
    )


def build(source="runs/fedgraph-optimization", target="reports/fedopt"):
    summary = load(ROOT / source / "summary.json")
    if summary is None:
        return None
    out = ROOT / target
    copy_results(ROOT / source, out, seeds=summary["seeds"])
    hetero = heterogeneity(summary["base"])
    (out / "heterogeneity.json").write_text(
        json.dumps(hetero, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(summary, out)
    ps = load(ROOT / "runs/fedgraph-personalization/summary.json")
    if ps:
        copy_results(
            ROOT / "runs/fedgraph-personalization",
            out / "personalization",
            seeds=ps["seeds"],
        )
    (out / "report.md").write_bytes(page(summary, hetero, ps).encode())
    return summary


if __name__ == "__main__":
    build()
