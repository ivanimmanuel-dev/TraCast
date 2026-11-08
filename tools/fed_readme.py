"""README and report text for the federated graph studies, read from the saved summaries.

pages.readme_blocks merges these into the blocks the README and the technical report
are built from, so the numbers quoted there are the ones in reports/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fed_pages import claim, load, mae, plural, seed_phrase, versus  # noqa: E402
from pages import join_and  # noqa: E402


def pair(summary, first, second):
    return summary["paired_day_bootstrap"].get(f"{first}_vs_{second}")


def table(rows, head):
    return "\n".join(head + rows)


# --- the 207-sensor federated study ---------------------------------------------------


def federated(summary):
    if not summary:
        return None
    arms, base, part = summary["arms"], summary["base"], summary["partition"]
    names = [
        ("fed_sgraph", "Federated shared Graph-GRU"),
        ("fed_sgru", "Federated shared GRU (no graph)"),
        ("central_sgraph_full", "Pooled shared Graph-GRU, full graph"),
        ("central_sgraph_local", "Pooled shared Graph-GRU, local graphs only"),
        ("local_sgraph", "Local-only shared Graph-GRU (no communication)"),
    ]
    rows = [
        f"| {label} | {mae(arms[name])} | {arms[name]['rmse']['mean']:.3f} |"
        for name, label in names
        if name in arms
    ]
    rows += [
        f"| Persistence | {summary['baselines']['persistence']['mae']:.3f} | {summary['baselines']['persistence']['rmse']:.3f} |",
        f"| Historical average | {summary['baselines']['ha']['mae']:.3f} | {summary['baselines']['ha']['rmse']:.3f} |",
    ]
    local = pair(summary, "fed_sgraph", "local_sgraph")
    price = pair(summary, "fed_sgraph", "central_sgraph_local")
    graph = pair(summary, "fed_sgraph", "fed_sgru")
    fed = arms["fed_sgraph"]
    sizes = sorted(c["sensors"] for c in part["clients"])
    pooled = pair(summary, "local_sgraph", "central_sgraph_local")
    title = claim(
        local,
        "and it beat every client training alone",
        "but training alone was better",
        "and was not clearly different from training alone",
    )
    bullet = (
        f"- **Federated graph forecasting works at full scale, {title}.** "
        f"A federated shared Graph-GRU on all {base['sensors']} sensors ({base['clients']} geographic clients with "
        f"{sizes[0]} to {sizes[-1]} sensors, each seeing only its own part of the road graph) scores "
        f"{mae(fed)} mph, about {summary['baselines']['ha']['mae'] - fed['mae']['mean']:.2f} mph better than the "
        f"historical average. "
        + claim(
            local,
            "It beat every client training by itself by {amount} mph (interval {interval}).",
            "Every client training by itself did {amount} mph better (interval {interval})",
            "The gap to every client training by itself was {amount} mph (interval {interval}).",
        )
        + (
            ", and also beat one pooled model with the same graph, which points to a single set of "
            "weights for every area as the limit."
            if pooled and pooled["ci95"][1] < 0 and local["ci95"][0] > 0
            else "."
        )
    )
    wins = sum(
        a["mae"]["mean"] < b["mae"]["mean"]
        for a, b in zip(arms["local_sgraph"]["clients"], fed["clients"])
    )
    alone = (
        "Training alone won on every client"
        if wins == len(fed["clients"])
        else f"Training alone won on {wins} of {len(fed['clients'])} clients"
    )
    text = (
        f"## Federated graph learning\n\n"
        f"The first federated experiment was small: 12 sensors and a GRU that looked at one sensor at a time. "
        f"The shared Graph-GRU can be federated properly, because its {fed['parameters']:,} weights don't "
        f"depend on how many sensors a client has. I cut all {base['sensors']} sensors into {base['clients']} "
        f"geographic clients ({join_and(sizes)} sensors) with k-means on their coordinates. A client sees only "
        f"its own sensors' readings and the road graph among them, so {1 - part['graph']['edges_kept_fraction']:.0%} "
        f"of the road edges (the ones crossing into another client) are gone for everyone. Flower's FedAvg "
        f"averages only the learned weights. Test MAE in mph, {plural(len(summary['seeds']), 'seed')}, with a pooled model "
        f"(same data and training examples, one place) and local-only training (no communication) as references:\n\n"
        + table(rows, ["| Arm | MAE (SD) | RMSE |", "|---|---:|---:|"])
        + "\n\n![Clients and the road graph](reports/fedgraph/partition-map.png)\n\n"
        + (
            versus(
                price, "Federated training", "a pooled model with the same cut graph"
            )
            + ", and "
            + versus(
                graph,
                "the federated Graph-GRU",
                "the federated GRU with no graph",
            )
            + (", so the local graph still helps. " if graph["ci95"][1] < 0 else ". ")
            if price and graph
            else ""
        )
        + f"{alone}; the [headline report](reports/fedgraph/report.md) has the numbers by client and by "
        f"forecast horizon. The weights are tiny next to the data "
        f"({fed['communication']['total_parameter_bytes'] / 2**20:.1f} MiB exchanged over the whole run), "
        f"though this is a single-machine simulation, not a deployment.\n"
    )
    return bullet, text


# --- DCRNN ----------------------------------------------------------------------------


def diffusion(central, ablation, fed):
    if not (central and ablation):
        return None
    models, sizes = central["models"], central["sizes"]
    k = str(sizes[0])
    rows = []
    for name, label in (
        ("gru_time", "Dense GRU"),
        ("sgru_time", "Shared GRU (no graph)"),
        ("sgraph_time", "Shared Graph-GRU (one-hop neighbor averages)"),
        ("dcrnn_k0_time", "DCRNN with no diffusion (0 hops)"),
        ("dcrnn_perm_time", "DCRNN, shuffled graph"),
        ("dcrnn_time", "DCRNN (2 hops, forward and reverse)"),
    ):
        by = models.get(name, {})
        if k in by:
            rows.append(f"| {label} | {mae(by[k])} | {by[k]['parameters']:,} |")
    gaps = central["day_bootstrap"].get(k, {})
    g_graph = gaps.get("dcrnn_time_vs_sgraph_time")
    g_perm = gaps.get("dcrnn_time_vs_dcrnn_perm_time")
    g_k0 = gaps.get("dcrnn_time_vs_dcrnn_k0_time")
    last = str(sizes[-1])
    g_last = central["day_bootstrap"].get(last, {}).get("dcrnn_time_vs_sgraph_time")
    ab = ablation["day_bootstrap"]["16"]
    g_dir = ab.get("dcrnn_sym_time_vs_dcrnn_time")
    g_depth = ab.get("dcrnn_k3_time_vs_dcrnn_time")
    federated_part = ""
    federated_short = ""
    if fed:
        fg = fed["paired_day_bootstrap"].get("fed_dcrnn_vs_fed_sgraph")
        pg = fed["paired_day_bootstrap"].get(
            "central_dcrnn_full_vs_central_sgraph_full"
        )
        price_d = fed["paired_day_bootstrap"].get("fed_dcrnn_vs_central_dcrnn_local")
        price_s = fed["paired_day_bootstrap"].get("fed_sgraph_vs_central_sgraph_local")
        if fg and pg and price_d and price_s:
            count = fed["base"]["clients"]
            count = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}.get(
                count, str(count)
            )
            where = f"with {count} clients on {fed['base']['sensors']} sensors"
            pooled_clause = (
                f"{-pg['estimate']:.3f} mph better"
                if pg["ci95"][1] < 0
                else "not clearly different"
            )
            federated_short = (
                f" Federated, {where}, that advantage "
                + claim(
                    fg,
                    "grows to {amount} mph in DCRNN's favor",
                    "reverses: federated DCRNN is {amount} mph worse than the federated Graph-GRU "
                    f"(interval {{interval}}), although pooled it is {pooled_clause}",
                    "is not clearly there",
                )
                + "."
            )
            federated_part = (
                federated_short
                + f" Federating costs DCRNN {price_d['estimate']:.3f} mph and the Graph-GRU "
                f"{price_s['estimate']:.3f} against pooled models with the same cut graph."
            )
    bullet = (
        "- **A DCRNN-style diffusion GRU is the best centralized model I tried.** "
        + (
            claim(
                g_graph,
                "At 16 sensors it beats the shared Graph-GRU by {amount} mph (interval {interval})",
                "At 16 sensors it is {amount} mph worse than the shared Graph-GRU",
                "At 16 sensors it is not clearly different from the shared Graph-GRU",
            )
            if g_graph
            else ""
        )
        + (
            claim(
                g_last,
                f" and by {{amount}} mph at {last} sensors (one seed there)",
                f" and {{amount}} mph worse at {last} sensors",
                f" and not clearly different at {last} sensors",
            )
            if g_last
            else ""
        )
        + ". "
        + (
            claim(
                g_perm,
                "A randomly relabelled graph is {amount} mph worse, so the real road structure is what it uses",
                "A shuffled graph was {amount} mph better, which would mean the road structure isn't what helps",
                "A shuffled graph is not clearly different",
            )
            if g_perm
            else ""
        )
        + "."
        + federated_short
    )
    text = (
        "## A DCRNN-style diffusion GRU\n\n"
        "The shared Graph-GRU only averages direct neighbors. DCRNN (Li et al., ICLR 2018) puts diffusion "
        "convolution inside the GRU gates: random walks along the road and against it, several hops, so "
        "information can travel along the network and the direction of the road matters. I wrote it from the "
        "paper's equations as a direct multi-horizon model, with the number of hops, the directions and the "
        "isolated-sensor handling configurable, and it works with the pooled and the federated code. "
        f"Test MAE in mph at {sizes[0]} sensors, same data, splits and update budget for every model:\n\n"
        + table(rows, ["| Model | MAE (SD) | Parameters |", "|---|---:|---:|"])
        + "\n\n![The model](reports/diffusion/dcrnn-architecture.png)\n\n"
        + " ".join(
            fact
            for fact in (
                f"Diffusion itself is worth {-g_k0['estimate']:.3f} mph over the same model with no hops."
                if g_k0 and g_k0["ci95"][1] < 0
                else "",
                "Two hops are enough, since three are not clearly better."
                if g_depth and g_depth["ci95"][0] <= 0 <= g_depth["ci95"][1]
                else "",
                f"Treating the road as undirected costs {g_dir['estimate']:.3f} mph against using both directions, so the direction matters."
                if g_dir and g_dir["ci95"][0] > 0
                else "",
                federated_part.strip(),
            )
            if fact
        )
        + " The [report](reports/diffusion/report.md) has the ablations, the 64-sensor "
        "results and the federated runs.\n"
    )
    return bullet, text


# --- held-out sensors -----------------------------------------------------------------


def holdout_paragraph(summary):
    pairs = summary["paired_day_bootstrap"]
    m = summary["methods"]
    new = pairs.get("sgraph_zero_shot_vs_oracle_sgraph")
    fitted = pairs.get("sgru_id_fitted_vs_sgraph_zero_shot")
    ridge = pairs.get("sgru_id_ridge_vs_sgru_id_mean")
    parts = []
    if new:
        parts.append(
            versus(
                new,
                "On a sensor it never saw, the shared Graph-GRU",
                "the same model trained on it",
                interval=False,
            )
            + "."
        )
    parts.append(
        f"The sensor-vector model has no vector to use ({m['sgru_id_zero']['mae']['mean']:.3f} mph "
        f"with a zero vector)."
    )
    if fitted:
        parts.append(
            versus(
                fitted,
                "Fitting just that vector on the first week",
                "the zero-shot Graph-GRU",
                interval=False,
            )
            + "."
        )
    if ridge:
        parts.append(
            versus(
                ridge,
                "A vector predicted from sensor descriptors",
                "the average vector",
                interval=False,
            )
            + "."
        )
    return " ".join(parts)


def holdout(summary):
    if not summary:
        return None
    m = summary["methods"]
    zero = m["sgraph_zero_shot"]
    oracle = m["oracle_sgraph"]
    fitted = m["sgru_id_fitted"]
    ridge = m["sgru_id_ridge"]
    mean = m["sgru_id_mean"]
    pairs = summary["paired_day_bootstrap"]
    g_fit = pairs.get("sgru_id_fitted_vs_sgraph_zero_shot")
    g_ridge = pairs.get("sgru_id_ridge_vs_sgru_id_mean")
    n = len(summary["held_out"][0])
    bullet = (
        f"- **A model can forecast sensors it never trained on, and the sensor-vector models can be rescued.** "
        f"On {n} held-out sensors (a whole area, with only their first week of data available) the shared "
        f"Graph-GRU scores {mae(zero)} mph zero-shot against {mae(oracle)} when it trained on them. "
        f"The vector model breaks without a vector ({mae(m['sgru_id_zero'])} with a zero vector), but fitting just "
        f"that vector on the first week gives {mae(fitted)} mph"
        + (
            ", "
            + claim(
                g_fit,
                "{amount} mph better than the plain zero-shot model",
                "{amount} mph worse than the plain zero-shot model",
                "not clearly different from the plain zero-shot model",
            )
            if g_fit
            else ""
        )
        + f". Predicting the vector from sensor descriptors did not work ({mae(ridge)} against {mae(mean)} for the average vector)."
        if g_ridge
        else ""
    )
    text = (
        "## Sensors the model never saw\n\n"
        "A sensor-vector model has a row per training sensor, so a new sensor has no vector, while the plain "
        "shared models can take any sensor. I held out one geographic area and gave each new sensor only its "
        "first week of readings. Test MAE in mph on the held-out sensors, "
        f"{len(summary['seeds'])} seeds:\n\n"
        + table(
            [
                f"| {m[n_]['label']} | {mae(m[n_])} |"
                for n_ in (
                    "persistence",
                    "ha_first_days",
                    "sgru_zero_shot",
                    "sgraph_zero_shot",
                    "sgru_id_zero",
                    "sgru_id_mean",
                    "sgru_id_ridge",
                    "sgru_id_fitted",
                    "oracle_sgraph",
                    "oracle_sgru_id",
                )
            ],
            ["| Method | MAE (SD) |", "|---|---:|"],
        )
        + "\n\n"
        + holdout_paragraph(summary)
        + " The [report](reports/holdout/report.md) has the comparisons with intervals and the limits.\n"
    )
    return bullet, text


# --- FedAvg / FedProx -----------------------------------------------------------------


def optimization(summary, personal):
    if not summary:
        return None
    bullets = []
    gaps = []
    for tag, label in (
        ("geo", "geographic"),
        ("cong", "congestion-grouped"),
        ("bal", "balanced"),
    ):
        for mu in ("0.001", "0.01", "0.1"):
            g = pair(summary, f"{tag}_fedprox_{mu}", f"{tag}_fedavg")
            if g:
                gaps.append((label, mu, g))
    if gaps:
        worse = [x for x in gaps if x[2]["ci95"][0] > 0]
        better = [x for x in gaps if x[2]["ci95"][1] < 0]
        geo = [(mu, g["estimate"]) for label, mu, g in gaps if label == "geographic"]
        trend = ", ".join(f"{e:+.3f} at mu {mu}" for mu, e in geo)
        e5 = pair(summary, "geo_e5_fedprox_0.1", "geo_e5")
        if len(worse) == len(gaps):
            verdict = (
                f"FedProx was worse than FedAvg in all {len(gaps)} settings I ran, and the penalty grows "
                f"with mu (geographic clients: {trend})"
            )
        elif len(better) == len(gaps):
            verdict = (
                f"FedProx was better than FedAvg in all {len(gaps)} settings I ran"
            )
        else:
            verdict = (
                f"FedProx helped in {len(better)} and hurt in {len(worse)} of {len(gaps)} settings "
                f"(geographic clients: {trend})"
            )
        if e5:
            verdict += "; " + claim(
                e5,
                "with five local epochs a round it was {amount} mph better",
                "with five local epochs a round, where there is more drift to correct, it was still {amount} mph worse",
                "with five local epochs a round it was not clearly different",
            )
        bullets.append(
            "- **FedProx did not help on 64 sensors with six clients.** "
            + verdict
            + ". A round here is short and the model is still improving at the end, so the proximal pull mostly slows learning."
        )
    if personal:
        g = pair(personal, "fed_geo+ft", "local_geo")
        f = pair(personal, "fed_geo+ft", "fed_geo")
        if g and f:
            bullets.append(
                "- **Fine-tuning the federated model on each client.** "
                "On geographic clients, letting every client keep training the final federated model "
                "on its own data "
                + claim(
                    f,
                    "improves it by {amount} mph (interval {interval}) and ",
                    "makes it {amount} mph worse and ",
                    "doesn't clearly change it and ",
                )
                + claim(
                    g,
                    "ends {amount} mph better than every client training alone.",
                    "ends {amount} mph worse than every client training alone.",
                    "ends level with every client training alone (no clear difference).",
                )
            )
    text = (
        "## FedAvg, FedProx and different kinds of clients\n\n"
        "The [FedProx study](reports/fedopt/report.md) uses 64 sensors and six clients, cut three ways "
        "(by area, balanced across congestion levels, or grouped by congestion level). It compares "
        "FedAvg and FedProx at several strengths, five local epochs against one, pooled and "
        "local-only references, and local fine-tuning of the federated model. Test MAE in mph, "
        f"{seed_phrase(summary['seeds'])}:\n\n"
        + prox_table(summary)
        + "\n\nFedAvg's error followed how different the clients' speed distributions were rather "
        "than how much of the road graph each cut kept"
        + (
            " (one seed, so a hint rather than a result)"
            if len(summary["seeds"]) == 1
            else ""
        )
        + ", and five local epochs a round cut the communication by about a factor of five at a "
        "small cost in accuracy. " + fine_tuning_sentence(personal) + "\n"
    )
    return bullets, text


def prox_table(summary):
    rows = []
    for tag, label in (
        ("geo", "Geographic areas"),
        ("bal", "Balanced"),
        ("cong", "Congestion-grouped"),
    ):
        names = (
            f"{tag}_fedavg",
            f"{tag}_fedprox_0.001",
            f"{tag}_fedprox_0.01",
            f"{tag}_fedprox_0.1",
        )
        cells = [
            f"{summary['arms'][n]['mae']['mean']:.3f}" if n in summary["arms"] else "-"
            for n in names
        ]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    head = (
        "| Clients | FedAvg | FedProx, mu 0.001 | FedProx, mu 0.01 | FedProx, mu 0.1 |\n"
        "|---|---:|---:|---:|---:|"
    )
    return head + "\n" + "\n".join(rows)


def fine_tuning_sentence(personal):
    if not personal:
        return ""
    arms = personal["arms"]
    g = pair(personal, "fed_geo+ft", "local_geo")
    if not (g and "fed_geo+ft" in arms and "local_geo" in arms):
        return ""
    return (
        f"Letting each client keep training the federated model on its own data took geographic "
        f"clients from {arms['fed_geo']['mae']['mean']:.3f} to "
        f"{arms['fed_geo+ft']['mae']['mean']:.3f} mph. Every client training alone scores "
        f"{arms['local_geo']['mae']['mean']:.3f}, and "
        + versus(g, "the fine-tuned model", "that", interval=False)
        + "."
    )


def blocks():
    fed = load(ROOT / "reports/fedgraph/summary.json")
    central = load(ROOT / "reports/diffusion/central-summary.json")
    ablation = load(ROOT / "reports/diffusion/ablation-summary.json")
    fed_diffusion = load(ROOT / "reports/diffusion/federated/summary.json")
    hold = load(ROOT / "reports/holdout/summary.json")
    opt = load(ROOT / "reports/fedopt/summary.json")
    personal = load(ROOT / "reports/fedopt/personalization/summary.json")
    results = {
        "fedgraph": federated(fed),
        "diffusion": diffusion(central, ablation, fed_diffusion),
        "holdout": holdout(hold),
        "fedopt": optimization(opt, personal),
    }
    bullets, sections = [], {}
    for key, value in results.items():
        if not value:
            continue
        got, text = value
        bullets += got if isinstance(got, list) else [got]
        sections[key] = text
    return bullets, sections


if __name__ == "__main__":
    bullets, sections = blocks()
    print("\n\n".join(bullets))
    print()
    print("\n\n".join(sections.values()))
