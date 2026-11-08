"""The "At a glance" list at the top of the README, read from the saved summaries.

Each sentence states a direction ("helps", "worse", "closes the gap"), so each one is
checked against its numbers when the README is built. If a result moves far enough that
a sentence stops being true, the build fails instead of printing it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pages import join_and, load, pct, variant_numbers  # noqa: E402
from presentation import top_three  # noqa: E402


def need(condition, what):
    if not condition:
        raise AssertionError(f"README sentence no longer holds: {what}")


def better(g):
    return g["ci95"][1] < 0


def worse(g):
    return g["ci95"][0] > 0


def level(g):
    return g["ci95"][0] <= 0 <= g["ci95"][1]


def pair(summary, first, second):
    return summary["paired_day_bootstrap"][f"{first}_vs_{second}"]


def clock(ext):
    m = ext["models"]
    g = ext["found"]["gru_time_vs_gru"]
    need(better(g), "calendar inputs help the GRU")
    need(
        all(
            m[k]["mae"]["mean"] < m["persistence"]["mae"]["mean"]
            for k in ("ha", "linear", "gru", "gru_time")
        ),
        "every model beats persistence",
    )
    return (
        f"- **Does the time of day help?** More than anything else I tried. Adding it, with a "
        f"weekend flag, takes the GRU from {m['gru']['mae']['mean']:.3f} to "
        f"{m['gru_time']['mae']['mean']:.3f} mph MAE ({g['estimate']:+.3f} mph, 95% CI "
        f"{g['ci95'][0]:+.3f} to {g['ci95'][1]:+.3f}), and every model beats persistence "
        f"({m['persistence']['mae']['mean']:.3f}). "
        f"([benchmark](reports/benchmark-extended/report.md))"
    )


def graph(ext, scaling):
    spread = top_three(ext["models"])[2]
    shared = [
        scaling["day_bootstrap"][str(k)]["sgraph_time_vs_sgru_time"]
        for k in scaling["sizes"]
    ]
    need(
        all(better(g) for g in shared),
        "the graph helps the shared models at every size",
    )
    estimates = [g["estimate"] for g in shared]
    return (
        f"- **Does the road graph help?** Hardly in the dense models: the three best "
        f"16-sensor models are within {spread:.3f} mph of each other. In a shared-weight "
        f"model, where the graph is the only view of the neighbors, it helps at all "
        f"{len(shared)} sizes ({min(estimates):+.3f} to {max(estimates):+.3f} mph). "
        f"([scaling](reports/scaling/report.md))"
    )


def scale(scaling, variants):
    models, gaps = scaling["models"], scaling["day_bootstrap"]
    large = str(scaling["sizes"][-1])
    vs = [(k, gaps[str(k)]["sgru_time_vs_gru_time"]) for k in scaling["sizes"]]
    wins = [(k, g) for k, g in vs if better(g)]
    need(
        wins and wins[-1][0] == scaling["sizes"][-1], "sharing wins at the largest size"
    )
    _, vector, _, graph_gain = variant_numbers(variants, scaling)
    need(min(vector) > max(graph_gain), "a sensor vector helps more than the graph")
    return (
        f"- **What changes at 207 sensors?** Sharing one small GRU across all sensors takes "
        f"over: at {join_and([k for k, _ in wins])} sensors it beats the dense GRU by "
        f"{-max(g['estimate'] for _, g in wins):.3f} to {-min(g['estimate'] for _, g in wins):.3f} mph "
        f"with {models['sgru_time'][large]['parameters']:,} parameters instead of "
        f"{models['gru_time'][large]['parameters']:,}. Giving each sensor a learned "
        f"4-number vector helps more than the graph does (MAE {-max(vector):+.3f} to "
        f"{-min(vector):+.3f} mph, against {-max(graph_gain):+.3f} to {-min(graph_gain):+.3f} "
        f"for the graph). "
        f"([scaling](reports/scaling/report.md), [variants](reports/variants/report.md))"
    )


def intervals(unc):
    m = unc["mean"]["methods"]
    pooled, cqr = m["conformal_pooled"], m["cqr"]
    jam, free = (
        pooled["origin_congested"]["coverage"],
        pooled["origin_free_flow"]["coverage"],
    )
    need(
        jam < free - 0.15 and cqr["origin_congested"]["coverage"] > jam + 0.15,
        "conformal under-covers congestion and the quantile network fixes it",
    )
    return (
        f"- **Can the uncertainty be trusted?** On average, yes: {pct(1 - unc['alpha'])} "
        f"conformal intervals cover {pct(pooled['overall']['coverage'], 1)} of the test speeds. "
        f"In congestion, no: {pct(jam)} when the last speed was under 40 mph, against "
        f"{pct(free)} in free flow. A quantile network with a conformal correction holds "
        f"{pct(cqr['origin_congested']['coverage'])} there at the same average width. "
        f"([intervals](reports/uncertainty/report.md))"
    )


def drift(shf):
    std, stale = shf["settings"]["standard"], shf["settings"]["stale"]

    def gain(setting, kind):
        return -setting["strategies"][kind]["vs_static"]["estimate"]

    naive = min(
        std["strategies"][k]["vs_static"]["estimate"] for k in ("bias_ema", "bias_slot")
    )
    need(
        gain(std, "head_daily") > 0 and gain(std, "retrain_weekly") > 0 and naive > 0,
        "fine-tuning helps and averaging past errors hurts",
    )
    return (
        f"- **What happens when traffic drifts?** Not much over these four months, so "
        f"adapting helps a little: fine-tuning the output layer daily gains "
        f"{gain(std, 'head_daily'):.3f} mph and the whole network weekly "
        f"{gain(std, 'retrain_weekly'):.3f} ({gain(stale, 'head_daily'):.3f} and "
        f"{gain(stale, 'retrain_weekly'):.3f} for a model trained on only the first 40% of "
        f"the timeline). Averaging past errors into the forecast makes it worse. "
        f"([drift](reports/shift/report.md))"
    )


def federated(fed, personal):
    arms, base = fed["arms"], fed["base"]
    price = pair(fed, "fed_sgraph", "central_sgraph_local")
    alone = pair(fed, "fed_sgraph", "local_sgraph")
    need(
        worse(price) and worse(alone),
        "federating costs accuracy and training alone wins",
    )
    tuned = pair(personal, "fed_geo+ft", "local_geo")
    need(
        level(tuned), "fine-tuning the federated model closes the gap to training alone"
    )
    return (
        f"- **Does federated learning cost accuracy?** On all {base['sensors']} sensors, split "
        f"into {base['clients']} geographic clients that each see only their own part of the "
        f"road graph, the federated shared Graph-GRU scores "
        f"{arms['fed_sgraph']['mae']['mean']:.3f} mph (the historical average gets "
        f"{fed['baselines']['ha']['mae']:.3f}). Federating costs {price['estimate']:.3f} mph "
        f"against pooling the data, and every client training alone does "
        f"{alone['estimate']:.3f} mph better. In a 64-sensor study, fine-tuning the federated "
        f"model on each client closed that gap. "
        f"([federated graph](reports/fedgraph/report.md), [FedProx](reports/fedopt/report.md))"
    )


def diffusion(central, fed):
    gaps = central["day_bootstrap"]
    g16, g64 = (gaps[k]["dcrnn_time_vs_sgraph_time"] for k in ("16", "64"))
    shuffled = gaps["16"]["dcrnn_time_vs_dcrnn_perm_time"]
    federated_gap = pair(fed, "fed_dcrnn", "fed_sgraph")
    need(
        better(g16) and better(g64) and better(shuffled) and worse(federated_gap),
        "DCRNN wins centrally and loses its edge when federated",
    )
    return (
        f"- **Is a diffusion model worth it?** Centrally, yes. A DCRNN-style diffusion GRU "
        f"beats the shared Graph-GRU by {-g16['estimate']:.3f} mph at 16 sensors and "
        f"{-g64['estimate']:.3f} at 64 (one seed there), and a shuffled graph costs it "
        f"{-shuffled['estimate']:.3f}, so it does use the road structure. Federated, the "
        f"advantage reverses: it is {federated_gap['estimate']:.3f} mph worse than the "
        f"federated Graph-GRU. ([DCRNN](reports/diffusion/report.md))"
    )


def holdout(hold):
    m = hold["methods"]
    zero = pair(hold, "sgraph_zero_shot", "oracle_sgraph")
    fitted = pair(hold, "sgru_id_fitted", "sgraph_zero_shot")
    ridge = pair(hold, "sgru_id_ridge", "sgru_id_mean")
    need(
        worse(zero) and better(fitted) and worse(ridge),
        "zero-shot costs a little, a fitted vector beats it, a predicted vector fails",
    )
    return (
        f"- **Can it forecast a sensor it never trained on?** Yes. On "
        f"{len(hold['held_out'][0])} held-out sensors with only their first week of data, the "
        f"shared Graph-GRU scores {m['sgraph_zero_shot']['mae']['mean']:.3f} mph against "
        f"{m['oracle_sgraph']['mae']['mean']:.3f} when it trained on them. The "
        f"sensor-vector model has no vector for a new sensor; fitting one on that first week "
        f"gives {m['sgru_id_fitted']['mae']['mean']:.3f} mph, {-fitted['estimate']:.3f} better "
        f"than the zero-shot Graph-GRU, while predicting the vector from sensor descriptors "
        f"didn't work. ([held-out sensors](reports/holdout/report.md))"
    )


def fedprox(opt):
    arms = opt["arms"]
    gaps = [
        pair(opt, f"{t}_fedprox_{mu}", f"{t}_fedavg")
        for t in ("geo", "cong", "bal")
        for mu in ("0.001", "0.01", "0.1")
        if f"{t}_fedprox_{mu}_vs_{t}_fedavg" in opt["paired_day_bootstrap"]
    ]
    five = pair(opt, "geo_e5", "geo_fedavg")
    need(
        all(worse(g) for g in gaps) and worse(five),
        "FedProx hurts, five epochs cost a little",
    )
    rounds = arms["geo_fedavg"]["budgets"]["30"]["communication_bytes"]
    fewer = arms["geo_e5"]["budgets"]["6"]["communication_bytes"]
    return (
        f"- **Does FedProx help?** Not here: it was worse than FedAvg in all {len(gaps)} "
        f"settings I ran, and the penalty grew with mu. Five local epochs per round, though, "
        f"cut communication about {rounds / fewer:.0f} times for {five['estimate']:.3f} mph "
        f"(64 sensors, six clients, one seed). ([FedProx](reports/fedopt/report.md))"
    )


def bullets(ext):
    """The README's at-a-glance bullets, in the order of the sections below them."""
    scaling = load("reports/scaling/scaling-summary.json")
    variants = load("reports/variants/variants-summary.json")
    unc = load("reports/uncertainty/summary.json")
    shf = load("reports/shift/summary.json")
    fed = load("reports/fedgraph/summary.json")
    personal = load("reports/fedopt/personalization/summary.json")
    central = load("reports/diffusion/central-summary.json")
    fed_dcrnn = load("reports/diffusion/federated/summary.json")
    hold = load("reports/holdout/summary.json")
    opt = load("reports/fedopt/summary.json")
    parts = [clock(ext)]
    if scaling:
        parts.append(graph(ext, scaling))
        if variants:
            parts.append(scale(scaling, variants))
    if unc:
        parts.append(intervals(unc))
    if shf:
        parts.append(drift(shf))
    if fed and personal:
        parts.append(federated(fed, personal))
    if central and fed_dcrnn:
        parts.append(diffusion(central, fed_dcrnn))
    if hold:
        parts.append(holdout(hold))
    if opt:
        parts.append(fedprox(opt))
    return parts
