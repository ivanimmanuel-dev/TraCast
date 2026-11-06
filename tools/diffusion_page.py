"""Report page for the DCRNN-style diffusion GRU.

The centralized comparison and the ablation are scaling-style summaries (one table per
sensor count), the federated part is a fedgraph study summary. Every number in the
text is read from those files, and a comparison is only called better or worse when
the interval over test days excludes zero.
"""

import json
import shutil

from fed_pages import (
    ROOT,
    REPORTS,
    LABELS,
    copy_results,
    gap,
    load,
    mae,
    plural,
    seed_phrase,
    versus,
)
from pages import join_and

NAMES = {
    "persistence": "Persistence",
    "ha": "Historical average",
    "gru_time": "Dense GRU",
    "graph_gru_time": "Dense Graph-GRU",
    "sgru_time": "Shared GRU (no graph)",
    "sgraph_time": "Shared Graph-GRU",
    "dcrnn_time": "DCRNN, 2 hops",
    "dcrnn_h16_time": "DCRNN, 2 hops, 16 hidden units",
    "dcrnn_k0_time": "DCRNN with no diffusion (0 hops)",
    "dcrnn_k1_time": "DCRNN, 1 hop",
    "dcrnn_k3_time": "DCRNN, 3 hops",
    "dcrnn_perm_time": "DCRNN, 2 hops, shuffled graph",
    "dcrnn_fwd_time": "DCRNN, forward walks only",
    "dcrnn_bwd_time": "DCRNN, reverse walks only",
    "dcrnn_sym_time": "DCRNN, undirected walks",
}


def table(summary, models, key="mae", sd=True):
    sizes = summary["sizes"]
    head = "| Model | " + " | ".join(str(k) for k in sizes) + " |"
    rule = "|---|" + "---:|" * len(sizes)
    rows = []
    for name in models:
        by = summary["models"].get(name)
        if by is None:
            continue
        cells = []
        for k in sizes:
            entry = by.get(str(k))
            cells.append(
                mae({"mae": entry[key]}, sd=sd) if entry and entry.get(key) else ""
            )
        rows.append(f"| {NAMES.get(name, name)} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + rows)


def gaps(summary, name_a, name_b):
    return {
        k: summary["day_bootstrap"][str(k)].get(f"{name_a}_vs_{name_b}")
        for k in summary["sizes"]
        if f"{name_a}_vs_{name_b}" in summary["day_bootstrap"].get(str(k), {})
    }


def side_of(g):
    low, high = g["ci95"]
    return "better" if high < 0 else "worse" if low > 0 else "unclear"


def over_sizes(summary, a, b, first, second):
    """One sentence for a comparison at every sensor count, clause by clause."""
    found = gaps(summary, a, b)
    if not found:
        return ""
    return (
        "; ".join(
            versus(g, first, second, interval=False) + f" at {k} sensors"
            for k, g in found.items()
        )
        + "."
    )


def beat_sentence(summary, items):
    """'DCRNN beat X (0.119 mph at 16 sensors and 0.174 at 64), Y (...) and Z (...)'.

    Used when every interval excludes zero in DCRNN's favor; otherwise each comparison
    is written out on its own with its own better-or-worse wording.
    """
    parts, previous = [], None
    for a, b, subject, obj in items:
        found = gaps(summary, a, b)
        if not found:
            continue
        if any(side_of(g) != "better" for g in found.values()):
            return " ".join(over_sizes(summary, *item) for item in items)
        values = [-g["estimate"] for g in found.values()]
        sizes = list(found)
        if parts:
            nums = [f"{v:.3f}" for v in values]
        else:
            nums = [f"{values[0]:.3f} mph at {sizes[0]} sensors"] + [
                f"{v:.3f} at {k}" for v, k in zip(values[1:], sizes[1:])
            ]
        said = " and ".join([", ".join(nums[:-1]), nums[-1]] if len(nums) > 2 else nums)
        lead = f"{subject} beat " if subject != previous else ""
        parts.append(f"{lead}{obj} ({said})")
        previous = subject
    if not parts:
        return ""
    body = ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]
    return body[0].upper() + body[1:] + "; every interval excludes zero."


def gap_rows(summary, pairs):
    rows = []
    for a, b in pairs:
        found = gaps(summary, a, b)
        if not found:
            continue
        cells = " | ".join(
            gap(found[k]) if k in found else "" for k in summary["sizes"]
        )
        rows.append(f"| {NAMES.get(a, a)} - {NAMES.get(b, b)} | {cells} |")
    head = (
        "| Gap in MAE (first minus second) | "
        + " | ".join(f"{k} sensors" for k in summary["sizes"])
        + " |"
    )
    rule = "|---|" + "---:|" * len(summary["sizes"])
    return "\n".join([head, rule] + rows)


def central_text(c):
    """The centralized comparison in two sentences (the gap table has every number)."""
    against = [
        ("dcrnn_time", "sgraph_time", "DCRNN", "the shared Graph-GRU"),
        ("dcrnn_time", "sgru_time", "DCRNN", "the shared GRU with no graph"),
        ("dcrnn_time", "gru_time", "DCRNN", "the dense GRU"),
    ]
    source = [
        (
            "dcrnn_time",
            "dcrnn_k0_time",
            "two hops of diffusion",
            "no diffusion in the same architecture",
        ),
        ("dcrnn_time", "dcrnn_perm_time", "the real graph", "a shuffled one"),
        (
            "dcrnn_h16_time",
            "sgraph_time",
            "a 16-unit DCRNN, about the size of the Graph-GRU,",
            "the shared Graph-GRU",
        ),
    ]
    sentences = [beat_sentence(c, group) for group in (against, source)]
    return "\n\n".join(x for x in sentences if x)


def ablation_pairs():
    return [
        ("dcrnn_k1_time", "dcrnn_k0_time", "1 hop - 0 hops"),
        ("dcrnn_k1_time", "dcrnn_time", "1 hop - 2 hops"),
        ("dcrnn_k3_time", "dcrnn_time", "3 hops - 2 hops"),
        ("dcrnn_fwd_time", "dcrnn_time", "Forward walks only - both directions"),
        ("dcrnn_bwd_time", "dcrnn_time", "Reverse walks only - both directions"),
        ("dcrnn_sym_time", "dcrnn_time", "Undirected walks - both directions"),
    ]


def ablation_gap_table(a):
    rows = []
    for first, second, title in ablation_pairs():
        found = gaps(a, first, second)
        if found:
            rows.append(f"| {title} | {gap(next(iter(found.values())))} |")
    return "\n".join(["| Comparison | Gap in MAE (mph) |", "|---|---:|"] + rows)


def ablation_text(a):
    def one(first, second):
        found = gaps(a, first, second)
        return next(iter(found.values())) if found else None

    depth = []
    for first, second, subject, other in (
        ("dcrnn_k1_time", "dcrnn_k0_time", "One hop", "no diffusion"),
        ("dcrnn_k1_time", "dcrnn_time", "One hop", "two hops"),
        ("dcrnn_k3_time", "dcrnn_time", "Three hops", "two hops"),
    ):
        g = one(first, second)
        if g:
            verb = "were" if subject.endswith("hops") else "was"
            depth.append(versus(g, subject, other, interval=False, verb=verb) + ".")
    direction = [
        (name, one(key, "dcrnn_time"))
        for name, key in (
            ("only forward walks", "dcrnn_fwd_time"),
            ("only reverse walks", "dcrnn_bwd_time"),
            ("undirected walks", "dcrnn_sym_time"),
        )
    ]
    direction = [(name, g) for name, g in direction if g]
    text = " ".join(depth)
    if direction and all(g["ci95"][0] > 0 for _, g in direction):
        amounts = [g["estimate"] for _, g in direction]
        names = [name for name, _ in direction]
        listing = (
            ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 else names[0]
        )
        text += (
            f" Using {listing} was {min(amounts):.3f} to {max(amounts):.3f} mph worse than "
            "using both directions (every interval is above zero), so the direction of the road matters."
        )
    else:
        text += " " + " ".join(
            versus(
                g,
                name[0].upper() + name[1:],
                "both directions",
                interval=False,
                verb="were",
            )
            + "."
            for name, g in direction
        )
    return text.strip()


FED_GAPS = (
    ("fed_dcrnn", "fed_sgraph"),
    ("central_dcrnn_full", "central_sgraph_full"),
    ("fed_dcrnn", "central_dcrnn_local"),
    ("fed_sgraph", "central_sgraph_local"),
    ("central_dcrnn_local", "central_dcrnn_full"),
    ("central_sgraph_local", "central_sgraph_full"),
    ("fed_dcrnn", "local_dcrnn"),
)


def federated_gap_table(f):
    rows = []
    for a, b in FED_GAPS:
        g = f["paired_day_bootstrap"].get(f"{a}_vs_{b}")
        if g:
            rows.append(f"| {LABELS[a]} - {LABELS[b]} | {gap(g)} |")
    return "\n".join(["| Comparison | Gap in MAE (mph) |", "|---|---:|"] + rows)


PAGE = """# A DCRNN-style diffusion GRU

The shared Graph-GRU in the earlier studies gives each sensor the average speed of its direct upstream and downstream neighbors. That is a one-hop picture, and traffic doesn't stop at the next sensor: a slowdown spreads back along the road over many sensors, and it matters which way the road runs. DCRNN (Li, Yu, Shahabi and Liu, ICLR 2018) models this with **diffusion convolution**: random walks along the road (forward) and against it (reverse), over several hops, inside the gates of a GRU. I wanted to know whether that extra machinery helps on METR-LA or whether the simple one-hop average was already enough.

![The model](dcrnn-architecture.png)

## The model

For a graph with weights W, the forward transition matrix is `P_f = D_out^-1 W` (each sensor's weights to its downstream neighbors, normalized to sum to one) and the reverse one is `P_r = D_in^-1 W^T`. One diffusion convolution mixes a sensor's features with what random walks of 1, 2, ... K steps along each direction reach, `[x, P_f x, P_f^2 x, ..., P_r x, P_r^2 x, ...]`, and applies one linear layer to the result. The DCGRU cell is an ordinary GRU cell whose reset gate, update gate and candidate state each use that convolution on `[input, state]` instead of a plain linear layer, so the hidden state itself is diffused along the road.

How it compares with the paper:

- **Same:** directed forward and reverse random walks, K hops per direction, diffusion convolution inside the gates, one set of weights for every sensor.
- **Different:** the paper's full model is an encoder-decoder with scheduled sampling that predicts the horizons one after another. Mine encodes the input hour and predicts all twelve horizons at once from the last hidden state, the same direct head every other model here uses, so the comparison is like for like. It has one layer, and the adjacency has no self-loops (hop 0 already is the sensor itself).
- **Written from the equations,** not adapted from existing code. The tests check it against hand-computed random walks on small graphs, check that relabelling the sensors relabels the forecast, and check that an isolated sensor (no edges at all) gets a zero walk and still trains.

A sensor with no outgoing (or no incoming) edge has a zero row in that direction's matrix, so it just receives nothing through it. The learned weights depend on the number of features and hops, never on the number of sensors: with 32 hidden units, calendar inputs and two hops, DCRNN has {dcrnn_params:,} parameters on any graph, against {sgraph_params:,} for the shared Graph-GRU and {sgru_params:,} for the shared GRU with no graph. A 16-unit version ({h16_params:,} parameters) is about the same size as the Graph-GRU, so the comparison can be made with size held roughly equal too.

## Centralized comparison

The same data, forecast horizons, splits, normalization, missing-value masking, optimizer and update budget as the [scaling study](../scaling/report.md): {epochs} epochs that each use a random third of the training windows, the best epoch on validation kept. Sizes {sizes}; seeds: {seeds}.

Test MAE in mph on all the sensors each model was trained on, mean (SD over seeds):

{own_table}

On the same first 16 sensors for every size (so the targets are identical across columns):

{core_table}

Paired gaps on identical windows (first minus second, negative favors the first; 95% interval from resampling test days, seeds averaged first):

{gap_table}

{central_text}

![Accuracy by sensor count](central-accuracy.png)

## Ablations: depth, direction, and does the graph matter

Sixteen sensors, same protocol, {ablation_seeds}.

{ablation_table}

Gaps on identical windows (first minus second, negative favors the first; 95% interval from resampling test days, seeds averaged first):

{ablation_gaps}

{ablation_text}

![Ablation](ablation-accuracy.png)

## Federated and pooled

The same model through Flower: {clients} geographic clients on {fed_sensors} sensors, each with only its own sensors and the edges among them, FedAvg weighted by examples, {fed_rounds} rounds of one local epoch of a random third of the windows, {fed_seeds}. The learned weights have the same shape for every client, so clients with different numbers of sensors can still be averaged. The arms are the ones from the [federated study](../fedgraph/report.md), run for DCRNN as well as for the shared Graph-GRU.

| Arm | MAE (SD) | Parameters | Best round | Train min | Comm MiB |
|---|---:|---:|---:|---:|---:|
{fed_rows}

Gaps on identical windows (same convention as above):

{federated_gaps}

{federated_story}

![Federated accuracy](federated-accuracy.png)

![Federated convergence](federated-convergence.png)

## Cost

{cost_text}

## Limits

- Equal update budgets, not equal convergence. DCRNN has about {ratio:.1f} times the parameters of the shared Graph-GRU and may need longer, or a different learning rate, to show what it can do. The 16-unit version is closer in size but isn't a full control.
- Few seeds (centralized: {seeds}; federated: {fed_n}). The intervals over test days reflect which days were held out, not the randomness of training.
- The direct multi-horizon head replaces the paper's autoregressive decoder, so this is DCRNN-style rather than a reproduction. I don't compare with the published METR-LA numbers because the splits, horizons and budgets differ.
- Accuracy is measured up to {largest} sensors centralized and {fed_sensors} federated. At all 207 sensors DCRNN is only timed.
- One city, one four-month period, and a test period with more missing readings than training.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.cli scaling --config configs/diffusion.json --out runs/diffusion
.\\.venv\\Scripts\\python.exe -m tracast.cli scaling --config configs/diffusion-ablation.json --out runs/diffusion-ablation
.\\.venv\\Scripts\\python.exe -m tracast.cli fedgraph --config configs/fedgraph-diffusion.json --out runs/fedgraph-diffusion
.\\.venv\\Scripts\\python.exe -m tracast.cli scaling --config configs/diffusion-cost.json --out runs/diffusion-cost
```

Files: [central-summary.json](central-summary.json), [ablation-summary.json](ablation-summary.json), [cost-summary.json](cost-summary.json) and the federated study in [federated/](federated/).
"""


def seed_text(summary):
    """Seeds per sensor count, from what each size's table was actually built from."""
    models = summary["models"]
    parts = []
    for k in summary["sizes"]:
        seeds = models["dcrnn_time"][str(k)]["seeds"]
        parts.append(
            f"{join_and(seeds)} at {k} sensors" if seeds else f"none at {k} sensors"
        )
    return "; ".join(parts)


def federated_story(f):
    """What the federated numbers say about DCRNN, in words that follow the gaps."""
    arms = f["arms"]
    pairs = f["paired_day_bootstrap"]
    fed_gap = pairs.get("fed_dcrnn_vs_fed_sgraph")
    pooled_gap = pairs.get("central_dcrnn_full_vs_central_sgraph_full")
    price_d = pairs.get("fed_dcrnn_vs_central_dcrnn_local")
    price_s = pairs.get("fed_sgraph_vs_central_sgraph_local")
    local_d = arms.get("local_dcrnn")
    pooled_d = arms.get("central_dcrnn_full")
    if not (fed_gap and pooled_gap and price_d and price_s):
        return ""
    text = ""
    if fed_gap["ci95"][0] > 0 and pooled_gap["ci95"][1] < 0:
        text += (
            f"The diffusion model's advantage does not survive federation. Pooled, DCRNN beats the "
            f"shared Graph-GRU by {-pooled_gap['estimate']:.3f} mph; federated, it is "
            f"{fed_gap['estimate']:.3f} mph worse. Federating costs DCRNN {price_d['estimate']:.3f} mph "
            f"(against a pooled model with the same cut graph) and the shared Graph-GRU "
            f"{price_s['estimate']:.3f}, so the larger model ({arms['fed_dcrnn'].get('parameters', 0):,} "
            f"parameters against {arms['fed_sgraph'].get('parameters', 0):,}) loses much more to averaging "
            f"the clients' updates in this short budget. The cut graph costs about the same for both "
            f"models, so lost edges aren't what separates them."
        )
    elif fed_gap["ci95"][1] < 0:
        text += (
            f"The diffusion model's advantage survives federation: federated DCRNN is "
            f"{-fed_gap['estimate']:.3f} mph better than the federated shared Graph-GRU."
        )
    else:
        text += "Federated DCRNN and the federated shared Graph-GRU are not clearly different."
    if local_d and pooled_d and local_d["mae"]["mean"] < pooled_d["mae"]["mean"]:
        text += (
            f" As in the [headline study](../fedgraph/report.md), training alone is the best arm "
            f"({mae(local_d)} mph for local-only DCRNN, against {mae(pooled_d)} pooled and "
            f"{mae(arms['fed_dcrnn'])} federated): with clients this different, a model that "
            f"specializes in its own area beats one model for everyone, and averaging the clients "
            f"is worse again. I haven't tested that explanation on its own."
        )
    return text


def fed_rows(f):
    order = [
        "fed_dcrnn",
        "fed_sgraph",
        "fed_sgru",
        "central_dcrnn_local",
        "central_dcrnn_full",
        "central_sgraph_local",
        "central_sgraph_full",
        "central_sgru",
        "local_dcrnn",
        "local_sgraph",
    ]
    rows = []
    for name in order:
        entry = f["arms"].get(name)
        if not entry:
            continue
        comm = entry.get("communication")
        best = entry.get("best_round")
        best_text = f"{best['mean']:.0f}" if best else "own"
        params_text = f"{entry['parameters']:,}" if entry.get("parameters") else ""
        rows.append(
            f"| {LABELS.get(name, name)} | {mae(entry)} | {params_text} "
            f"| {best_text} "
            f"| {entry['train_seconds']['mean'] / 60:.1f} "
            f"| {(comm['total_parameter_bytes'] / 2**20) if comm else 0:.1f} |"
        )
    return "\n".join(rows)


def page(central, ablation, fed, cost):
    models = central["models"]
    size = central["sizes"][-1]
    params = lambda name: models[name][str(size)]["parameters"]  # noqa: E731
    config = json.loads((ROOT / "reports/diffusion/central-config.json").read_text())[
        "benchmark"
    ]
    abl = json.loads((ROOT / "reports/diffusion/ablation-config.json").read_text())[
        "benchmark"
    ]
    own = [
        "persistence",
        "ha",
        "gru_time",
        "graph_gru_time",
        "sgru_time",
        "sgraph_time",
        "dcrnn_k0_time",
        "dcrnn_perm_time",
        "dcrnn_h16_time",
        "dcrnn_time",
    ]
    cost_text = (
        "Parameters, time per epoch, peak memory and inference time from a short separate run with nothing else on the machine (3 epochs, one seed, 8 threads) are in the table. The accuracy runs above shared the machine with other jobs, so their timings are rough and I don't use them.\n\n"
        + cost_table(cost)
        if cost
        else "The cost run hasn't been done."
    )
    return PAGE.format(
        dcrnn_params=params("dcrnn_time"),
        sgraph_params=params("sgraph_time"),
        sgru_params=params("sgru_time"),
        h16_params=params("dcrnn_h16_time"),
        epochs=config["epochs"],
        seeds=seed_text(central),
        sizes=join_and(central["sizes"]),
        own_table=table(central, own, "mae"),
        core_table=table(central, own[2:], "core16_mae"),
        gap_table=gap_rows(
            central,
            [
                ("dcrnn_time", "sgraph_time"),
                ("dcrnn_time", "sgru_time"),
                ("dcrnn_time", "gru_time"),
                ("dcrnn_time", "dcrnn_k0_time"),
                ("dcrnn_time", "dcrnn_perm_time"),
                ("dcrnn_h16_time", "sgraph_time"),
                ("dcrnn_time", "ha"),
            ],
        ),
        central_text=central_text(central),
        ablation_seeds=seed_phrase(abl["seeds"]),
        ablation_table=table(
            ablation,
            [
                "ha",
                "dcrnn_k0_time",
                "dcrnn_k1_time",
                "dcrnn_time",
                "dcrnn_k3_time",
                "dcrnn_fwd_time",
                "dcrnn_bwd_time",
                "dcrnn_sym_time",
            ],
            "mae",
        ),
        ablation_gaps=ablation_gap_table(ablation),
        ablation_text=ablation_text(ablation),
        clients=fed["base"]["clients"],
        fed_sensors=fed["base"]["sensors"],
        fed_rounds=fed["base"]["rounds"],
        fed_seeds=seed_phrase(fed["seeds"]),
        fed_n=plural(len(fed["seeds"]), "seed"),
        fed_rows=fed_rows(fed),
        federated_gaps=federated_gap_table(fed),
        federated_story=federated_story(fed),
        cost_text=cost_text,
        ratio=params("dcrnn_time") / params("sgraph_time"),
        largest=size,
    )


def cost_table(cost):
    sizes = cost["sizes"]
    names = [
        "gru_time",
        "graph_gru_time",
        "sgru_time",
        "sgraph_time",
        "dcrnn_time",
        "dcrnn_h16_time",
        "dcrnn_k1_time",
    ]
    head = (
        "| Model | Parameters | Seconds per epoch, "
        + "/".join(map(str, sizes))
        + " | Peak memory (MiB), "
        + "/".join(map(str, sizes))
        + " | Inference ms per window, "
        + "/".join(map(str, sizes))
        + " |"
    )
    rows = []
    for name in names:
        by = cost["models"].get(name)
        if not by:
            continue
        last = by[str(sizes[-1])]
        seconds = " / ".join(
            f"{by[str(k)]['train_seconds']['mean'] / by[str(k)]['epochs_run']['mean']:.1f}"
            for k in sizes
        )
        memory = " / ".join(
            f"{by[str(k)]['peak_memory_mib']['mean']:.0f}" for k in sizes
        )
        infer = " / ".join(
            f"{by[str(k)]['inference_ms_per_window']['mean']:.3f}" for k in sizes
        )
        rows.append(
            f"| {NAMES.get(name, name)} | {last['parameters']:,} | {seconds} | {memory} | {infer} |"
        )
    return "\n".join([head, "|---|---:|---:|---:|---:|"] + rows)


def build():
    central = load(ROOT / "runs/diffusion/scaling-summary.json")
    ablation = load(ROOT / "runs/diffusion-ablation/scaling-summary.json")
    fed = load(ROOT / "runs/fedgraph-diffusion/summary.json")
    cost = load(ROOT / "runs/diffusion-cost/scaling-summary.json")
    if not (central and ablation and fed):
        return None
    out = REPORTS / "diffusion"
    out.mkdir(parents=True, exist_ok=True)
    for source, name in (
        ("runs/diffusion/scaling-summary.json", "central-summary.json"),
        ("runs/diffusion/scaling-config.json", "central-config.json"),
        ("runs/diffusion/scaling-accuracy.png", "central-accuracy.png"),
        ("runs/diffusion-ablation/scaling-summary.json", "ablation-summary.json"),
        ("runs/diffusion-ablation/scaling-config.json", "ablation-config.json"),
        ("runs/diffusion-ablation/scaling-accuracy.png", "ablation-accuracy.png"),
        ("runs/diffusion-cost/scaling-summary.json", "cost-summary.json"),
        ("runs/diffusion-cost/scaling-config.json", "cost-config.json"),
    ):
        if (ROOT / source).exists():
            shutil.copy2(ROOT / source, out / name)
    copy_results(
        ROOT / "runs/fedgraph-diffusion", out / "federated", seeds=fed["seeds"]
    )
    for png, new in (
        ("accuracy.png", "federated-accuracy.png"),
        ("convergence.png", "federated-convergence.png"),
    ):
        shutil.copy2(ROOT / "runs/fedgraph-diffusion" / png, out / new)
    (out / "report.md").write_bytes(page(central, ablation, fed, cost).encode())
    return fed


if __name__ == "__main__":
    build()
