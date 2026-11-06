"""Schematics and the partition map for the federated graph experiments (not results).

The map is drawn from the real sensor coordinates, the real road graph and the same
partition rule the experiments use, so it shows exactly which edges each client keeps.
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracast import fedgraph as F  # noqa: E402

BLUE, TEAL, GREY, ORANGE, RED = "#1d4ed8", "#087e8b", "#667085", "#d97706", "#dc2626"
PALETTE = [
    "#1d4ed8",
    "#d97706",
    "#087e8b",
    "#9333ea",
    "#dc2626",
    "#16a34a",
    "#a16207",
    "#0ea5e9",
    "#e11d48",
    "#475569",
    "#65a30d",
    "#c026d3",
]


def box(ax, x, y, w, h, text, color, size=8.5, fill="#f4f8fa"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="#111827",
    )


def arrow(ax, start, end, color=GREY, rad=0.0, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            color=color,
            linewidth=1.2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def partition_map(config_path, out):
    settings = json.loads(Path(config_path).read_text())
    config = F.GraphFedConfig(**settings["base"])
    prep = F.prepare(config, ROOT / "data/raw/metr-la.h5", ROOT / "data/topology")
    ids, weights, groups = prep["ids"], prep["weights"], prep["groups"]
    meta = prep["graph_meta"]["locations"]
    x = np.array([meta[s]["longitude"] for s in ids])
    y = np.array([meta[s]["latitude"] for s in ids])
    owner = np.empty(len(ids), dtype=int)
    for client, group in enumerate(groups):
        owner[group] = client
    fig, ax = plt.subplots(figsize=(8, 6.2), constrained_layout=True)
    sources, targets = np.nonzero(weights)
    kept = owner[sources] == owner[targets]
    for a, b in zip(sources[~kept], targets[~kept]):
        ax.plot([x[a], x[b]], [y[a], y[b]], color=RED, alpha=0.5, lw=0.8, zorder=1)
    for a, b in zip(sources[kept], targets[kept]):
        ax.plot(
            [x[a], x[b]], [y[a], y[b]], color="#334155", alpha=0.55, lw=0.8, zorder=1
        )
    for client, group in enumerate(groups):
        ax.scatter(
            x[group],
            y[group],
            s=26,
            color=PALETTE[client % len(PALETTE)],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=f"Client {client} ({len(group)} sensors)",
        )
    report = F.graph_report(weights, groups)
    ax.plot([], [], color="#334155", lw=1.2, label="Edge a client keeps")
    ax.plot([], [], color=RED, lw=1.2, label="Edge between clients (dropped)")
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title=(
            f"{len(ids)} sensors in {config.clients} geographic clients: "
            f"{report['edges_kept_fraction']:.0%} of {report['directed_edges']:,} "
            "directed edges stay inside a client"
        ),
    )
    ax.title.set_fontsize(9)
    ax.legend(fontsize=7, loc="lower left", frameon=True)
    ax.set_aspect(1 / np.cos(np.radians(float(np.mean(y)))))
    fig.savefig(Path(out) / "partition-map.png", dpi=150)
    plt.close(fig)
    (Path(out) / "partition.json").write_text(
        json.dumps(
            {
                "rule": config.partition,
                "seed": config.partition_seed,
                "clients": [
                    {"client": i, "sensors": [ids[j] for j in g]}
                    for i, g in enumerate(groups)
                ],
                "graph": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def protocol(out):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set(xlim=(0, 11), ylim=(-0.9, 4.6))
    ax.axis("off")
    box(
        ax,
        4.0,
        3.2,
        3.0,
        1.1,
        "Flower server\nFedAvg or FedProx\naverages the parameters",
        BLUE,
    )
    centers = (1.45, 4.15, 6.85, 9.55)
    for i, cx in enumerate(centers):
        box(
            ax,
            cx - 1.15,
            0.3,
            2.3,
            1.6,
            f"Client {i if i < 3 else 'K'}\nown sensors' readings\nlocal road graph\nlocal training",
            TEAL,
            size=7.8,
        )
        arrow(ax, (4.6 + 0.8 * i / 3, 3.2), (cx - 0.18, 1.9), BLUE)
        arrow(ax, (cx + 0.18, 1.9), (5.0 + 0.8 * i / 3, 3.2), ORANGE)
    ax.text(0.2, 3.9, "global parameters (down)", color=BLUE, fontsize=8)
    ax.text(0.2, 3.55, "updated parameters (up)", color=ORANGE, fontsize=8)
    ax.text(
        5.5,
        -0.45,
        "Only the learned weights travel, about 17 KB each way. Readings and road graphs never leave a client.\n"
        "One simulated machine; Flower runs the clients in turn.",
        ha="center",
        fontsize=8.5,
        color="#111827",
    )
    fig.savefig(Path(out) / "protocol.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def dcrnn(out):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set(xlim=(0, 11.5), ylim=(-0.6, 4.8))
    ax.axis("off")
    # a small directed corridor with one sensor highlighted
    xs = [0.5, 1.2, 1.9, 2.6, 3.3]
    for i, x in enumerate(xs):
        ax.add_patch(
            plt.Circle((x, 3.7), 0.17, color=ORANGE if i == 2 else GREY, zorder=3)
        )
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 0.17, 3.77), (b - 0.17, 3.77), TEAL)
        arrow(ax, (b - 0.17, 3.63), (a + 0.17, 3.63), RED)
    ax.text(1.9, 4.2, "directed road graph", ha="center", fontsize=9)
    ax.text(0.5, 3.15, "forward walk (with the road)", color=TEAL, fontsize=7.5)
    ax.text(0.5, 2.9, "reverse walk (against it)", color=RED, fontsize=7.5)
    box(
        ax,
        4.2,
        2.7,
        3.0,
        1.5,
        "Diffusion convolution\n[x, P_f x, P_f^2 x, P_r x, P_r^2 x]\n-> one linear layer\n(hops per direction = K)",
        BLUE,
        size=8,
    )
    arrow(ax, (3.6, 3.7), (4.2, 3.5))
    box(
        ax,
        4.2,
        0.4,
        3.0,
        1.7,
        "DCGRU cell\nreset and update gates =\ndiffusion conv of [x_t, h]\ncandidate = diffusion conv\nof [x_t, reset * h]",
        TEAL,
        size=8,
    )
    arrow(ax, (5.7, 2.7), (5.7, 2.1))
    box(
        ax,
        7.9,
        0.4,
        1.4,
        1.7,
        "hidden state\nper sensor\n(12 steps\nof input)",
        GREY,
        size=8,
    )
    arrow(ax, (7.2, 1.25), (7.9, 1.25))
    arrow(ax, (8.6, 0.4), (5.7, 0.4), GREY, rad=-0.35)
    ax.text(
        7.15,
        -0.3,
        "state fed back at the next input step",
        ha="center",
        fontsize=7.5,
        color=GREY,
    )
    box(
        ax,
        9.8,
        0.4,
        1.4,
        1.7,
        "linear head\n-> 12 horizons\nper sensor",
        ORANGE,
        size=8,
    )
    arrow(ax, (9.3, 1.25), (9.8, 1.25))
    ax.text(
        7.7,
        4.45,
        "same weights for every sensor, so the model has the same size on any graph",
        ha="center",
        fontsize=8.5,
        color="#111827",
    )
    fig.savefig(Path(out) / "dcrnn-architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    for name in ("fedgraph", "diffusion", "fedopt", "holdout"):
        (ROOT / "reports" / name).mkdir(parents=True, exist_ok=True)
    partition_map(ROOT / "configs/fedgraph.json", ROOT / "reports/fedgraph")
    protocol(ROOT / "reports/fedgraph")
    dcrnn(ROOT / "reports/diffusion")
    print("figures written")


if __name__ == "__main__":
    main()
