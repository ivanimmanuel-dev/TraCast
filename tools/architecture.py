"""Draw the shared-weight graph model as a schematic (not a result)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE, TEAL, GREY, ORANGE = "#1d4ed8", "#087e8b", "#667085", "#d97706"


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


def arrow(ax, start, end, color=GREY, style="-|>", rad=0.0):
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


def main():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set(xlim=(0, 11), ylim=(0, 4.2))
    ax.axis("off")

    # the road graph: a small corridor of sensors, one of them highlighted
    xs = [0.5, 1.15, 1.8, 2.45, 3.1]
    for i, x in enumerate(xs):
        ax.add_patch(
            plt.Circle((x, 3.2), 0.17, color=ORANGE if i == 2 else GREY, zorder=3)
        )
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 0.17, 3.25), (b - 0.17, 3.25), GREY, "->", 0.0)
        arrow(ax, (b - 0.17, 3.15), (a + 0.17, 3.15), GREY, "->", 0.0)
    ax.text(
        1.8,
        3.65,
        "Road graph, 16 to 207 sensors",
        ha="center",
        fontsize=9,
        color="#111827",
    )
    ax.text(
        1.8,
        2.75,
        "every sensor runs the same network",
        ha="center",
        fontsize=8,
        color=GREY,
    )

    box(
        ax,
        0.2,
        1.35,
        3.2,
        0.95,
        "last 12 steps of speeds\n(normalized per sensor)",
        GREY,
    )
    arrow(ax, (1.8, 2.95), (1.8, 2.32), GREY)

    # features of one sensor
    labels = [
        ("own speed", BLUE),
        ("mean of upstream\nneighbors", TEAL),
        ("mean of downstream\nneighbors", TEAL),
        ("time of day (sin, cos)\nweekend flag", ORANGE),
    ]
    for i, (text, color) in enumerate(labels):
        box(ax, 4.0, 3.2 - i * 0.85, 2.3, 0.62, text, color, size=8)
        arrow(ax, (3.4, 1.8), (4.0, 3.5 - i * 0.85), GREY, "-|>", 0.0)
    ax.text(
        5.15,
        3.95,
        "features per sensor and step",
        ha="center",
        fontsize=9,
        color="#111827",
    )

    box(
        ax,
        6.9,
        1.35,
        1.9,
        1.5,
        "GRU, 32 units\n\nsame weights\nfor every sensor",
        BLUE,
        size=8.5,
        fill="#eef2ff",
    )
    for i in range(4):
        arrow(ax, (6.3, 3.5 - i * 0.85), (6.9, 2.1), GREY, "-|>", 0.0)
    box(ax, 9.2, 1.7, 1.5, 0.8, "linear head", BLUE, size=8.5)
    arrow(ax, (8.8, 2.1), (9.2, 2.1), BLUE)
    for i in range(12):
        ax.add_patch(
            plt.Rectangle((9.25 + i * 0.1, 0.55), 0.07, 0.25 + 0.015 * i, color=TEAL)
        )
    ax.text(
        9.95, 0.25, "12 forecasts for this sensor", ha="center", fontsize=8, color=GREY
    )
    arrow(ax, (9.95, 1.7), (9.95, 0.95), TEAL)
    ax.text(
        7.85,
        0.55,
        "Parameters: about 4,000,\nwhatever the number of sensors",
        ha="center",
        fontsize=8.5,
        color="#111827",
    )
    ax.text(
        5.5,
        0.25,
        "Without the two neighbor features this is the shared GRU baseline.",
        ha="center",
        fontsize=8,
        color=GREY,
    )
    fig.savefig(
        ROOT / "reports/figures/shared-architecture.png", dpi=170, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
