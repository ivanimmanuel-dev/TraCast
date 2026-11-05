"""Build the technical report and the two-page brief from the saved results."""

from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

import presentation as P

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LABELS = {
    "persistence": "Persistence",
    "gru": "GRU",
    "lstm": "LSTM",
    "graph_gru": "Graph-GRU",
    "graph_self": "Graph-GRU, self only",
}
ORDER = ["persistence", "gru", "lstm", "graph_gru", "graph_self"]
NL = chr(10)


def value(item, places=3):
    suffix = f" ({item['sd']:.{places}f})" if item["sd"] is not None else ""
    return f"{item['mean']:.{places}f}" + suffix


REPORT = """# TraCast: traffic speed forecasting on METR-LA

Technical report, November 2026

## Summary

{summary}

## 1. What I wanted to find out

This started as a way to learn how traffic forecasting works from end to end. I ended up with four questions:

1. Do GRU and LSTM models beat simple baselines like persistence, a historical average and a linear model?
2. Does the road graph help?
3. Does knowing the time of day help?
4. Does federated training lose anything compared with training on all the data in one place, or with each client training alone?

To keep the comparisons fair I wrote one pipeline (data checks, split, normalization, masks, metrics) and ran every model through it.

## 2. Data

METR-LA has speed readings from 207 loop detectors in Los Angeles every five minutes from March 1 to June 27, 2012: 34,272 timestamps and 7,094,304 readings. The file comes from a Hugging Face copy pinned to one revision and checked against a SHA-256 hash ([DATA.md](../DATA.md) has the sources and licenses). On load I check the timestamps, the five-minute spacing, the sensor IDs and that the values are numeric, finite and non-negative. Timestamps are kept as given because the data doesn't say which timezone they use.

Most experiments use 16 sensors (548,352 readings), picked from the road graph (section 3). Rows are split in time: the first 70% is train (March 1 to May 23), the next 10% validation (to June 4) and the last 20% test (June 4 to June 27). Windows are built after the split, so none crosses a boundary. That leaves 23,967 train, 3,404 validation and 6,832 test windows. A model sees 12 steps (one hour) and predicts the next 12.

Zeros and NaNs count as missing. Inputs are forward-filled within their split (leading gaps get the sensor's training mean), each sensor is normalized with the mean and standard deviation of its observed training readings, and missing targets count toward neither the loss nor the scores. Treating zero as missing can throw away real stopped traffic. For a given seed, every model gets the same windows, statistics, masks and batch order.

## 3. Models

**Baselines.** Persistence repeats the last reading. The historical average predicts the training mean for the same sensor, weekday and five-minute slot, and uses no recent readings. The linear model is one linear layer from the flattened input window to all 12 x 16 forecasts.

**GRU and LSTM.** One recurrent layer with 32 hidden units reads all 16 sensors at each step. A linear head turns the last hidden state into all 12 forecast steps for all 16 sensors at once.

**Graph-GRU.** The graph comes from the DCRNN authors' distance table (revision `{revision}`). Only pairs between the 207 METR-LA sensors are used. Their finite costs set the standard deviation s of a Gaussian kernel, W[i,j] = exp(-(D[i,j] / s)^2), which I keep when it is at least 0.1 and drop for self pairs. That gives 1,515 directed edges among the 207 sensors. I picked the 16 sensors by walking the graph breadth-first from the first sensor, following edges in either direction and taking stronger ones first, which gives a connected subset with {edges} directed edges. Neither the choice of sensors nor the weights use any speed data.

For each sensor and time step, Graph-GRU takes its own normalized speed plus two averages: one over the sensors that feed into it and one over the sensors it feeds, both weighted by the edge weights. A small shared layer (Linear 3 to 4, then ReLU) turns those three numbers into four features per sensor. The 16 x 4 features go into the GRU and the same kind of linear head. "Self only" zeroes the two averages and changes nothing else, so it has the same 15,760 parameters. Every model here reads all 16 sensors, so this ablation tests the graph-weighted averages, not whether neighboring sensors matter.

![Selected sensors and graph edges](benchmark/topology.png)

**Calendar inputs.** Models marked "+ calendar" get three extra inputs at every step: the sine and cosine of the time of day, and a weekend flag.

**Training.** All learned models use batch size 64, Adam with learning rate 0.001, masked MAE in normalized units and gradient clipping at 5. The checkpoint with the lowest validation MAE is the one scored on the test set.

Everything ran on a Ryzen 7 5825U laptop (8 cores, about 14 GB of RAM) with 4 PyTorch threads, Python 3.11.9, CPU PyTorch 2.5.1, NumPy 1.26.4 and pandas 2.2.3. MKLDNN is off and deterministic algorithms are on, so a repeat with the same seed gives the same numbers.

## 4. First pass: five epochs

I started with five epochs per learned model and three seeds (7, 42, 2026). Training time sums the update loops, including batch loading and excluding validation. Forward time is the median of 30 passes over the first test batch, per window, after 5 warmups. Peak memory (Windows PeakWorkingSetSize for the whole process) was about 320 MiB for every model.

| Model | MAE (SD) | RMSE (SD) | Parameters | Train time, s | Forward, ms/window |
|---|---:|---:|---:|---:|---:|
{first_pass_rows}

LSTM and Graph-GRU end up at about the same MAE, and the LSTM has fewer parameters. Graph-GRU minus self-only is {ablation} mph by seed (mean {ablation_mean:+.4f}), so the neighbor averages barely matter and the sign changes between seeds. Most of Graph-GRU's lead over the GRU stays when the averages are zeroed.

![Forecast error by horizon](benchmark/horizons.png)

![Validation and training convergence](benchmark/convergence.png)

Five epochs turned out to be too short. In {at_last} of {n_runs} runs the best validation epoch was the last one, so the models were still improving when training stopped.

## 5. Training longer, and adding calendar inputs

{intro}

| Model | MAE (SD) | MAE range over test days | 15 min | 30 min | 60 min | RMSE (SD) | Parameters | Epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

MAE and RMSE are test means in mph; 15/30/60 min are MAE at those horizons. The range column resamples the {days} test days to show how much a score depends on which days ended up in the test set. Persistence and the historical average aren't trained, so they run once.

{narrative}

| Difference in MAE (first - second) | Estimate (mph) | 95% CI over test days |
|---|---:|---:|
{gaps}

Negative means the first model is better. Both models in a row are scored on the same windows and the same resampled days. The intervals are about which days were held out; the seed SDs in the table above cover random initialization.

![Forecast error by horizon, longer-training study](figures/extended-horizons.png)

## 6. Federated learning

The centralized models above read all 16 sensors together. For federated learning I wanted a model that means the same thing on every client, so the federated task uses a univariate GRU shared across sensors: one sensor per example, 3,756 parameters (15,024 bytes).

The setup is twelve sensors, the first 4,032 timestamps (fourteen days), three clients with four sensors each (no sensor on two clients), three seeds (7, 42, 2026) and one local epoch per round. The runtime is Flower 1.42.0 (`ClientApp`, `ServerApp`, `FedAvg`) through `flwr run` on Ray, with all clients on one machine. Each sensor is normalized with its own training statistics. The server averages weights, weighted by each client's number of usable training examples (examples with at least one observed target). Errors are pooled as sums and counts before computing MAE and RMSE.

There are two references. Centralized trains one model on all clients' examples with the same starting weights, scalers, masks and total number of examples. Local-only trains each client by itself with the same local work and no communication, and each client keeps the round with its own best validation MAE. Optimizers restart at every local fit, and no optimizer state is shared.

Sensors are ranked by the fraction of their training readings below 40 mph. Balanced groups interleave those ranks across clients and congestion-grouped clients get consecutive blocks. The mean Jensen-Shannon divergence between each client's training speed histogram and the pooled one is {js_balanced:.4f} bits for balanced groups and {js_congestion:.4f} for congestion groups.

{fed_header}
{fed_rows}

{fed_compare} {fed_verdict}

{fed_cost} Congestion grouping raises MAE by {congestion_mae:.4f} mph at five rounds while lowering RMSE by {congestion_rmse:.4f}, so the change in the speed distribution doesn't hit both measures the same way.

As a check, the centralized reference is identical for both ownership rules, and for each seed the global model after r rounds matches the one from the longer runs. [Client assignments, per-seed numbers and the communication count](federated-study/findings.md).

![Communication and convergence](federated-study/tradeoffs.png)

## 7. What I take from this

- **Give the model the clock.** Time of day and a weekend flag beat every architecture change I tried. By 30 minutes ahead, a plain historical average already beats persistence.
- **The graph wasn't worth it here.** The neighbor averages helped a little without the clock and hurt a little with it, and Graph-GRU with the clock is not clearly different from a GRU with the clock.
- **Check training length before comparing models.** After five epochs the LSTM and Graph-GRU both looked better than the GRU by about {five_epoch_gain:.3f} mph. With proper training the LSTM is level with the GRU ({lstm_gap:+.3f}) and Graph-GRU is {graph_gap:.3f} ahead, a lead that disappears once the clock is added.
- **Federated averaging didn't beat training alone** for this small model and short slice, and pooling the data was best.

What this doesn't cover: one 16-sensor subset picked by a graph walk (not representative of every road, and neighbors outside the subset are missing), one 24-day test window (scores move by about 0.3 mph with the choice of days, though the gaps between models are much tighter), no hyperparameter search, and a federated task that is univariate and simulated on one machine, so its errors shouldn't be compared with the centralized tables. Timings are rough, and the byte counts only cover model weights.

## 8. Reproducing it

The [README](../README.md) and the [run guide](../docs/reproducibility.md) have the commands. They fetch checksum-verified data and use pinned dependencies. Each experiment's folder in `reports/` has its config, source and data hashes, sensor IDs, split ranges, scaler values, histories and per-horizon counts. Unit tests run on small constructed arrays, and the experiment commands are the real-data runs.

## References

1. Li, Y., Yu, R., Shahabi, C., Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR. https://arxiv.org/abs/1707.01926
2. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP. https://arxiv.org/abs/1406.1078
3. Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. McMahan, H. B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS. https://arxiv.org/abs/1602.05629
5. Beutel, D. J. et al. (2020). Flower: A Friendly Federated Learning Research Framework. https://arxiv.org/abs/2007.14390
"""

BENCHMARK_PAGE = """# Five-epoch benchmark

My first benchmark: 16 sensors picked from the METR-LA distance graph, the full March to June timeline, seeds 7, 42 and 2026, and five epochs per learned model. Each trial runs in a fresh CPU process. It turned out five epochs is too short, which is why there is a [longer-training study](../benchmark-extended/report.md).

The [technical report](../research-report.md) covers the data, models and results.

| File | What's in it |
|---|---|
| [summary.json](summary.json) | Mean and SD per model, horizon metrics, paired differences |
| [results.json](results.json) | Per-seed scores, convergence, timings, model hashes |
| [protocol.json](protocol.json) | Sensor IDs, splits, scalers, graph weights, missing counts |
| [manifest.json](manifest.json) | Source, dependency and input hashes |
| [config.json](config.json) | Resolved model and training settings |
| [repeatability.json](repeatability.json) | Exact repeats of the seed-42 Graph-GRU and LSTM runs |

![Topology](topology.png)

![Validation convergence](../figures/validation-convergence.png)

To run it, see the [run guide](../../docs/reproducibility.md#five-epoch-benchmark).
"""


def bold(text):
    parts = text.split("**")
    return "".join(f"<b>{part}</b>" if i % 2 else part for i, part in enumerate(parts))


def build_report():
    first_pass = json.loads((REPORTS / "benchmark/summary.json").read_text())["models"]
    results = json.loads((REPORTS / "benchmark/results.json").read_text())
    protocol = json.loads((REPORTS / "benchmark/protocol.json").read_text())
    ext = P.extended()
    conditions = P.load("reports/federated-study/summary.json")["conditions"]
    fed = P.federated(conditions)
    m, found = ext["models"], ext["found"]
    delta = (
        first_pass["graph_gru"]["mae"]["mean"] - first_pass["graph_self"]["mae"]["mean"]
    )
    ablation = [
        r["test"]["mae"]
        - next(
            s["test"]["mae"]
            for s in results
            if s["seed"] == r["seed"] and s["model"] == "graph_self"
        )
        for r in results
        if r["model"] == "graph_gru"
    ]
    best_epochs = [
        (r["best_epoch"], max(h["epoch"] for h in r["history"]))
        for r in results
        if r["history"]
    ]
    names = [n for n in P.EXTENDED_LABELS if n in m]
    gru, gru_t = m["gru"], m["gru_time"]
    summary = (
        f"I compared simple baselines, GRU and LSTM models and a graph-aware GRU for "
        f"forecasting traffic speed 5 to 60 minutes ahead on METR-LA, using 16 sensors and "
        f"a chronological split. Every model beats persistence "
        f"({m['persistence']['mae']['mean']:.3f} mph MAE); the best reaches "
        f"{m[P.top_three(m)[0]]['mae']['mean']:.3f}. The biggest "
        f"single improvement came from giving the model the time of day (the GRU goes from "
        f"{gru['mae']['mean']:.3f} to {gru_t['mae']['mean']:.3f}), not from changing the "
        f"architecture or adding the road graph, which changes MAE by "
        f"{found['graph_gru_vs_graph_self']['estimate']:+.3f} mph without the clock and "
        f"{found['graph_gru_time_vs_graph_self_time']['estimate']:+.3f} with it. A "
        f"federated simulation (three clients, twelve sensors) ends up close to training "
        f"each client alone."
    )
    table = NL.join(
        f"| {LABELS[n]} | {value(first_pass[n]['mae'])} | {value(first_pass[n]['rmse'])} "
        f"| {first_pass[n]['parameter_count']:,} | {value(first_pass[n]['train_seconds'], 2)} "
        f"| {value(first_pass[n]['inference_ms_per_window'], 4)} |"
        for n in ORDER
    )
    balanced, grouped = conditions["balanced-5"], conditions["congestion-5"]
    text = REPORT.format(
        summary=summary,
        revision=protocol["graph"]["revision"],
        edges=protocol["graph"]["subset_directed_edges"],
        first_pass_rows=table,
        ablation=", ".join(f"{d:+.4f}" for d in ablation),
        ablation_mean=delta,
        at_last=sum(best == last for best, last in best_epochs),
        n_runs=len(best_epochs),
        intro=P.extended_intro(ext["config"]),
        rows=P.extended_rows(ext, names),
        days=next(iter(found.values()))["days"],
        narrative=P.extended_narrative(ext, first_pass),
        gaps=P.gap_rows(ext),
        js_balanced=balanced["mean_training_js_bits"],
        js_congestion=grouped["mean_training_js_bits"],
        fed_header=fed["header"],
        fed_rows=fed["rows"],
        fed_compare=fed["compare"],
        fed_verdict=fed["verdict"],
        fed_cost=fed["cost"],
        five_epoch_gain=first_pass["gru"]["mae"]["mean"]
        - (first_pass["lstm"]["mae"]["mean"] + first_pass["graph_gru"]["mae"]["mean"])
        / 2,
        lstm_gap=found["lstm_vs_gru"]["estimate"],
        graph_gap=-found["graph_gru_vs_gru"]["estimate"],
        congestion_mae=grouped["fedavg_mae"]["mean"] - balanced["fedavg_mae"]["mean"],
        congestion_rmse=balanced["fedavg_rmse"]["mean"]
        - grouped["fedavg_rmse"]["mean"],
    )
    (REPORTS / "research-report.md").write_bytes(text.encode())
    (REPORTS / "benchmark/report.md").write_text(
        BENCHMARK_PAGE, encoding="utf-8", newline="\n"
    )
    return first_pass, ext, fed


def build_brief(first_pass, ext, fed):
    m = ext["models"]
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=6
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletSmall",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            leftIndent=10,
            bulletIndent=0,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionSmall",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#176b87"),
            spaceBefore=9,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionSmall",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#536471"),
            spaceAfter=6,
        )
    )

    def body(text):
        return Paragraph(bold(text), styles["BodySmall"])

    def heading(text):
        return Paragraph(text, styles["SectionSmall"])

    story = [
        Paragraph("TraCast", styles["Title"]),
        body("**Traffic speed forecasting on METR-LA**"),
        body(
            "ivanimmanuel-dev / TraCast &nbsp; | &nbsp; Research brief, November 2026"
        ),
        heading("What I wanted to know"),
        body(
            "Do GRU and LSTM models beat simple baselines? Does the road graph help? "
            "Does knowing the time of day help? And does federated training lose "
            "anything compared with training on all the data in one place?"
        ),
        heading("What I found"),
    ]
    for bullet in P.findings(ext, first_pass, fed):
        story.append(Paragraph(bold(bullet[2:]), styles["BulletSmall"], bulletText="•"))
    rows = [["Model", "MAE (SD)", "15 min", "60 min"]] + [
        [
            P.EXTENDED_LABELS[n],
            P.mean_sd(m[n]["mae"]),
            f"{P.at(m[n], 15):.3f}",
            f"{P.at(m[n], 60):.3f}",
        ]
        for n in P.COMPACT
    ]
    t = Table(rows, colWidths=[170, 110, 80, 80], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f1f4")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#176b87")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    story += [
        heading("Results"),
        t,
        Paragraph(
            f"Test MAE in mph, 16 sensors, 24 test days; learned models average "
            f"{len(ext['config']['seeds'])} seeds.",
            styles["CaptionSmall"],
        ),
        Image(str(REPORTS / "figures/extended-horizons.png"), width=478, height=182),
        PageBreak(),
        heading("Setup"),
        body(
            "METR-LA has 34,272 five-minute readings from 207 sensors (March 1 to "
            "June 27, 2012). I use 16 of them, picked by walking the DCRNN distance "
            "graph. Rows are split in time: 70% train, 10% validation, 20% test. A model "
            "sees one hour and predicts the next hour. Zero and missing readings are left "
            "out of the loss and the scores, and normalization uses training data only."
        ),
        body(
            "Graph-GRU gives each sensor its own speed plus the average of the sensors "
            "upstream and downstream of it, passes that through a small shared layer and "
            "feeds all 16 sensors into a GRU."
        ),
    ]
    drawing = Drawing(478, 62)
    boxes = [
        (0, 64, "Own speed +", "in/out means"),
        (76, 110, "Shared Linear(3,4)", "ReLU per node"),
        (198, 104, "Flatten 16 nodes", "GRU hidden 32"),
        (314, 164, "Linear forecast head", "12 steps x 16 sensors"),
    ]
    for x, w, a, b in boxes:
        drawing.add(
            Rect(
                x,
                10,
                w,
                44,
                rx=4,
                ry=4,
                fillColor=colors.HexColor("#eef5f7"),
                strokeColor=colors.HexColor("#176b87"),
                strokeWidth=0.6,
            )
        )
        drawing.add(
            String(
                x + w / 2,
                36,
                a,
                textAnchor="middle",
                fontName="Helvetica",
                fontSize=7.5,
            )
        )
        drawing.add(
            String(
                x + w / 2,
                23,
                b,
                textAnchor="middle",
                fontName="Helvetica",
                fontSize=7.5,
            )
        )
        if x:
            drawing.add(
                Line(x - 10, 32, x - 2, 32, strokeColor=colors.HexColor("#176b87"))
            )
            drawing.add(
                Polygon(
                    [x - 2, 32, x - 6, 35, x - 6, 29],
                    fillColor=colors.HexColor("#176b87"),
                    strokeColor=None,
                )
            )
    story += [
        drawing,
        body(
            '"Self only" zeroes the two averages. Every model reads all 16 sensors, so '
            "that tests the graph-weighted averages, not whether neighbors matter."
        ),
        heading("Federated learning"),
        body(
            f"Three simulated clients each hold four of twelve sensors and train the same "
            f"small GRU with Flower; the server averages the weights. I compare it with "
            f"training on all the data together and with each client training alone. "
            f"{fed['compare']} {fed['verdict']}"
        ),
        body(fed["cost"]),
        heading("What this doesn't cover"),
        body(
            "One 16-sensor subset, one 24-day test window, no hyperparameter search, and "
            "a federated task that is univariate and simulated on one machine."
        ),
        heading("Code"),
        body(
            '<link href="https://github.com/ivanimmanuel-dev/TraCast">github.com/ivanimmanuel-dev/TraCast</link><br/>'
            "The technical report has the details; DATA.md has the data sources."
        ),
        Paragraph(
            "References: Li et al., ICLR 2018 (METR-LA, DCRNN); Cho et al., EMNLP 2014 (GRU); Hochreiter and Schmidhuber, 1997 (LSTM); McMahan et al., AISTATS 2017 (FedAvg); Flower 1.42.0.",
            styles["CaptionSmall"],
        ),
    ]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536471"))
        canvas.drawString(36, 24, "TraCast | Research brief | November 2026")
        canvas.drawRightString(576, 24, str(doc.page))

    doc = SimpleDocTemplate(
        str(REPORTS / "research-brief.pdf"),
        pagesize=(612, 792),
        leftMargin=36,
        rightMargin=36,
        topMargin=30,
        bottomMargin=38,
        title="TraCast research brief",
        author="ivanimmanuel-dev",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(REPORTS / "research-brief.pdf")


def main():
    build_brief(*build_report())


if __name__ == "__main__":
    main()
