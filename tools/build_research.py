"""Build the research report and two-page brief from committed experiment records."""

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


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def value(item, places=3):
    suffix = f" ({item['sd']:.{places}f})" if item["sd"] is not None else ""
    return f"{item['mean']:.{places}f}" + suffix


def main():
    summary = load(REPORTS / "benchmark/summary.json")
    models = summary["models"]
    results = load(REPORTS / "benchmark/results.json")
    protocol = load(REPORTS / "benchmark/protocol.json")
    manifest = load(REPORTS / "benchmark/manifest.json")
    table = "\n".join(
        f"| {LABELS[n]} | {value(models[n]['mae'])} | {value(models[n]['rmse'])} | {models[n]['parameter_count']:,} | {value(models[n]['train_seconds'],2)} | {value(models[n]['inference_ms_per_window'],4)} |"
        for n in ORDER
    )
    graph_delta = (
        models["graph_gru"]["mae"]["mean"] - models["graph_self"]["mae"]["mean"]
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
    study_path = REPORTS / "federated-study/summary.json"
    study = load(study_path)
    rows = "\n".join(
        f"| {c['partition_rule']} | {c['round_budget']} | {value(c['fedavg_mae'])} | {value(c['fedavg_rmse'])} | {value(c['centralized_mae'])} | {c['parameter_bytes']:,} |"
        for c in study["conditions"].values()
    )
    fl_text = f"""Twelve experiments use twelve sensors, fourteen days, three clients, and seeds 7/42/2026. Each round trains one local epoch. Sensors rank by their observed training fraction below 40 mph: balanced groups interleave ranks; congestion groups take consecutive blocks. All clients share chronological boundaries. The pooled reference retains canonical sensor order across partition rules.

| Partition | Rounds | FedAvg MAE (SD) | FedAvg RMSE (SD) | Pooled MAE (SD) | Parameter bytes |
|---|---:|---:|---:|---:|---:|
{rows}

Balanced-partition MAE falls by 0.2729 mph from one to five rounds, with a fourfold payload increase. Congestion grouping raises mean MAE by 0.0616 mph at five rounds while lowering mean RMSE by 0.0072 mph. Mean training-histogram Jensen-Shannon divergence is 0.0169 bits for balanced groups and 0.0307 for congestion groups. Pooled outputs match across ownership rules; global model prefixes match across round budgets for every seed.

![Communication and convergence](federated-study/tradeoffs.png)

[Partition assignments, client counts, per-seed results, and communication accounting](federated-study/findings.md)."""
    report = f"""# TraCast: recurrent encoders, road-distance graph features, and federated traffic forecasting

Technical report - October 2026

## Abstract

TraCast evaluates traffic-speed forecasting on METR-LA with chronological evaluation, training-only normalization, and masked targets. A fixed 16-sensor subset is selected from the published DCRNN road-distance graph. GRU, LSTM, and Graph-GRU models receive the same training windows and five-epoch budgets across seeds 7, 42, and 2026. Mean test MAE is {models['gru']['mae']['mean']:.3f} mph for GRU, {models['lstm']['mae']['mean']:.3f} for LSTM, and {models['graph_gru']['mae']['mean']:.3f} for Graph-GRU, compared with {models['persistence']['mae']['mean']:.3f} for persistence. Removing graph neighbors changes mean MAE by {abs(graph_delta):.4f} mph, with mixed seed-level effects. A separate Flower study evaluates shared univariate forecasting across disjoint sensor partitions and records accuracy, convergence, sample counts, and parameter communication.

## 1. Introduction

Traffic forecasts depend on both temporal patterns and relationships between sensors. This project asks two questions: how recurrent encoder choice and explicit road-distance context affect forecasting under a fixed training budget, and how local training plus model averaging changes accuracy and communication relative to pooled training. The centralized graph benchmark and federated experiment are separate tasks with their own matched references.

## 2. Related work

[Li et al. (2018)](https://arxiv.org/abs/1707.01926) combine directed diffusion with recurrent forecasting and distribute METR-LA sensor data and distance relationships. TraCast uses those relationships in a one-hop spatial aggregation layer before the recurrent encoder. [Cho et al. (2014)](https://arxiv.org/abs/1406.1078) introduce the gated recurrent encoder used as the GRU reference. [Hochreiter and Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735) introduce LSTM. [McMahan et al. (2017)](https://arxiv.org/abs/1602.05629) describe federated averaging. [Flower](https://arxiv.org/abs/2007.14390) supplies the client/server runtime used here.

## 3. Data and preprocessing

The checksum-verified HDF5 contains 34,272 five-minute timestamps and 207 sensors, from March 1 through June 27, 2012: 7,094,304 speed entries. The benchmark selects 16 sensors, giving 548,352 speed entries before missing-value masking. Raw timestamps are retained without assigning an undocumented timezone. Numeric columns, unique sensor IDs, increasing timestamps, five-minute spacing, finite nonnegative observations, and file hashes are checked before selection.

The earliest 70% of rows are used for training, the next 10% for validation, and the final 20% for testing. Sequences are constructed after this split. Training spans March 1 00:00 to May 23 07:05, validation May 23 07:10 to June 4 04:40, and test June 4 04:45 to June 27 23:55. These contain 23,967, 3,404, and 6,832 windows respectively. Every window stays within its split. Twelve input time steps predict the next twelve steps, covering 5-60 minute forecast horizons.

Zero and NaN readings are missing. Inputs forward-fill within their split and use the sensor's training mean for leading gaps. A mean and population standard deviation are fitted to each sensor's observed training readings; constant sensors use a scale of one. Missing targets contribute to neither loss nor scores. The same windows, statistics, masks, and batch ordering are used across model architectures for a given seed. The zero-as-missing convention can exclude genuine stopped traffic.

## 4. Distance graph and architecture

The author-distributed distance table is pinned to revision `{protocol['graph']['revision']}` and checked by SHA-256. Endpoints outside the canonical 207 METR-LA sensors are excluded. The finite costs between these aligned sensors determine the population standard deviation s used by the kernel. For a supplied directed cost D[i,j], W[i,j] = exp(-(D[i,j]/s)^2), retaining values at least 0.1 and removing self pairs. No speed correlation, validation score, or test target enters graph construction.

A breadth-first traversal from the first canonical sensor selects a connected 16-sensor subset using the union of edge directions, with stronger weights first and ID tie-breaking. Its induced graph contains {protocol['graph']['subset_directed_edges']} directed relationships. Full IDs, coordinates, weights, source hashes, and the kernel scale are in [protocol.json](benchmark/protocol.json). The figure joins supplied sensor coordinates to show directed distance relationships.

Graph-GRU computes separate row-normalized incoming and outgoing means. At each sensor and time step, a shared Linear(3,4) plus ReLU transforms its own normalized speed and two neighbor means into four features. The 16 node embeddings flatten to 64 features, enter a one-layer GRU with hidden width 32, and feed a linear head producing 12 x 16 forecasts. The self-only ablation sets both neighbor operators to zero while retaining the same learned projection, recurrent encoder, head, initialization rule, and parameter count.

The GRU reference has 16 input channels, 32 hidden units, and the same direct horizon head. LSTM replaces the recurrent encoder with a 32-unit LSTM. Parameter counts are reported alongside the equal-epoch comparison.

![Selected topology](benchmark/topology.png)

## 5. Experimental setup

All learned models train for five epochs using batch size 64, Adam at 0.001, masked MAE in normalized units, and gradient clipping at 5. Every model processes the same training-window budget. The checkpoint with lowest validation MAE is selected separately per model and seed. Test scoring occurs after selection.

Trials run sequentially in fresh Python processes on an AMD Ryzen 7 5825U (8 cores, 16 logical processors), with four PyTorch threads. Windows reports 14,861,852,672 bytes of physical memory. Python 3.11.9, CPU PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3, and the remaining runtime packages are pinned. MKLDNN is disabled and deterministic PyTorch algorithms are enabled.

Training time sums update loops, including batch retrieval and excluding validation. Inference timing uses five warmups followed by thirty forward passes on the same first test batch and reports the median per window. It excludes inverse scaling, loading, and metric computation. Full test-evaluation time is logged separately. Windows PeakWorkingSetSize measures the whole fresh process, including HDF5/CSV loading and training. Timings include normal workstation background activity.

## 6. Centralized results

Values are mean (sample SD) across three training seeds. Persistence is deterministic and evaluated once. Seed variation uses one shared test block and does not quantify uncertainty across future dates or networks.

| Model | MAE, mph | RMSE, mph | Parameters | Train updates, s | Forward ms/window |
|---|---:|---:|---:|---:|---:|
{table}

![Forecast horizon errors](benchmark/horizons.png)

![Validation and training convergence](benchmark/convergence.png)

### Graph ablation

Graph-GRU minus self-only MAE differences by seed are {', '.join(f'{d:+.6f}' for d in ablation)} mph. Their mean is {graph_delta:+.6f} mph. Graph-GRU improves over the raw-input GRU in all three runs, but the edge contribution is small and changes sign. The self-only projection retains most of the mean MAE improvement over the smaller GRU reference. LSTM has nearly the same mean MAE as Graph-GRU, with fewer parameters and lower seed dispersion in this configuration.

Whole-process peak memory was approximately 320 MiB for all architectures, dominated by the runtime and preprocessing.

## 7. Federated learning

{fl_text}

Clients are logical sensor partitions on one machine. The shared GRU has one sensor per example so input/output weights retain the same meaning on every client. Sensor normalization uses local observed training readings only. The reference pools those same normalized examples, matches the initial model and exposure, and uses canonical sensor ordering. Model averaging is weighted by usable training-window counts. Evaluation pools absolute errors, squared errors, and valid target counts before calculating MAE/RMSE. The shared-host simulation uses standard FedAvg with logical client ownership.

## 8. Discussion and limits

The learned centralized encoders improve over persistence on this sensor subset and test period. The experiments give little evidence that these one-hop distance features outperform the self-only projection under a five-epoch budget. Only three initialization seeds are used, with no hyperparameter search, temporal resampling, or alternative graph-selection study. The topology-only BFS subset is connected but not representative of every road in METR-LA. Excluding neighbors outside the subset removes spatial context.

The fixed train/validation/test period, short local federated budgets, simulated ownership, and differing model capacities limit broader conclusions. The federated model is univariate; the centralized graph benchmark is multivariate. Their absolute errors should be compared within each experiment, not across the two tables. Peak memory is a process high-water mark. Communication estimates include tensor values only, excluding serialization, metrics, configuration, transport framing, latency, and energy. Raw data provenance and the distribution's license declarations are recorded in [DATA.md](../DATA.md).

## 9. Reproducibility

[README commands](../README.md) fetch hash-checked inputs, run tests, execute the model benchmark, and launch Flower. Experiment records retain resolved configuration, seeds, source/data/dependency hashes, sensor IDs, adjacency, split ranges, scaler values, per-horizon counts, histories, timing definitions, and checkpoint hashes. The benchmark source commit is `{manifest['git_commit']}`; source content hashes identify its executed working tree. Timings and process IDs are not treated as deterministic numerical outputs.

The benchmark and Flower runs use separate output directories and never overwrite completed experiments. Raw HDF5 files, topology CSVs, checkpoints, generated client partitions, and local Flower state are excluded from Git. Unit tests cover schema validation, chronological boundaries, train-only scaling, masking, horizon pooling, graph direction/alignment, changed-edge predictions, LSTM gradients, actual Flower aggregation and message dispatch, and failed launch states. The experiment CLIs provide the real-data integration runs.

## References

1. Li, Y., Yu, R., Shahabi, C., Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR. https://arxiv.org/abs/1707.01926
2. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP. https://arxiv.org/abs/1406.1078
3. Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. McMahan, H. B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS. https://arxiv.org/abs/1602.05629
5. Beutel, D. J. et al. (2020). Flower: A Friendly Federated Learning Research Framework. https://arxiv.org/abs/2007.14390
6. Flower 1.42.0 ClientApp, ServerApp and FedAvg documentation. https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
"""
    (REPORTS / "research-report.md").write_bytes(report.encode())
    (REPORTS / "benchmark/report.md").write_text(
        "# Centralized model benchmark\n\n"
        "The benchmark uses sixteen sensors selected from the METR-LA distance graph, the full March–June 2012 timeline, "
        "seeds 7/42/2026, and five epochs per learned model. Each trial runs in a fresh CPU process.\n\n"
        "The [technical report](../research-report.md#3-data-and-preprocessing) describes preprocessing, "
        "graph construction, architecture, training, and interpretation.\n\n"
        "| Record | Contents |\n|---|---|\n"
        "| [Summary](summary.json) | Mean and sample SD, horizon metrics, paired differences |\n"
        "| [Trials](results.json) | Per-seed scores, convergence, timings, model hashes |\n"
        "| [Protocol](protocol.json) | Sensor IDs, splits, scalers, graph weights, missing counts |\n"
        "| [Manifest](manifest.json) | Executed source, dependencies, and input hashes |\n"
        "| [Configuration](config.json) | Resolved model and training settings |\n"
        "| [Repeatability](repeatability.json) | Exact seed-42 Graph-GRU and LSTM repeats |\n\n"
        "![Topology](topology.png)\n\n"
        "![Validation convergence](../figures/validation-convergence.png)\n\n"
        "[Run commands](../../docs/reproducibility.md#centralized-models)\n",
        encoding="utf-8",
        newline="\n",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=6
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
        return Paragraph(text, styles["BodySmall"])

    def heading(text):
        return Paragraph(text, styles["SectionSmall"])

    story = [
        Paragraph("TraCast", styles["Title"]),
        body(
            "<b>Traffic-speed forecasting with recurrent encoders, road-distance graphs, and Flower</b>"
        ),
        body(
            "ivanimmanuel-dev / TraCast &nbsp; | &nbsp; Research brief - October 2026"
        ),
        heading("Research question"),
        body(
            "How much do recurrent encoder choice, published sensor relationships, and federated model averaging change traffic forecasts under controlled training budgets?"
        ),
        heading("Centralized benchmark"),
    ]
    rows = [["Model", "MAE (SD)", "RMSE (SD)", "Parameters"]] + [
        [
            LABELS[n],
            value(models[n]["mae"]),
            value(models[n]["rmse"]),
            f"{models[n]['parameter_count']:,}",
        ]
        for n in ORDER
    ]
    t = Table(rows, colWidths=[156, 113, 113, 96], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f1f4")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#176b87")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story += [
        t,
        Paragraph(
            "Errors in mph; three training seeds on one fixed test period. Persistence is deterministic and evaluated once.",
            styles["CaptionSmall"],
        ),
        body(
            f"LSTM and Graph-GRU reached nearly equal mean MAE. Graph-GRU improved over GRU by {models['gru']['mae']['mean']-models['graph_gru']['mae']['mean']:.4f} mph, while removing graph neighbors changed mean MAE by only {abs(graph_delta):.4f} mph. The edge effect changed sign across seeds."
        ),
        heading("Data and evaluation"),
        body(
            "METR-LA contains 34,272 five-minute timestamps and 207 sensors. The benchmark uses 16 topology-selected sensors across the full available date range, March 1-June 27, 2012: 548,352 speed entries before masking."
        ),
        body(
            "The pipeline splits rows chronologically into 70% training, 10% validation, and 20% test data before making sequences. Twelve input steps predict the next twelve steps. Means and scales use observed training values only. Inputs forward-fill within their split; zero/NaN targets remain masked. Validation MAE selects checkpoints."
        ),
        heading("Graph-GRU architecture"),
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
            "The graph comes from the DCRNN author's distance table. Its Gaussian kernel, sensor IDs, and file hashes are recorded. The selected graph has 85 directed non-self relationships; no speed correlation or held-out targets determine its edges."
        ),
        PageBreak(),
        heading("Forecast horizon errors"),
        Image(str(REPORTS / "figures/forecast-horizons.png"), width=478, height=182),
        heading("Federated forecasting"),
    ]
    b = study["conditions"]["balanced-5"]
    g = study["conditions"]["congestion-5"]
    story.append(
        body(
            f"The federated study uses twelve sensors, fourteen days, three clients, and three seeds. At five rounds, "
            f"balanced partitions reached MAE {b['fedavg_mae']['mean']:.3f} mph against "
            f"{b['centralized_mae']['mean']:.3f} for pooled training. Congestion-grouped clients "
            f"reached {g['fedavg_mae']['mean']:.3f} mph. Increasing the balanced budget from "
            "one to five rounds reduced MAE by 0.273 mph at four times the model payload "
            f"({b['parameter_bytes']:,} bytes)."
        )
    )
    story += [
        body(
            "The federated task shares a univariate GRU across sensors. Local training-only scalers, masks, time boundaries, and initial weights match its pooled reference. Model updates use sample-count weighting; errors pool by valid target count. Client partitions run on one host using standard FedAvg."
        ),
        heading("Execution and reproducibility"),
        body(
            f"Each learned model trains for five epochs with seeds 7/42/2026, batch size 64, and hidden width 32. Training updates averaged {models['gru']['train_seconds']['mean']:.1f} s for GRU, {models['lstm']['train_seconds']['mean']:.1f} s for LSTM, and {models['graph_gru']['train_seconds']['mean']:.1f} s for Graph-GRU. Whole-process peak memory was about 320 MiB on a Ryzen 7 5825U with four PyTorch threads."
        ),
        body(
            "Configs, source/data hashes, normalization statistics, convergence logs, horizon metrics, timings, model hashes, and regression tests accompany the reports. Commands fetch checksum-verified inputs and use pinned Python/PyTorch/Flower dependencies."
        ),
        heading("Interpretation"),
        body(
            "The encoder comparison favors LSTM's smaller parameter count at nearly the same mean error as Graph-GRU. The graph ablation shows a small, mixed seed-level edge contribution. Three seeds on one test block describe initialization variation, not uncertainty across future traffic periods. Communication estimates cover tensor values; runtime overhead and real-network costs are separate."
        ),
        heading("Code and evidence"),
        body(
            '<link href="https://github.com/ivanimmanuel-dev/TraCast">github.com/ivanimmanuel-dev/TraCast</link><br/>The full methods, metric tables, configurations, and figures are in reports/research-report.md. DATA.md documents the dataset and graph sources.'
        ),
        Paragraph(
            "References: Li et al., ICLR 2018 (METR-LA/DCRNN); Cho et al., EMNLP 2014 (GRU); Hochreiter and Schmidhuber, Neural Computation 1997 (LSTM); McMahan et al., AISTATS 2017 (FedAvg); Flower 1.42.0.",
            styles["CaptionSmall"],
        ),
    ]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#536471"))
        canvas.drawString(36, 24, "TraCast | Research brief | October 2026")
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


if __name__ == "__main__":
    main()
