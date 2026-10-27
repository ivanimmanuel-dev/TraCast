"""Compare Flower round budgets and training-distribution partitions."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np

from .federated import prepare, execute
from .federated_task import FederatedConfig
from .experiment import write_json


def js_divergence(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if (
        first.ndim != 1
        or first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or (first < 0).any()
        or (second < 0).any()
        or not first.sum()
        or not second.sum()
    ):
        raise ValueError("Invalid distribution counts")
    first, second = first / first.sum(), second / second.sum()
    middle = (first + second) / 2

    def divergence(p):
        mask = p > 0
        return float((p[mask] * np.log2(p[mask] / middle[mask])).sum())

    return 0.5 * (divergence(first) + divergence(second))


def heterogeneity(clients):
    hist = [
        np.asarray(c["training_distribution"]["speed_histogram"], dtype=float)
        for c in clients
    ]
    pooled = np.sum(hist, axis=0)
    values = [js_divergence(h, pooled) for h in hist]
    return {
        "mean_js_bits": float(np.mean(values)),
        "client_js_bits": values,
        "congestion_fractions": [
            c["training_distribution"]["congestion_fraction_below_40"] for c in clients
        ],
        "scope": "Observed training speed histograms; mean client-to-pooled Jensen-Shannon divergence in bits",
    }


def aggregate(cases):
    conditions = {}
    for key in dict.fromkeys(
        f"{c['partition_rule']}-{c['round_budget']}" for c in cases
    ):
        values = [
            c for c in cases if f"{c['partition_rule']}-{c['round_budget']}" == key
        ]

        def stats(items):
            return {
                "mean": float(np.mean(items)),
                "sd": float(np.std(items, ddof=1)) if len(items) > 1 else None,
            }

        conditions[key] = {
            "n": len(values),
            "seeds": [c["seed"] for c in values],
            "partition_rule": values[0]["partition_rule"],
            "round_budget": values[0]["round_budget"],
            "fedavg_mae": stats([c["fedavg"]["test"]["mae"] for c in values]),
            "fedavg_rmse": stats([c["fedavg"]["test"]["rmse"] for c in values]),
            "centralized_mae": stats([c["centralized"]["test"]["mae"] for c in values]),
            "centralized_rmse": stats(
                [c["centralized"]["test"]["rmse"] for c in values]
            ),
            "persistence_mae": stats(
                [c["centralized"]["persistence_test"]["mae"] for c in values]
            ),
            "parameter_bytes": values[0]["fedavg"]["communication"][
                "total_parameter_bytes"
            ],
            "mean_training_js_bits": values[0]["heterogeneity"]["mean_js_bits"],
            "wall_seconds": stats([c["wall_seconds"] for c in values]),
        }
    reference_checks = {}
    prefix_checks = {}
    for seed in sorted(set(c["seed"] for c in cases)):
        balanced = max(
            (
                c
                for c in cases
                if c["seed"] == seed and c["partition_rule"] == "balanced"
            ),
            key=lambda c: c["round_budget"],
        )
        grouped_cases = [
            c
            for c in cases
            if c["seed"] == seed and c["partition_rule"] == "congestion"
        ]
        for grouped in grouped_cases:
            reference = next(
                c
                for c in cases
                if c["seed"] == seed
                and c["partition_rule"] == "balanced"
                and c["round_budget"] == grouped["round_budget"]
            )
            key = (
                str(seed)
                if len(grouped_cases) == 1
                else f"{seed}-{grouped['round_budget']}"
            )
            reference_checks[key] = reference["centralized"] == grouped["centralized"]
        for short in [
            c for c in cases if c["seed"] == seed and c["partition_rule"] == "balanced"
        ]:
            prefix_checks[short["case_id"]] = [
                r["state_sha256"] for r in short["rounds"]
            ] == [
                r["state_sha256"] for r in balanced["rounds"][: short["round_budget"]]
            ]
    if not all(reference_checks.values()) or not all(prefix_checks.values()):
        raise ValueError("Reference or round-prefix reproducibility check failed")
    budget = max(
        c["round_budget"] for c in cases if c["partition_rule"] == "congestion"
    )
    b, g = conditions[f"balanced-{budget}"], conditions[f"congestion-{budget}"]
    config = cases[0]["config"]
    budgets = sorted(
        {c["round_budget"] for c in cases if c["partition_rule"] == "balanced"}
    )
    brief = (
        f"The Flower study used {config['clients'] * config['sensors_per_client']} sensors, "
        f"{config['rows'] / 288:g} days, {config['clients']} clients, and {len(b['seeds'])} seeds. "
        f"At {budget} rounds, balanced partitions scored MAE {b['fedavg_mae']['mean']:.3f} mph "
        f"versus {b['centralized_mae']['mean']:.3f} for the pooled reference. "
        f"Congestion-grouped clients scored {g['fedavg_mae']['mean']:.3f} mph. "
        f"Training histogram divergence was {b['mean_training_js_bits']:.4f} versus "
        f"{g['mean_training_js_bits']:.4f} bits. Estimated model parameter payload: "
        f"{b['parameter_bytes']:,} bytes. Balanced round budgets: {'/'.join(map(str, budgets))}."
    )
    return {
        "conditions": conditions,
        "centralized_reference_identical": reference_checks,
        "round_prefix_identical": prefix_checks,
        "brief_findings": brief,
        "uncertainty": "Sample SD across training seeds on the same held-out dates",
    }


def plot(out, cases, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    balanced = sorted(
        [
            c
            for c in summary["conditions"].values()
            if c["partition_rule"] == "balanced"
        ],
        key=lambda c: c["round_budget"],
    )
    axes[0].errorbar(
        [c["parameter_bytes"] / 1024 for c in balanced],
        [c["fedavg_mae"]["mean"] for c in balanced],
        yerr=[c["fedavg_mae"]["sd"] or 0 for c in balanced],
        fmt="o-",
        capsize=4,
        label="Balanced",
    )
    grouped = sorted(
        (
            c
            for c in summary["conditions"].values()
            if c["partition_rule"] == "congestion"
        ),
        key=lambda c: c["round_budget"],
    )
    axes[0].errorbar(
        [c["parameter_bytes"] / 1024 for c in grouped],
        [c["fedavg_mae"]["mean"] for c in grouped],
        yerr=[c["fedavg_mae"]["sd"] or 0 for c in grouped],
        fmt="s--",
        capsize=4,
        label="Congestion grouped",
    )
    for c in balanced:
        axes[0].annotate(
            f" {c['round_budget']} round" + ("s" if c["round_budget"] != 1 else ""),
            (c["parameter_bytes"] / 1024, c["fedavg_mae"]["mean"]),
            fontsize=8,
        )
    axes[0].set(
        xlabel="Estimated parameter payload (KiB)",
        ylabel="Selected-model test MAE (mph)",
        title="Accuracy and communication budget",
    )
    axes[0].legend(fontsize=8)
    budget = max(c["round_budget"] for c in grouped)
    for rule, label in [("balanced", "Balanced"), ("congestion", "Congestion grouped")]:
        selected = [
            c
            for c in cases
            if c["partition_rule"] == rule and c["round_budget"] == budget
        ]
        values = np.array(
            [[r["validation"]["mae"] for r in c["rounds"]] for c in selected]
        )
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[1].plot(range(1, budget + 1), mean, label=label)
        axes[1].fill_between(range(1, budget + 1), mean - sd, mean + sd, alpha=0.15)
    axes[1].set(
        xlabel="Communication round",
        ylabel="Validation MAE (mph)",
        title="Validation convergence, mean and seed SD",
    )
    axes[1].set_xticks(range(1, budget + 1))
    axes[1].legend(fontsize=8)
    fig.savefig(out / "tradeoffs.png", dpi=150)
    plt.close(fig)


def validate_settings(settings):
    base = FederatedConfig(**settings["base"])
    base.validate()
    seeds = settings["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("The study needs multiple distinct integer seeds")
    budgets = {}
    for rule in ("balanced", "congestion"):
        values = settings[f"{rule}_round_budgets"]
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{rule} round budgets must be distinct positive integers")
        budgets[rule] = values
        for seed in seeds:
            for budget in values:
                replace(base, seed=seed, rounds=budget, partition_rule=rule).validate()
    if not set(budgets["congestion"]) <= set(budgets["balanced"]):
        raise ValueError("Each congestion budget needs a matching balanced reference")
    return base


def run(settings, data, out):
    base = validate_settings(settings)
    seeds = settings["seeds"]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", settings)
    write_json(out / "status.json", {"status": "running"})
    cases = []
    try:
        for seed in seeds:
            for rule, budgets in [
                ("balanced", settings["balanced_round_budgets"]),
                ("congestion", settings["congestion_round_budgets"]),
            ]:
                for budget in budgets:
                    config = replace(
                        base, seed=seed, rounds=budget, partition_rule=rule
                    )
                    case_id = f"{rule[0]}{budget}-s{seed}"
                    start = time.perf_counter()
                    job = prepare(config, data, out / case_id)
                    execute(job)

                    def read(name):
                        return json.loads((job / name).read_text())

                    clients = read("clients.json")
                    case = {
                        "case_id": case_id,
                        "seed": seed,
                        "partition_rule": rule,
                        "round_budget": budget,
                        "config": asdict(config),
                        "fedavg": read("federated.json"),
                        "centralized": read("centralized.json"),
                        "clients": clients,
                        "rounds": read("rounds.json"),
                        "centralized_history": read("centralized-history.json"),
                        "heterogeneity": heterogeneity(clients),
                        "wall_seconds": time.perf_counter() - start,
                    }
                    cases.append(case)
                    write_json(out / "cases.json", cases)
                    print(
                        f"{case_id}: FedAvg MAE={case['fedavg']['test']['mae']:.6f}; central={case['centralized']['test']['mae']:.6f}; wall={case['wall_seconds']:.1f}s",
                        flush=True,
                    )
        summary = aggregate(cases)
        write_json(out / "summary.json", summary)
        plot(out, cases, summary)
        write_json(out / "status.json", {"status": "complete"})
        return summary
    except BaseException as error:
        write_json(
            out / "status.json",
            {"status": "failed", "type": type(error).__name__, "error": str(error)},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/federated-study.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.data, args.out)


if __name__ == "__main__":
    main()
