"""Arms x seeds for the federated graph experiments, one fresh process per case.

A settings file lists a base configuration, the seeds and the arms. An arm is a
federated run (through Flower or the in-process loop), a pooled centralized run or
local-only training, each with its own overrides. Cases are independent folders, so
a run can be stopped, resumed, or split over several processes by arms and seeds.
`--summarize-only` then builds one table from whatever finished.
"""

import argparse
from dataclasses import asdict, fields, replace
from importlib.metadata import version
from pathlib import Path
import json
import platform
import subprocess
import sys
import time

import numpy as np

from . import fedgraph as F
from .data import SHA256
from .experiment import code_manifest, write_json
from .provenance import git_state
from .resources import peak_memory
from .stats import day_bootstrap

MODES = ("federated", "central", "local")
RUNNERS = ("flower", "inprocess")
FIELDS = {f.name for f in fields(F.GraphFedConfig)}
RESERVED = {"name", "mode", "runner", "edges", "seeds"}


def load_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_settings(settings)
    return settings


def validate_settings(settings):
    base = F.GraphFedConfig(**settings["base"])
    base.validate()
    seeds = settings["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct integers")
    names = [a["name"] for a in settings["arms"]]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Arm names must be unique")
    for arm in settings["arms"]:
        if arm["mode"] not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if arm.get("runner", "inprocess") not in RUNNERS:
            raise ValueError(f"runner must be one of {RUNNERS}")
        if arm.get("runner") == "flower" and arm["mode"] != "federated":
            raise ValueError("Only federated arms run through Flower")
        if arm.get("edges", "full") not in ("full", "local"):
            raise ValueError("edges must be full or local")
        unknown = set(arm) - FIELDS - RESERVED
        if unknown:
            raise ValueError(f"Unknown arm keys: {sorted(unknown)}")
        for seed in arm_seeds(settings, arm):
            arm_config(settings, arm, seed).validate()
    tuned = {
        f"{a['name']}+ft"
        for a in settings["arms"]
        if arm_config(settings, a, arm_seeds(settings, a)[0]).personalize_rounds
    }
    for pair in settings.get("pairs", []):
        if len(pair) != 2 or not set(pair) <= set(names) | tuned | {
            "persistence",
            "ha",
        }:
            raise ValueError("Each pair names two arms")
        found = [
            find_arm(settings, n.removesuffix("+ft"))
            for n in pair
            if n.removesuffix("+ft") in names
        ]
        if len({tuple(arm_seeds(settings, a)) for a in found}) > 1:
            raise ValueError("Paired arms must use the same seeds")


def arm_seeds(settings, arm):
    """An arm normally runs every seed; a cheap sweep arm can name fewer."""
    seeds = arm.get("seeds", settings["seeds"])
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("arm seeds must be distinct integers")
    return seeds


def arm_config(settings, arm, seed):
    changes = {k: v for k, v in arm.items() if k in FIELDS}
    return replace(F.GraphFedConfig(**settings["base"]), seed=seed, **changes)


def find_arm(settings, name):
    for arm in settings["arms"]:
        if arm["name"] == name:
            return arm
    raise KeyError(name)


def environment(config):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("torch", "numpy", "pandas", "flwr", "ray", "tables")
        },
        "threads": config.threads,
        "package_path": str(Path(__file__).resolve().parents[1]),
        "data_sha256": SHA256,
        **git_state(),
        "code_sha256": code_manifest(),
        "deterministic": {"torch_algorithms": True, "mkldnn": False},
    }


# --- one case -------------------------------------------------------------------------


def run_case(settings, name, seed, data, graph_dir, out):
    """Run one arm for one seed and write its folder; returns the result."""
    arm = find_arm(settings, name)
    config = arm_config(settings, arm, seed)
    folder = Path(out) / f"{name}-s{seed}"
    done = folder / "result.json"
    if done.exists():
        return json.loads(done.read_text())
    if folder.exists():
        folder.rename(folder.with_name(f"{folder.name}.unfinished-{int(time.time())}"))
    folder.mkdir(parents=True)
    write_json(folder / "status.json", {"status": "running"})
    prep = F.prepare(config, data, graph_dir)
    clients = F.build_clients(prep, config)
    start = F.initial_parameters(config)
    write_json(folder / "setup.json", F.describe(prep, clients, config))
    write_json(folder / "manifest.json", environment(config) | {"arm": arm})
    log = lambda text: print(f"[{name} s{seed}] {text}", flush=True)  # noqa: E731
    mode = arm["mode"]
    if mode == "central":
        pooled = F.pooled_client(prep, config, arm.get("edges", "full"))
        result = F.run_pooled(config, pooled, start, name, prep["groups"], log)
        result["peak_memory"] = peak_memory()
    elif mode == "local":
        result = F.run_local_only(config, clients, start, log)
        result["peak_memory"] = peak_memory()
    elif arm.get("runner", "inprocess") == "flower":
        from . import fedgraph_flower as FF

        job = folder / "job"
        job.mkdir()
        FF.write_job(prep, config, job, start)
        result = FF.execute(job)
        # the client partitions are big and the hashes stay in setup.json
        for path in (job / "clients").glob("*.npz"):
            path.unlink()
        result["peak_memory"] = {
            "bytes": result.pop("peak_client_process_bytes"),
            "method": "largest Ray client process (whole process, all rounds)",
        }
    else:
        result = F.run_federated(config, clients, start, log=log)
        result["peak_memory"] = peak_memory()
    result["arm"], result["seed"], result["config"] = name, seed, asdict(config)
    write_json(done, result)
    write_json(folder / "status.json", {"status": "complete"})
    return result


def run_baselines(settings, data, graph_dir, out):
    """Persistence and the historical average: nothing trains and no seed matters."""
    folder = Path(out) / "baselines"
    done = folder / "result.json"
    if done.exists():
        return json.loads(done.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    config = F.GraphFedConfig(**settings["base"])
    prep = F.prepare(config, data, graph_dir)
    clients = F.build_clients(prep, config)
    result = {
        "persistence": F.persistence_arm(clients, config),
        "ha": F.historical_average_arm(prep, clients, config),
        "config": asdict(config),
    }
    write_json(done, result)
    return result


# --- orchestration --------------------------------------------------------------------


def run(settings_path, data, graph_dir, out, arms=None, seeds=None):
    settings = load_settings(settings_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "settings.json", settings | git_state())
    selected = arms or [a["name"] for a in settings["arms"]]
    for name in selected:
        find_arm(settings, name)
    run_baselines(settings, data, graph_dir, out)
    for seed in seeds or settings["seeds"]:
        for name in selected:
            if seed not in arm_seeds(settings, find_arm(settings, name)):
                continue
            if (out / f"{name}-s{seed}" / "result.json").exists():
                print(f"{name} seed {seed}: already done", flush=True)
                continue
            began = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tracast.fedgraph_study",
                    "--config",
                    str(settings_path),
                    "--data",
                    str(data),
                    "--topology",
                    str(graph_dir),
                    "--out",
                    str(out),
                    "--case",
                    name,
                    "--seed",
                    str(seed),
                ],
                check=True,
            )
            print(
                f"{name} seed {seed}: done in {time.perf_counter() - began:.0f} s",
                flush=True,
            )
    return summarize(settings_path, out)


# --- summary --------------------------------------------------------------------------


def _stats(values):
    values = [float(v) for v in values]
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def load_cases(settings, out):
    cases = {}
    for arm in settings["arms"]:
        runs = []
        for seed in arm_seeds(settings, arm):
            path = Path(out) / f"{arm['name']}-s{seed}" / "result.json"
            if path.exists():
                runs.append(json.loads(path.read_text()))
        if runs:
            cases[arm["name"]] = runs
    return cases


def tuned_entry(base, runs):
    """The summary row for a federated arm after local fine-tuning (name + '+ft')."""
    tuned = [r["personalized"] for r in runs]
    first = tuned[0]
    return {
        "mode": "federated+finetune",
        "seeds": [r["seed"] for r in runs],
        "mae": _stats([t["test"]["mae"] for t in tuned]),
        "rmse": _stats([t["test"]["rmse"] for t in tuned]),
        "horizon_mae": {
            str(h["minutes"]): _stats(
                [t["test"]["per_horizon"][i]["mae"] for t in tuned]
            )
            for i, h in enumerate(first["test"]["per_horizon"])
        },
        "horizon_rmse": {
            str(h["minutes"]): _stats(
                [t["test"]["per_horizon"][i]["rmse"] for t in tuned]
            )
            for i, h in enumerate(first["test"]["per_horizon"])
        },
        "train_seconds": _stats(
            [r["train_seconds"] + r["personalized"]["train_seconds"] for r in runs]
        ),
        "wall_seconds": base["wall_seconds"],
        "peak_memory_mib": base["peak_memory_mib"],
        "peak_memory_method": base["peak_memory_method"],
        "clients": [
            {
                "client": c["client"],
                "sensors": c["sensors"],
                "mae": _stats([t["clients"][i]["metrics"]["mae"] for t in tuned]),
                "rmse": _stats([t["clients"][i]["metrics"]["rmse"] for t in tuned]),
                "fine_tune_round": _stats(
                    [t["clients"][i]["best_round"] for t in tuned]
                ),
            }
            for i, c in enumerate(first["clients"])
        ],
        "fine_tune_rounds_allowed": first["rounds"],
        "parameters": base.get("parameters"),
        "communication": base.get("communication"),
        "setup": base.get("setup"),
    }


def summarize(settings_path, out):
    settings = load_settings(settings_path)
    out = Path(out)
    cases = load_cases(settings, out)
    baselines = json.loads((out / "baselines" / "result.json").read_text())
    arms = {}
    for name, runs in cases.items():
        first = runs[0]
        entry = {
            "mode": find_arm(settings, name)["mode"],
            "seeds": [r["seed"] for r in runs],
            "mae": _stats([r["test"]["mae"] for r in runs]),
            "rmse": _stats([r["test"]["rmse"] for r in runs]),
            "horizon_mae": {
                str(h["minutes"]): _stats(
                    [r["test"]["per_horizon"][i]["mae"] for r in runs]
                )
                for i, h in enumerate(first["test"]["per_horizon"])
            },
            "horizon_rmse": {
                str(h["minutes"]): _stats(
                    [r["test"]["per_horizon"][i]["rmse"] for r in runs]
                )
                for i, h in enumerate(first["test"]["per_horizon"])
            },
            "train_seconds": _stats([r["train_seconds"] for r in runs]),
            "wall_seconds": _stats([r["wall_seconds"] for r in runs]),
            "peak_memory_mib": _stats(
                [r["peak_memory"]["bytes"] / 2**20 for r in runs]
            ),
            "peak_memory_method": first["peak_memory"]["method"],
            "clients": [
                {
                    "client": c["client"],
                    "sensors": c["sensors"],
                    "mae": _stats([r["clients"][i]["metrics"]["mae"] for r in runs]),
                    "rmse": _stats([r["clients"][i]["metrics"]["rmse"] for r in runs]),
                }
                for i, c in enumerate(first["clients"])
            ],
        }
        if "best_round" in first:
            entry["best_round"] = _stats([r["best_round"] for r in runs])
        setup_path = out / f"{name}-s{first['seed']}" / "setup.json"
        if setup_path.exists():
            setup = json.loads(setup_path.read_text())
            graph = setup["graph"]
            entry["setup"] = {
                "clients": len(setup["clients"]),
                "client_sensors": [c["sensor_count"] for c in setup["clients"]],
                "edges_kept_fraction": graph["edges_kept_fraction"],
                "weight_kept_fraction": graph["weight_kept_fraction"],
                "partition": setup["config"]["partition"],
            }
        if "parameter_count" in first:
            entry["parameters"] = first["parameter_count"]
        if "communication" in first:
            entry["communication"] = first["communication"]
        if "budgets" in first:
            entry["budgets"] = {
                b: {
                    "mae": _stats([r["budgets"][b]["mae"] for r in runs]),
                    "rmse": _stats([r["budgets"][b]["rmse"] for r in runs]),
                    "best_round": _stats([r["budgets"][b]["best_round"] for r in runs]),
                    "communication_bytes": first["budgets"][b]["communication_bytes"],
                }
                for b in first["budgets"]
            }
        if "history" in first and "validation" in first["history"][0]:
            rounds = len(first["history"])
            entry["validation_curve"] = [
                _stats([r["history"][i]["validation"]["mae"] for r in runs])
                for i in range(rounds)
            ]
        arms[name] = entry
        if "personalized" in first:
            arms[f"{name}+ft"] = tuned_entry(entry, runs)
    daily = {name: [r["daily"] for r in runs] for name, runs in cases.items()}
    for name, runs in cases.items():
        if "personalized" in runs[0]:
            daily[f"{name}+ft"] = [r["personalized"]["daily"] for r in runs]
    daily["persistence"] = [baselines["persistence"]["daily"]]
    daily["ha"] = [baselines["ha"]["daily"]]
    paired = {}
    for a, b in settings.get("pairs", []):
        if a in daily and b in daily:
            paired[f"{a}_vs_{b}"] = day_bootstrap(daily[a], daily[b])
    summary = {
        "name": settings.get("name", "federated graph study"),
        "base": settings["base"],
        "seeds": settings["seeds"],
        "arms": arms,
        "baselines": {
            key: {
                "mae": baselines[key]["test"]["mae"],
                "rmse": baselines[key]["test"]["rmse"],
                "horizon_mae": {
                    str(h["minutes"]): h["mae"]
                    for h in baselines[key]["test"]["per_horizon"]
                },
                "clients": [
                    {"client": c["client"], "mae": c["metrics"]["mae"]}
                    for c in baselines[key]["clients"]
                ],
            }
            for key in ("persistence", "ha")
        },
        "paired_day_bootstrap": paired,
        "paired_note": "Gaps in test MAE (first minus second, negative favors the first) on identical targets; 95% intervals from resampling test days with training seeds averaged first.",
        **git_state(),
    }
    first_setup = None
    for name in cases:
        path = out / f"{name}-s{cases[name][0]['seed']}" / "setup.json"
        if path.exists():
            first_setup = json.loads(path.read_text())
            break
    if first_setup:
        summary["partition"] = {
            "clients": [
                {
                    "client": c["client"],
                    "sensors": c["sensor_count"],
                    "local_edges": c["local_edges"],
                    "train_examples": c["train_examples"],
                }
                for c in first_setup["clients"]
            ],
            "graph": first_setup["graph"],
            "split": first_setup["split"],
        }
    write_json(out / "summary.json", summary)
    figures(summary, out)
    return summary


# --- figures --------------------------------------------------------------------------

COLORS = [
    "#1d4ed8",
    "#087e8b",
    "#9333ea",
    "#d97706",
    "#dc2626",
    "#16a34a",
    "#a16207",
    "#475569",
]


def figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    arms = summary["arms"]
    names = list(arms)
    color = {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(
        x,
        [arms[n]["mae"]["mean"] for n in names],
        yerr=[arms[n]["mae"]["sd"] or 0 for n in names],
        color=[color[n] for n in names],
        capsize=4,
    )
    for key, style in (("persistence", ":"), ("ha", "--")):
        ax.axhline(
            summary["baselines"][key]["mae"],
            color="#667085",
            linestyle=style,
            label={"persistence": "Persistence", "ha": "Historical average"}[key],
        )
    low = min(arms[n]["mae"]["mean"] for n in names)
    ax.set_ylim(low - 0.4, summary["baselines"]["persistence"]["mae"] + 0.2)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set(ylabel="Test MAE (mph)", title="Test MAE, mean and seed SD")
    ax.legend(fontsize=8)
    fig.savefig(out / "accuracy.png", dpi=150)
    plt.close(fig)

    curves = [n for n in names if "validation_curve" in arms[n]]
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for n in curves:
            mean = np.array([c["mean"] for c in arms[n]["validation_curve"]])
            sd = np.array([c["sd"] or 0 for c in arms[n]["validation_curve"]])
            rounds = np.arange(1, len(mean) + 1)
            ax.plot(rounds, mean, label=n, color=color[n])
            ax.fill_between(rounds, mean - sd, mean + sd, alpha=0.15, color=color[n])
        ax.set(
            xlabel="Round (federated) or epoch block (pooled)",
            ylabel="Validation MAE (mph)",
            title="Validation convergence, mean and seed SD",
        )
        top = np.median([arms[n]["validation_curve"][0]["mean"] for n in curves])
        ax.set_ylim(None, top * 1.15)
        ax.legend(fontsize=8)
        fig.savefig(out / "convergence.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for n in names:
        steps = sorted(arms[n]["horizon_mae"], key=int)
        ax.plot(
            [int(s) for s in steps],
            [arms[n]["horizon_mae"][s]["mean"] for s in steps],
            label=n,
            color=color[n],
        )
    for key, style in (("persistence", ":"), ("ha", "--")):
        steps = sorted(summary["baselines"][key]["horizon_mae"], key=int)
        ax.plot(
            [int(s) for s in steps],
            [summary["baselines"][key]["horizon_mae"][s] for s in steps],
            color="#667085",
            linestyle=style,
            label={"persistence": "Persistence", "ha": "Historical average"}[key],
        )
    ax.set(
        xlabel="Forecast horizon (minutes)",
        ylabel="Test MAE (mph)",
        title="Test MAE by forecast horizon",
    )
    ax.legend(fontsize=7)
    fig.savefig(out / "horizons.png", dpi=150)
    plt.close(fig)

    clients = arms[names[0]]["clients"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    width = 0.8 / len(names)
    for j, n in enumerate(names):
        ax.bar(
            np.arange(len(clients)) + j * width,
            [c["mae"]["mean"] for c in arms[n]["clients"]],
            width,
            label=n,
            color=color[n],
        )
    ax.set_xticks(
        np.arange(len(clients)) + 0.4 - width / 2,
        [f"{c['client']} ({c['sensors']})" for c in clients],
    )
    ax.set(
        xlabel="Client (number of sensors)",
        ylabel="Test MAE (mph)",
        title="Test MAE by client, mean over seeds",
    )
    ax.legend(fontsize=7)
    fig.savefig(out / "clients.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fedgraph.json")
    parser.add_argument("--data", default="data/raw/metr-la.h5")
    parser.add_argument("--topology", default="data/topology")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", help="comma-separated arm names (default: all)")
    parser.add_argument("--seeds", help="comma-separated seeds (default: all)")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--case", help="internal: run one arm for --seed and stop")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.case:
        run_case(
            load_settings(args.config),
            args.case,
            args.seed,
            args.data,
            args.topology,
            args.out,
        )
        return
    if args.summarize_only:
        summarize(args.config, args.out)
        return
    run(
        args.config,
        args.data,
        args.topology,
        args.out,
        arms=args.arms.split(",") if args.arms else None,
        seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else None,
    )


if __name__ == "__main__":
    main()
