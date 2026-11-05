"""Write reports/federated-study/findings.md from the saved study records."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/federated-study"

PAGE = """# Federated study

Twelve Flower runs on twelve sensors, fourteen days of readings and seeds 7, 42 and 2026. Three clients hold four sensors each and no sensor is on two clients. The model is a shared univariate GRU with 3,756 parameters, each round is one local epoch, and all three clients take part every round. Flower 1.42.0's `ClientApp`, `ServerApp` and `FedAvg` run through `flwr run` on Ray, all on one machine.

## Results

Mean MAE and RMSE in mph over the three seeds, with the seed SD in parentheses. Pooled is one model trained on all clients' data together; local-only trains each client by itself with the same amount of local work and no communication. The checkpoint with the best validation MAE is the one scored.

| Client partition | Rounds | FedAvg MAE | FedAvg RMSE | Pooled MAE | Local-only MAE | Parameter bytes |
|---|---:|---:|---:|---:|---:|---:|
{results}

Going from one to five rounds cuts balanced MAE by {drop:.4f} mph and multiplies the bytes sent by {ratio:g}. The last two rounds are worth {late:.4f} mph. At five rounds FedAvg is {above_pooled:.4f} mph above the pooled model.

Congestion-grouped clients have {cong_mae:+.4f} mph MAE and {cong_rmse:+.4f} mph RMSE compared with balanced ones at five rounds, so the change in the training distribution moves the two measures in different directions.

## FedAvg vs training alone

Local-only starts every client from the shared initial weights and repeats the work it would do in federated training (one epoch per round, same seeds, fresh optimizer each round). Each client keeps the round with its own best validation MAE, and the test errors are pooled across clients so the numbers compare with the others.

{local}

The gaps are about the size of the seed spread, so this doesn't show that communication beats training alone for this small model and 14-day slice. Pooled is lowest in every row.

![Communication and convergence](tradeoffs.png)

## How sensors were grouped

Both rules use the first twelve sensors in file order and the first 4,032 timestamps, with the same 70/10/20 split for every client and case. Sensors are ranked by the fraction of their observed training speeds below 40 mph (training rows only). Balanced groups interleave the ranks across the three clients; congestion groups take consecutive blocks of ranks.

Training speeds are counted in 10 mph bins (0, 10, ... 80, 80+), ignoring missing readings. The mean Jensen-Shannon divergence between each client's histogram and the pooled one is {js_balanced:.6f} bits for balanced ownership and {js_congestion:.6f} for congestion grouping. It only depends on training data, so it is the same for every seed.

| Rule | Client | Sensor IDs | Usable train windows | Share of training speeds below 40 mph |
|---|---:|---|---:|---:|
{clients}

An example is one sensor's input window plus its forecast targets. Only windows with at least one observed target count as training examples (for local training and for the FedAvg weights); validation and test keep every window and mask the missing targets.

## Checks

The pooled model is bit-for-bit the same for balanced and congestion ownership with every seed, and the global model after one or three rounds matches the first rounds of the five-round run. [summary.json](summary.json) has all the checks.

All twelve cases finished in {wall:.1f} seconds ({minutes:.1f} minutes) on the Ryzen 7 5825U, including launch, simulation, test scoring, the pooled and local-only models and plots. Cases ran one after another with one client actor, two reserved Ray CPUs, one PyTorch thread and a 128 MiB object store.

## Communication

The model is 15,024 bytes of float32 values. A round sends three training downloads, three uploads and three validation downloads, which is 135,216 bytes. Testing the chosen model adds 45,072. So the total is (3 x rounds + 1) x 3 x 15,024: 180,288, 450,720 and 721,152 bytes for 1, 3 and 5 rounds. This leaves out array headers, messages, config, framing, latency and energy.

## Per-seed results

| Case | Selected round | FedAvg MAE | FedAvg RMSE | Pooled MAE | Local-only MAE | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
{per_seed}

`b3-s42` is balanced groups, three rounds, seed 42; `c5-s7` is congestion groups, five rounds, seed 7. Each [case folder](cases) has the config, hashes, client info, every round's metrics, the test scores, the pooled history and the Flower log. Raw partitions, checkpoints and Flower state stay out of git.

```powershell
.\\.venv\\Scripts\\python.exe -m tracast.federated_study --config configs/federated-study.json --out runs/federated-study-new
```

[Config](config.json) · [All cases](cases.json) · [Summary](summary.json) · [Technical report](../research-report.md)
"""


def cell(item):
    return f"{item['mean']:.4f} ({item['sd']:.4f})"


def side(gap):
    return "below" if gap < 0 else "above"


def main():
    summary = json.loads((STUDY / "summary.json").read_text())
    cases = json.loads((STUDY / "cases.json").read_text())
    c = summary["conditions"]
    results = "\n".join(
        f"| {v['partition_rule']} | {v['round_budget']} | {cell(v['fedavg_mae'])} "
        f"| {cell(v['fedavg_rmse'])} | {cell(v['centralized_mae'])} "
        f"| {cell(v['local_only_mae'])} | {v['parameter_bytes']:,} |"
        for v in c.values()
    )

    def gap(key):
        return c[key]["fedavg_mae"]["mean"] - c[key]["local_only_mae"]["mean"]

    b1, b3, b5, g5 = (
        gap(k) for k in ("balanced-1", "balanced-3", "balanced-5", "congestion-5")
    )
    local = (
        f"With balanced clients, federated averaging is {abs(b3):.4f} mph {side(b3)} "
        f"local-only at three rounds, {abs(b5):.4f} {side(b5)} at five, and "
        f"{abs(b1):.4f} {side(b1)} at one. With congestion-grouped clients at five "
        f"rounds it is {abs(g5):.4f} {side(g5)}."
    )
    clients = []
    for rule, case_id in (("balanced", "b5-s42"), ("congestion", "c5-s42")):
        case = next(x for x in cases if x["case_id"] == case_id)
        for client in case["clients"]:
            clients.append(
                f"| {rule} | {client['client_id']} | {', '.join(client['sensor_ids'])} "
                f"| {client['splits']['train']['examples']:,} "
                f"| {client['training_distribution']['congestion_fraction_below_40']:.4f} |"
            )
    per_seed = "\n".join(
        f"| {x['case_id']} | {x['fedavg']['best_round']} "
        f"| {x['fedavg']['test']['mae']:.6f} | {x['fedavg']['test']['rmse']:.6f} "
        f"| {x['centralized']['test']['mae']:.6f} "
        f"| {x['local_only']['test']['mae']:.6f} | {x['wall_seconds']:.1f} |"
        for x in cases
    )
    wall = sum(x["wall_seconds"] for x in cases)
    page = PAGE.format(
        results=results,
        drop=c["balanced-1"]["fedavg_mae"]["mean"]
        - c["balanced-5"]["fedavg_mae"]["mean"],
        ratio=c["balanced-5"]["parameter_bytes"] / c["balanced-1"]["parameter_bytes"],
        late=c["balanced-3"]["fedavg_mae"]["mean"]
        - c["balanced-5"]["fedavg_mae"]["mean"],
        above_pooled=c["balanced-5"]["fedavg_mae"]["mean"]
        - c["balanced-5"]["centralized_mae"]["mean"],
        cong_mae=c["congestion-5"]["fedavg_mae"]["mean"]
        - c["balanced-5"]["fedavg_mae"]["mean"],
        cong_rmse=c["congestion-5"]["fedavg_rmse"]["mean"]
        - c["balanced-5"]["fedavg_rmse"]["mean"],
        local=local,
        js_balanced=c["balanced-5"]["mean_training_js_bits"],
        js_congestion=c["congestion-5"]["mean_training_js_bits"],
        clients="\n".join(clients),
        wall=wall,
        minutes=wall / 60,
        per_seed=per_seed,
    )
    (STUDY / "findings.md").write_bytes(page.encode())


if __name__ == "__main__":
    main()
