"""Round-budget and training-distribution study using the existing Flower apps."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import json
import time

import numpy as np

from .federated import prepare,execute
from .federated_task import FederatedConfig
from .experiment import write_json


def js_divergence(first,second):
    first=np.asarray(first,dtype=float)
    second=np.asarray(second,dtype=float)
    if (first<0).any() or (second<0).any() or not first.sum() or not second.sum():
        raise ValueError("Invalid distribution counts")
    first,second=first/first.sum(),second/second.sum()
    middle=(first+second)/2
    def divergence(p):
        mask=p>0
        return float((p[mask]*np.log2(p[mask]/middle[mask])).sum())
    return .5*(divergence(first)+divergence(second))


def heterogeneity(clients):
    hist=[np.asarray(c["training_distribution"]["speed_histogram"],dtype=float) for c in clients]
    pooled=np.sum(hist,axis=0)
    values=[js_divergence(h,pooled) for h in hist]
    return {"mean_js_bits":float(np.mean(values)),"client_js_bits":values,
        "congestion_fractions":[c["training_distribution"]["congestion_fraction_below_40"] for c in clients],
        "scope":"Observed training speed histograms; mean client-to-pooled Jensen-Shannon divergence in bits"}


def aggregate(cases):
    conditions={}
    for key in dict.fromkeys(f"{c['partition_rule']}-{c['round_budget']}" for c in cases):
        values=[c for c in cases if f"{c['partition_rule']}-{c['round_budget']}"==key]
        def stats(items):
            return {"mean":float(np.mean(items)),"sd":float(np.std(items,ddof=1)) if len(items)>1 else None}
        conditions[key]={"n":len(values),"seeds":[c["seed"] for c in values],"partition_rule":values[0]["partition_rule"],"round_budget":values[0]["round_budget"],
            "fedavg_mae":stats([c["fedavg"]["test"]["mae"] for c in values]),
            "fedavg_rmse":stats([c["fedavg"]["test"]["rmse"] for c in values]),
            "centralized_mae":stats([c["centralized"]["test"]["mae"] for c in values]),
            "centralized_rmse":stats([c["centralized"]["test"]["rmse"] for c in values]),
            "persistence_mae":stats([c["centralized"]["persistence_test"]["mae"] for c in values]),
            "parameter_bytes":values[0]["fedavg"]["communication"]["total_parameter_bytes"],
            "mean_training_js_bits":values[0]["heterogeneity"]["mean_js_bits"],
            "wall_seconds":stats([c["wall_seconds"] for c in values])}
    reference_checks={}
    prefix_checks={}
    for seed in sorted(set(c["seed"] for c in cases)):
        balanced=next(c for c in cases if c["seed"]==seed and c["partition_rule"]=="balanced" and c["round_budget"]==5)
        grouped=next(c for c in cases if c["seed"]==seed and c["partition_rule"]=="congestion")
        reference_checks[str(seed)]=balanced["centralized"]==grouped["centralized"]
        for short in [c for c in cases if c["seed"]==seed and c["partition_rule"]=="balanced"]:
            prefix_checks[short["case_id"]]=[r["state_sha256"] for r in short["rounds"]]==[r["state_sha256"] for r in balanced["rounds"][:short["round_budget"]]]
    if not all(reference_checks.values()) or not all(prefix_checks.values()):
        raise ValueError("Reference or round-prefix reproducibility check failed")
    b,g=conditions["balanced-5"],conditions["congestion-5"]
    brief=(f"The expanded Flower study used {cases[0]['config']['clients']*cases[0]['config']['sensors_per_client']} sensors, {cases[0]['config']['rows']/288:g} days, {cases[0]['config']['clients']} clients, and {len(reference_checks)} seeds. At five rounds, balanced partitions scored MAE {b['fedavg_mae']['mean']:.3f} mph versus {b['centralized_mae']['mean']:.3f} for the matched pooled reference. Congestion-grouped clients scored {g['fedavg_mae']['mean']:.3f} mph. Training histogram divergence was {b['mean_training_js_bits']:.4f} versus {g['mean_training_js_bits']:.4f} bits. The five-round parameter payload was {b['parameter_bytes']:,} bytes. Round budgets of 1/3/5 and all per-seed results are recorded.")
    return {"conditions":conditions,"centralized_reference_identical":reference_checks,"round_prefix_identical":prefix_checks,
        "brief_findings":brief,"uncertainty":"Sample SD across training seeds on the same held-out dates"}


def plot(out,cases,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    balanced=sorted([c for c in summary["conditions"].values() if c["partition_rule"]=="balanced"],key=lambda c:c["round_budget"])
    axes[0].errorbar([c["parameter_bytes"]/1024 for c in balanced],[c["fedavg_mae"]["mean"] for c in balanced],
        yerr=[c["fedavg_mae"]["sd"] or 0 for c in balanced],fmt="o-",capsize=4,label="Balanced")
    grouped=summary["conditions"]["congestion-5"]
    axes[0].errorbar([grouped["parameter_bytes"]/1024],[grouped["fedavg_mae"]["mean"]],yerr=[grouped["fedavg_mae"]["sd"] or 0],fmt="s",capsize=4,label="Congestion grouped, 5 rounds")
    for c in balanced:
        axes[0].annotate(f" {c['round_budget']} rounds",(c["parameter_bytes"]/1024,c["fedavg_mae"]["mean"]),fontsize=8)
    axes[0].set(xlabel="Estimated parameter payload (KiB)",ylabel="Selected-model test MAE (mph)",title="Accuracy and communication budget")
    axes[0].legend(fontsize=8)
    for rule,label in [("balanced","Balanced"),("congestion","Congestion grouped")]:
        selected=[c for c in cases if c["partition_rule"]==rule and c["round_budget"]==5]
        values=np.array([[r["validation"]["mae"] for r in c["rounds"]] for c in selected])
        mean,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
        axes[1].plot(range(1,6),mean,label=label)
        axes[1].fill_between(range(1,6),mean-sd,mean+sd,alpha=.15)
    axes[1].set(xlabel="Communication round",ylabel="Validation MAE (mph)",title="Validation convergence, mean and seed SD")
    axes[1].set_xticks(range(1,6))
    axes[1].legend(fontsize=8)
    fig.savefig(out/"tradeoffs.png",dpi=150)
    plt.close(fig)


def run(settings,data,out):
    base=FederatedConfig(**settings["base"])
    base.validate()
    seeds=settings["seeds"]
    if len(seeds)<2 or len(set(seeds))!=len(seeds):
        raise ValueError("The study needs multiple distinct seeds")
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"config.json",settings)
    write_json(out/"status.json",{"status":"running"})
    cases=[]
    try:
        for seed in seeds:
            for rule,budgets in [("balanced",settings["balanced_round_budgets"]),("congestion",settings["congestion_round_budgets"])]:
                for budget in budgets:
                    config=replace(base,seed=seed,rounds=budget,partition_rule=rule)
                    case_id=f"{rule[0]}{budget}-s{seed}"
                    start=time.perf_counter()
                    job=prepare(config,data,out/case_id)
                    execute(job)
                    def read(name):
                        return json.loads((job/name).read_text())
                    clients=read("clients.json")
                    case={"case_id":case_id,"seed":seed,"partition_rule":rule,"round_budget":budget,
                        "config":asdict(config),"fedavg":read("federated.json"),"centralized":read("centralized.json"),
                        "clients":clients,"rounds":read("rounds.json"),"centralized_history":read("centralized-history.json"),
                        "heterogeneity":heterogeneity(clients),"wall_seconds":time.perf_counter()-start}
                    cases.append(case)
                    write_json(out/"cases.json",cases)
                    print(f"{case_id}: FedAvg MAE={case['fedavg']['test']['mae']:.6f}; central={case['centralized']['test']['mae']:.6f}; wall={case['wall_seconds']:.1f}s",flush=True)
        summary=aggregate(cases)
        write_json(out/"summary.json",summary)
        plot(out,cases,summary)
        write_json(out/"status.json",{"status":"complete"})
        return summary
    except BaseException as error:
        write_json(out/"status.json",{"status":"failed","type":type(error).__name__,"error":str(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/federated-study.json")
    parser.add_argument("--data",default="data/raw/metr-la.h5")
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.data,args.out)


if __name__=="__main__":
    main()
