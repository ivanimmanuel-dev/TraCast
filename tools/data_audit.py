"""Facts about the data that the experiments depend on, written to reports/data-audit.json."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from tracast import data, topology

ROOT = Path(__file__).resolve().parents[1]


def components(weights):
    undirected = np.maximum(weights, weights.T) > 0
    seen, sizes = -np.ones(len(weights), int), []
    for start in range(len(weights)):
        if seen[start] >= 0:
            continue
        stack, seen[start], size = [start], len(sizes), 0
        while stack:
            node = stack.pop()
            size += 1
            for other in np.flatnonzero(undirected[node]):
                if seen[other] < 0:
                    seen[other] = len(sizes)
                    stack.append(other)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def main():
    frame = data.load(ROOT / "data/raw/metr-la.h5", 207)
    graph = ROOT / "data/topology"
    ids = (graph / "graph_sensor_ids.txt").read_text().strip().split(",")
    distances = pd.read_csv(
        graph / "distances_la_2012.csv", dtype={"from": str, "to": str}
    )
    weights, meta = topology.adjacency(distances, ids)
    parts = data.split(frame)
    values = frame.to_numpy(dtype=float)
    missing = np.isnan(values) | (values == 0)
    speeds = np.where(missing, np.nan, values)
    bounds = np.cumsum([0] + [len(p) for p in parts.values()])
    splits = {}
    for (name, part), start, stop in zip(parts.items(), bounds, bounds[1:]):
        block = slice(start, stop)
        per_sensor = missing[block].mean(axis=0)
        splits[name] = {
            "first": str(part.index[0]),
            "last": str(part.index[-1]),
            "rows": len(part),
            "missing_share": float(missing[block].mean()),
            "sensors_over_30_percent_missing": int((per_sensor > 0.3).sum()),
            "worst_sensor_missing_share": float(per_sensor.max()),
            "mean_speed": float(np.nanmean(speeds[block])),
            "share_below_40_mph": float((speeds[block][~missing[block]] < 40).mean()),
        }
    weekday = frame.index.dayofweek < 5
    hours = np.floor(frame.index.hour + frame.index.minute / 60)
    slowest = {}
    for label, rows in (("weekday", weekday), ("weekend", ~weekday)):
        profile = [np.nanmean(speeds[rows & (hours == h)]) for h in range(24)]
        slowest[label] = [int(h) for h in np.argsort(profile)[:3]]
    audit = {
        "sensors": frame.shape[1],
        "timestamps": frame.shape[0],
        "graph": {
            "directed_edges": meta["directed_edges"],
            "components": components(weights),
            "sensor_order_matches_graph_file": list(map(str, frame.columns)) == ids,
        },
        "splits": splits,
        "slowest_hours_of_day": slowest,
        "note": (
            "The slowest weekday hours are the evening and morning rush, so the timestamps "
            "behave like local time. missing_share counts NaN and zero readings."
        ),
    }
    out = ROOT / "reports/data-audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, indent=1)[:1500])


if __name__ == "__main__":
    main()
