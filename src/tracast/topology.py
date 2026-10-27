"""Pinned METR-LA road-distance graph from the DCRNN author repository."""

from collections import deque
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from .data import digest

REVISION = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
FILES = {
    "distances_la_2012.csv": "a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5",
    "graph_sensor_ids.txt": "3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990",
    "graph_sensor_locations.csv": "eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f",
}


def fetch(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = directory / name
        if path.exists():
            if digest(path) != expected:
                raise ValueError(f"Topology checksum mismatch: {name}")
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            url = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{REVISION}/data/sensor_graph/{name}"
            with (
                urllib.request.urlopen(url, timeout=90) as response,
                partial.open("wb") as target,
            ):
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if digest(partial) != expected:
                raise ValueError(f"Topology checksum mismatch: {name}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return directory


def validate_alignment(columns, sensor_ids):
    columns, sensor_ids = list(map(str, columns)), list(map(str, sensor_ids))
    if len(set(columns)) != len(columns) or len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Duplicate sensor IDs")
    if set(columns) != set(sensor_ids):
        raise ValueError("Traffic and graph sensor IDs differ")
    return {sensor: columns.index(sensor) for sensor in sensor_ids}


def adjacency(distances, sensor_ids, threshold=0.1):
    if not 0 < threshold < 1:
        raise ValueError("Graph threshold must be between zero and one")
    if set(distances.columns) != {"from", "to", "cost"}:
        raise ValueError("Expected from/to/cost distance columns")
    if not sensor_ids or len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Expected unique sensor IDs")
    if distances[["from", "to"]].isna().any().any():
        raise ValueError("Missing distance endpoint")
    costs = distances["cost"].to_numpy(dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Distances must be finite and nonnegative")
    selected = distances[
        distances["from"].isin(sensor_ids) & distances["to"].isin(sensor_ids)
    ]
    if selected.duplicated(["from", "to"]).any():
        raise ValueError("Duplicate directed distance pairs")
    scale = float(selected["cost"].to_numpy(dtype=float).std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Distance kernel needs a positive scale")
    index = {sensor: i for i, sensor in enumerate(sensor_ids)}
    weights = np.zeros((len(sensor_ids), len(sensor_ids)), dtype=np.float32)
    for start, end, cost in selected.itertuples(index=False, name=None):
        value = np.exp(-((float(cost) / scale) ** 2))
        if start != end and value >= threshold:
            weights[index[start], index[end]] = value
    return weights, {
        "distance_std": scale,
        "threshold": threshold,
        "source_pairs": len(distances),
        "aligned_pairs": len(selected),
        "directed_edges": int(np.count_nonzero(weights)),
        "self_edges": "excluded; own value is a separate feature",
    }


def connected_subset(weights, sensor_ids, count):
    if weights.shape != (len(sensor_ids), len(sensor_ids)) or not 1 <= count <= len(
        sensor_ids
    ):
        raise ValueError("Invalid graph shape or subset size")
    # Traverse real edges in either direction. Stronger connections precede ties by ID.
    undirected = np.maximum(weights, weights.T)
    chosen, seen, queue = [], {0}, deque([0])
    while queue and len(chosen) < count:
        node = queue.popleft()
        chosen.append(node)
        neighbors = [i for i in np.flatnonzero(undirected[node]) if i not in seen]
        neighbors.sort(key=lambda i: (-float(undirected[node, i]), sensor_ids[i]))
        for other in neighbors:
            seen.add(other)
            queue.append(other)
    if len(chosen) != count:
        raise ValueError("The starting sensor's component is too small")
    return [sensor_ids[i] for i in chosen], weights[np.ix_(chosen, chosen)].copy()


def load(directory, columns, count, threshold=0.1):
    directory = Path(directory)
    for name, expected in FILES.items():
        if digest(directory / name) != expected:
            raise ValueError(f"Topology checksum mismatch: {name}")
    ids = (directory / "graph_sensor_ids.txt").read_text().strip().split(",")
    validate_alignment(columns, ids)
    distances = pd.read_csv(
        directory / "distances_la_2012.csv", dtype={"from": str, "to": str}
    )
    weights, metadata = adjacency(distances, ids, threshold)
    chosen, subgraph = connected_subset(weights, ids, count)
    locations = pd.read_csv(
        directory / "graph_sensor_locations.csv", dtype={"sensor_id": str}
    )
    if locations.sensor_id.duplicated().any() or set(locations.sensor_id) != set(ids):
        raise ValueError("Sensor location IDs differ")
    coordinates = locations.set_index("sensor_id").loc[
        chosen, ["latitude", "longitude"]
    ]
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Nonfinite sensor coordinates")
    metadata.update(
        {
            "revision": REVISION,
            "files": FILES,
            "sensor_ids": chosen,
            "subset_rule": "BFS from first canonical sensor; union of directions; descending weight then sensor ID",
            "subset_directed_edges": int(np.count_nonzero(subgraph)),
            "adjacency": subgraph.tolist(),
            "locations": coordinates.to_dict(orient="index"),
        }
    )
    return chosen, subgraph, metadata
