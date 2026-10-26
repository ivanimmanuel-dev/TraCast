"""Compare completed real experiments without comparing runtime log timestamps."""
import argparse
import json
from pathlib import Path

import torch
from .experiment import write_json


def compare(first, second):
    first, second = Path(first), Path(second)
    checks = {}
    for name in ("config.json", "clients.json", "initial.json", "rounds.json", "federated.json",
                 "centralized.json", "centralized-history.json", "status.json"):
        left, right = json.loads((first / name).read_text()), json.loads((second / name).read_text())
        checks[name] = left == right
    manifests = [json.loads((path / "manifest.json").read_text()) for path in (first, second)]
    checks["run_id"] = manifests[0]["run_id"] == manifests[1]["run_id"]
    checks["partition_hashes"] = manifests[0]["partition_files"] == manifests[1]["partition_files"]
    checks["completed"] = all(json.loads((path / "status.json").read_text())["status"] == "complete" for path in (first, second))
    filenames = sorted(path.name for path in first.glob("*.pt"))
    checks["checkpoint_names"] = filenames == sorted(path.name for path in second.glob("*.pt"))
    checks["checkpoint_tensors"] = bool(filenames)
    for name in filenames:
        left = torch.load(first / name, map_location="cpu", weights_only=True)
        right = torch.load(second / name, map_location="cpu", weights_only=True)
        checks["checkpoint_tensors"] &= left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
    return checks


def main():
    parser = argparse.ArgumentParser(description="Compare TraCast federated experiment outputs")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out")
    args = parser.parse_args()
    checks = compare(args.first, args.second)
    if args.out:
        write_json(args.out, checks)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
