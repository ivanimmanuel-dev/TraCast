import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .data import fetch, load, missing_report, split
from .experiment import Config, plots, run, write_json


def main():
    parser = argparse.ArgumentParser(description="TraCast: METR-LA CPU forecasting")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch")
    download.add_argument("--data", default="data/raw/metr-la.h5")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--data", default="data/raw/metr-la.h5")
    inspect.add_argument("--sensors", type=int, default=207)
    inspect.add_argument("--out", default="runs/inspection")
    train = commands.add_parser("run")
    train.add_argument("--data", default="data/raw/metr-la.h5")
    train.add_argument("--config", default="configs/subset.json")
    train.add_argument("--out", required=True)
    commands.add_parser("config")
    args = parser.parse_args()
    if args.command == "fetch":
        print(fetch(args.data))
    elif args.command == "config":
        print(json.dumps(asdict(Config()), indent=2))
    elif args.command == "inspect":
        frame = load(args.data, args.sensors)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        report = missing_report(frame)
        write_json(out / "missing.json", report)
        plots(split(frame)["train"].iloc[:, :16], out)
        print(
            json.dumps(
                {key: val for key, val in report.items() if key != "per_sensor"},
                indent=2,
            )
        )
    else:
        config = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        results = run(config, args.data, args.out)
        print(
            json.dumps(
                {name: results[name]["test"] for name in ("persistence", "gru")},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
