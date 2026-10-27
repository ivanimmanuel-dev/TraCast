"""Normalize workstation paths in report copies; retain original-content hashes."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    workspace = ROOT.parents[1]
    aliases = [
        (ROOT, "$REPO"),
        (ROOT.parent / "TrafficMind", "$REPO"),
        (workspace / "work" / "flwr-venv", "$PYTHON_ENV"),
        (workspace / "work" / "venv", "$PYTHON_ENV"),
    ]
    replacements = []
    for source, target in aliases:
        source = str(source)
        replacements.extend(
            [
                (source, target),
                (source.replace("\\", "/"), target),
                (source.replace("\\", "\\\\"), target),
            ]
        )

    def normalize(value):
        for source, target in replacements:
            value = value.replace(source, target)
        return value

    def convert(value):
        if isinstance(value, str):
            return normalize(value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    def numerical(value, path=""):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result.update(numerical(item, path + "/" + key))
            return result
        if isinstance(value, list):
            result = {}
            for i, item in enumerate(value):
                result.update(numerical(item, path + f"/{i}"))
            return result
        if isinstance(value, (int, float, bool)) or path.endswith(
            ("sha256", "run_id", "git_commit")
        ):
            return {path: value}
        return {}

    audit_path = REPORTS / "export-provenance.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for path in sorted(REPORTS.rglob("*")):
        if path == audit_path or path.suffix not in (".json", ".log"):
            continue
        original = path.read_bytes()
        if path.suffix == ".json":
            before = json.loads(original)
            after = convert(before)
            assert numerical(before) == numerical(after), path
            if before == after:
                continue
            exported = (
                json.dumps(after, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode()
        else:
            before = original.decode("utf-8")
            after = normalize(before)
            if before == after:
                continue
            exported = after.encode()
        path.write_bytes(exported)
        audit[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "export_sha256": hashlib.sha256(exported).hexdigest(),
            "operation": "Workstation path strings replaced by $REPO/$PYTHON_ENV; numerical values and identity hashes unchanged",
        }
    audit_path.write_bytes(
        (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    )
    print(f"Path-normalized records: {len(audit)}")


if __name__ == "__main__":
    main()
