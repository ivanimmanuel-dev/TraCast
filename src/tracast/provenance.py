"""Source-control provenance recorded in every run manifest."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def _git(root, *args):
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state(root=ROOT):
    """Commit and dirty flag for the repo; both None outside a checkout."""
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "git_commit": commit or None,
        "git_dirty": None if status is None else bool(status),
    }
