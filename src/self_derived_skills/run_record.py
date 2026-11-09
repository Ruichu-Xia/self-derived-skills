"""Record what produced a run so that its results can be traced and reproduced."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from self_derived_skills.data import MATH_500_REVISION, MATH_REVISION


def run_git(*args: str) -> str:
    """Return the stripped output of a git command, failing if it fails."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def write_run_record(output_dir: Path, config: dict[str, Any], device_name: str) -> None:
    """Write run.json with the command, config, code version, data revisions, and environment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "started_at": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "config": config,
        "git_commit": run_git("rev-parse", "HEAD"),
        "git_dirty": bool(run_git("status", "--porcelain")),
        "dataset_revisions": {"math": MATH_REVISION, "math_500": MATH_500_REVISION},
        "device": device_name,
        "python": platform.python_version(),
        "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
    }
    (output_dir / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
