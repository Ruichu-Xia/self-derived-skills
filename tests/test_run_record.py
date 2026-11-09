"""Tests for the per-run reproducibility record."""

import json

from self_derived_skills.run_record import run_git, write_run_record


def test_run_record_captures_config_code_and_environment(tmp_path):
    """A run record holds what is needed to trace a result back to its inputs."""
    output_dir = tmp_path / "nested" / "run"
    record = json.loads(write_run_record(output_dir, {"seed": 230}, "test-device").read_text())
    assert record["config"] == {"seed": 230}
    assert record["device"] == "test-device"
    assert record["git_commit"] == run_git("rev-parse", "HEAD")
    assert "math" in record["dataset_revisions"]
    assert "math-verify" in {name.lower() for name in record["packages"]}
