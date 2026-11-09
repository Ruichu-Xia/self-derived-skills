"""Store evaluation results, summarize them, and build the GRPO learning curve."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from self_derived_skills.reward import summarize_groups


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load per-problem results, ignoring a final line left incomplete by an interruption."""
    if not path.exists():
        return []

    lines = [line for line in path.read_text().splitlines() if line.strip()]
    results = []
    for index, line in enumerate(lines):
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return results


def append_results(path: Path, results: Iterable[dict[str, Any]]) -> None:
    """Append per-problem results to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text()
        if text and not text.endswith("\n"):
            # An interrupted write left a partial line, which new results must not join.
            path.write_text(text[: text.rfind("\n") + 1])
    with path.open("a") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")


def accuracy_summary(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Return the count, accuracy, and binomial standard error of some results."""
    n = len(results)
    if n == 0:
        raise ValueError("Cannot summarize zero results")
    accuracy = sum(result["correct"] for result in results) / n
    standard_error = math.sqrt(accuracy * (1 - accuracy) / n)
    return {"n": n, "accuracy": accuracy, "standard_error": standard_error}


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize results overall and by difficulty level and subject."""
    groups: dict[str, dict[Any, list]] = {"level": defaultdict(list), "subject": defaultdict(list)}
    for result in results:
        for field, group in groups.items():
            group[result[field]].append(result)

    return {
        **accuracy_summary(results),
        "truncated_frac": sum(result["truncated"] for result in results) / len(results),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in results) / len(results),
        "by_level": {str(k): accuracy_summary(v) for k, v in sorted(groups["level"].items())},
        "by_subject": {str(k): accuracy_summary(v) for k, v in sorted(groups["subject"].items())},
    }


def summarize_rollouts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize sampled rollouts: success, group contrast, truncation, and construction cost."""
    rollouts = [rollout for record in records for rollout in record["rollouts"]]
    rewards = [float(rollout["correct"]) for rollout in rollouts]
    return {
        "num_problems": len(records),
        "num_rollouts": len(rollouts),
        "accuracy": sum(rewards) / len(rollouts),
        "truncated_frac": sum(rollout["truncated"] for rollout in rollouts) / len(rollouts),
        "completion_tokens": sum(rollout["completion_tokens"] for rollout in rollouts),
        "seconds": sum(record["seconds"] for record in records),
        "groups": summarize_groups(rewards, len(records[0]["rollouts"])),
    }


def compare_skill_pilot(
    without: dict[str, Any], with_skills: dict[str, Any]
) -> dict[str, float | bool]:
    """Apply the pilot gate: skills must not lower accuracy and must raise the mixed-group share."""
    if without["num_rollouts"] != with_skills["num_rollouts"]:
        raise ValueError("The two pilot conditions cover different numbers of rollouts")

    mixed_without = without["groups"]["mixed"] / without["num_problems"]
    mixed_with = with_skills["groups"]["mixed"] / with_skills["num_problems"]
    return {
        "num_problems": without["num_problems"],
        "accuracy_without": without["accuracy"],
        "accuracy_with": with_skills["accuracy"],
        "mixed_share_without": mixed_without,
        "mixed_share_with": mixed_with,
        "passed": with_skills["accuracy"] >= without["accuracy"] and mixed_with > mixed_without,
    }


def build_learning_curve(
    log_history: Sequence[dict[str, Any]], evaluations: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join accuracy at each evaluated step with cumulative sampled tokens and training time."""
    sampled_tokens = {0: 0.0}
    training_seconds = {0: 0.0}
    elapsed = 0.0
    for row in log_history:
        if "tokens/sampled_total" in row:
            elapsed += row["step_time"]
            sampled_tokens[row["step"]] = row["tokens/sampled_total"]
            training_seconds[row["step"]] = elapsed

    return [
        {
            "step": step,
            "n": evaluations[step]["n"],
            "accuracy": evaluations[step]["accuracy"],
            "standard_error": evaluations[step]["standard_error"],
            "sampled_tokens": sampled_tokens[step],
            "training_seconds": training_seconds[step],
        }
        for step in sorted(evaluations)
    ]
