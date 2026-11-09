"""Store evaluation results, summarize them, and build the GRPO learning curve."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


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
