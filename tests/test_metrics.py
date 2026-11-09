"""Tests for stored evaluation results and summary metrics."""

import json

import pytest

from self_derived_skills.metrics import (
    accuracy_summary,
    append_results,
    build_learning_curve,
    compare_skill_pilot,
    load_results,
    summarize_results,
    summarize_rollouts,
)


def make_result(correct, level=1, subject="algebra", truncated=False, tokens=10):
    """Build one per-problem result."""
    return {
        "correct": correct,
        "level": level,
        "subject": subject,
        "truncated": truncated,
        "completion_tokens": tokens,
    }


def test_accuracy_summary_matches_the_binomial_formula():
    """Accuracy and standard error match a hand-computed example."""
    summary = accuracy_summary([make_result(True)] * 20 + [make_result(False)] * 80)
    assert summary["n"] == 100
    assert summary["accuracy"] == pytest.approx(0.2)
    assert summary["standard_error"] == pytest.approx(0.04)


def test_summarize_results_breaks_down_by_level_and_subject():
    """Breakdowns count each group separately, and overall statistics cover every result."""
    summary = summarize_results(
        [
            make_result(True, level=1, subject="algebra", truncated=False, tokens=10),
            make_result(False, level=1, subject="geometry", truncated=True, tokens=30),
            make_result(False, level=5, subject="geometry", truncated=True, tokens=20),
        ]
    )
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["truncated_frac"] == pytest.approx(2 / 3)
    assert summary["mean_completion_tokens"] == pytest.approx(20)
    assert summary["by_level"]["1"]["accuracy"] == pytest.approx(0.5)
    assert summary["by_level"]["5"]["n"] == 1
    assert summary["by_subject"]["geometry"]["n"] == 2


def test_summarize_rejects_empty_results():
    """An empty evaluation is an error, not an accuracy of zero."""
    with pytest.raises(ValueError, match="zero results"):
        summarize_results([])


def test_results_round_trip_and_append(tmp_path):
    """Batches appended across calls load back in order."""
    path = tmp_path / "nested" / "results.jsonl"
    assert load_results(path) == []
    append_results(path, [{"problem_id": "a"}])
    append_results(path, [{"problem_id": "b"}])
    assert [result["problem_id"] for result in load_results(path)] == ["a", "b"]


def test_load_ignores_only_an_incomplete_final_line(tmp_path):
    """A line cut off by a disconnect is skipped so that the problem is simply redone."""
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"problem_id": "a"}) + '\n{"problem_id": "b", "corr')
    assert [result["problem_id"] for result in load_results(path)] == ["a"]


def test_append_after_an_interrupted_write_keeps_the_file_readable(tmp_path):
    """New results replace a partial final line instead of joining it."""
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"problem_id": "a"}) + '\n{"problem_id": "b", "corr')
    append_results(path, [{"problem_id": "c"}])
    append_results(path, [{"problem_id": "d"}])
    assert [result["problem_id"] for result in load_results(path)] == ["a", "c", "d"]


def test_summarize_rollouts_counts_groups_tokens_and_time():
    """Totals cover every rollout, and each problem's group is classified."""

    def rollout(correct, truncated=False, tokens=10):
        """Build one sampled rollout."""
        return {"correct": correct, "truncated": truncated, "completion_tokens": tokens}

    records = [
        {"rollouts": [rollout(True), rollout(False, truncated=True, tokens=30)], "seconds": 1.5},
        {"rollouts": [rollout(False), rollout(False)], "seconds": 2.5},
    ]
    summary = summarize_rollouts(records)
    assert (summary["num_problems"], summary["num_rollouts"]) == (2, 4)
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["truncated_frac"] == pytest.approx(0.25)
    assert summary["completion_tokens"] == 60
    assert summary["seconds"] == pytest.approx(4.0)
    assert summary["groups"] == {"all_fail": 1, "all_pass": 0, "mixed": 1}


def pilot_summary(accuracy, mixed):
    """Build a rollout summary for 10 problems with 40 rollouts."""
    return {
        "num_problems": 10,
        "num_rollouts": 40,
        "accuracy": accuracy,
        "groups": {"mixed": mixed},
    }


@pytest.mark.parametrize(
    ("accuracy_with", "mixed_with", "passed"),
    [
        (0.30, 5, True),
        (0.25, 5, True),
        (0.20, 5, False),
        (0.30, 3, False),
        (0.30, 2, False),
    ],
)
def test_pilot_gate_needs_kept_accuracy_and_more_mixed_groups(accuracy_with, mixed_with, passed):
    """Skills must not lower accuracy and must strictly raise the mixed-group share."""
    result = compare_skill_pilot(pilot_summary(0.25, 3), pilot_summary(accuracy_with, mixed_with))
    assert result["passed"] is passed
    assert result["mixed_share_without"] == pytest.approx(0.3)


def test_pilot_rejects_conditions_of_different_sizes():
    """The two conditions must be measured on the same number of rollouts."""
    other = {**pilot_summary(0.3, 5), "num_rollouts": 20}
    with pytest.raises(ValueError, match="different numbers"):
        compare_skill_pilot(pilot_summary(0.25, 3), other)


def test_load_rejects_corruption_before_the_final_line(tmp_path):
    """Damage anywhere else is real corruption and must not be skipped."""
    path = tmp_path / "results.jsonl"
    path.write_text('{"problem_id": "a", "corr\n' + json.dumps({"problem_id": "b"}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        load_results(path)


def test_learning_curve_accumulates_time_and_reads_token_totals():
    """Each evaluated step gets the tokens and summed training time up to that step."""
    log_history = [
        {"step": 1, "step_time": 10.0, "tokens/sampled_total": 100.0},
        {"step": 2, "step_time": 20.0, "tokens/sampled_total": 250.0},
        {"step": 2, "train_runtime": 30.0},
    ]
    evaluations = {
        0: {"n": 200, "accuracy": 0.2, "standard_error": 0.03},
        2: {"n": 200, "accuracy": 0.3, "standard_error": 0.03},
    }
    curve = build_learning_curve(log_history, evaluations)
    assert [row["step"] for row in curve] == [0, 2]
    assert (curve[0]["sampled_tokens"], curve[0]["training_seconds"]) == (0.0, 0.0)
    assert (curve[1]["sampled_tokens"], curve[1]["training_seconds"]) == (250.0, 30.0)


def test_learning_curve_fails_when_an_evaluated_step_was_never_logged():
    """An evaluation without a matching training log cannot be placed on the curve."""
    with pytest.raises(KeyError):
        build_learning_curve([], {20: {"n": 1, "accuracy": 0.0, "standard_error": 0.0}})
