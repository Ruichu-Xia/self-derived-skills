"""Tests for the GRPO training dataset."""

import pytest

pytest.importorskip("trl")

from self_derived_skills.data import MathProblem  # noqa: E402
from self_derived_skills.prompts import build_math_messages  # noqa: E402
from self_derived_skills.train import make_train_dataset  # noqa: E402

PROBLEMS = [
    MathProblem(problem_id="a", problem="Find x.", solution="", answer="5", level=1, subject="s"),
    MathProblem(problem_id="b", problem="Find y.", solution="", answer="6", level=2, subject="s"),
]


def test_train_dataset_without_skills_uses_the_plain_prompt():
    """Standard GRPO trains on exactly the evaluation prompt."""
    dataset = make_train_dataset(PROBLEMS)
    assert dataset["prompt"][0] == build_math_messages("Find x.")


def test_train_dataset_adds_skills_without_changing_the_rows():
    """Skills change only the prompts, so TRL samples the same problems in the same order."""
    dataset = make_train_dataset(PROBLEMS, skills=["Check your work."])
    assert dataset["problem_id"] == ["a", "b"]
    assert all("Check your work." in prompt[0]["content"] for prompt in dataset["prompt"])
