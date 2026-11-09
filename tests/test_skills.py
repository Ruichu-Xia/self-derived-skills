"""Tests for choosing attempts, filtering written skills, and loading the bank."""

import json

import pytest

from self_derived_skills.skills import load_skill_bank, rejection_reason, select_attempts

PROBLEM = "Find the sum of 7 and 5."


def attempt(text, correct, truncated=False):
    """Build one sampled attempt."""
    return {"completion": text, "correct": correct, "truncated": truncated}


def test_select_attempts_pairs_correct_with_a_finished_incorrect_attempt():
    """An incorrect attempt cut off at the limit is skipped in favor of a finished one."""
    rollouts = [
        attempt("cut off", False, truncated=True),
        attempt("wrong", False),
        attempt("right", True),
    ]
    assert select_attempts(rollouts) == ("right", "wrong")


def test_select_attempts_uses_only_the_correct_attempt_when_nothing_else_finished():
    """A group with no finished incorrect attempt still yields a skill source."""
    rollouts = [attempt("right", True), attempt("cut off", False, truncated=True)]
    assert select_attempts(rollouts) == ("right", None)


def test_select_attempts_returns_none_when_every_attempt_fails():
    """Problems with no correct attempt produce no skill."""
    assert select_attempts([attempt("wrong", False), attempt("also wrong", False)]) is None


@pytest.mark.parametrize(
    ("skill", "truncated", "answer", "expected"),
    [
        ("Add the terms before simplifying.", False, "12", None),
        ("Use 2 as the base.", False, "12", None),
        ("Solve for x first.", False, "x", None),
        ("", False, "12", "empty"),
        ("Add the terms", True, "12", "truncated"),
        (r"The result is \boxed{a}.", False, "12", "boxed"),
        ("Expect a total of 12.", False, "12", "gold_answer"),
        ("The result is even.", False, r"\text{even}", "gold_answer"),
        (r"It equals \sqrt{ x }.", False, r"\sqrt{x}", "gold_answer"),
        ("Multiply by 7 first.", False, "12", "problem_number"),
        ("Add the terms in this case.", False, "12", "specific_wording"),
    ],
)
def test_rejection_reason(skill, truncated, answer, expected):
    """Skills that leak the problem or refer to it are dropped, and general ones pass."""
    assert rejection_reason(skill, truncated, PROBLEM, answer) == expected


def test_load_skill_bank_reads_the_skills(tmp_path):
    """The bank's skills are returned in order."""
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"skills": ["one", "two"]}))
    assert load_skill_bank(path) == ["one", "two"]


def test_load_skill_bank_rejects_an_empty_bank(tmp_path):
    """Training on an empty bank would silently be standard GRPO."""
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"skills": []}))
    with pytest.raises(ValueError, match="no skills"):
        load_skill_bank(path)
