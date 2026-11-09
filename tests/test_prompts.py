"""Tests for the shared prompt format."""

from self_derived_skills.prompts import (
    MATH_INSTRUCTION,
    SKILL_INSTRUCTION,
    build_math_messages,
    build_skill_messages,
)
from self_derived_skills.reward import BOXED_PREFIX


def test_skills_go_before_an_otherwise_unchanged_prompt():
    """Adding skills only prepends a tips block to the evaluation prompt."""
    plain = build_math_messages("Find x.")[0]["content"]
    with_skills = build_math_messages("Find x.", ["Tip one.", "Tip two."])[0]["content"]
    assert plain == f"Find x.\n\n{MATH_INSTRUCTION}"
    assert with_skills.endswith(plain)
    assert "- Tip one.\n- Tip two." in with_skills


def test_skill_prompt_includes_the_incorrect_attempt_only_when_there_is_one():
    """Both attempts are shown when available, and the instruction comes last."""
    contrast = build_skill_messages("Find x.", "good", "bad")[0]["content"]
    correct_only = build_skill_messages("Find x.", "good", None)[0]["content"]
    assert "good" in contrast and "An incorrect attempt:\nbad" in contrast
    assert "incorrect attempt" not in correct_only
    assert contrast.endswith(SKILL_INSTRUCTION)


def test_prompt_contains_problem_and_requests_the_rewarded_format():
    """The prompt carries the problem and asks for the boxed format the reward reads."""
    messages = build_math_messages("Find x.")
    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"].startswith("Find x.")
    assert BOXED_PREFIX in messages[0]["content"]
