"""Tests for the shared prompt format."""

from self_derived_skills.prompts import build_math_messages
from self_derived_skills.reward import BOXED_PREFIX


def test_prompt_contains_problem_and_requests_the_rewarded_format():
    """The prompt carries the problem and asks for the boxed format the reward reads."""
    messages = build_math_messages("Find x.")
    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"].startswith("Find x.")
    assert BOXED_PREFIX in messages[0]["content"]
