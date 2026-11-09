"""Choose the attempts a skill is written from, filter written skills, and load the bank."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Three or more letters, excluding LaTeX command names such as \frac.
ANSWER_WORD = re.compile(r"(?<![\\a-zA-Z])[a-zA-Z]{3,}")
SPECIFIC_WORDING = re.compile(r"\b(this|these|here|above)\b", re.IGNORECASE)


def select_attempts(rollouts: Sequence[dict[str, Any]]) -> tuple[str, str | None] | None:
    """Return the first correct and first finished incorrect attempt, or None if none is correct.

    An incorrect attempt that hit the token limit has no answer to contrast, so it is not used.
    """
    correct = next((r["completion"] for r in rollouts if r["correct"]), None)
    if correct is None:
        return None
    incorrect = next(
        (r["completion"] for r in rollouts if not r["correct"] and not r["truncated"]), None
    )
    return correct, incorrect


def contains_answer(skill: str, answer: str) -> bool:
    """Return whether a skill states the gold answer, as a number, a word, or the expression."""
    if set(NUMBER.findall(answer)) & set(NUMBER.findall(skill)):
        return True
    compact_skill = "".join(skill.split()).lower()
    compact_answer = "".join(answer.split()).lower()
    # One or two characters, such as x, match too much ordinary text to be evidence of a leak.
    if len(compact_answer) >= 3 and compact_answer in compact_skill:
        return True
    skill_words = set(re.findall(r"[a-z]+", skill.lower()))
    return any(word.lower() in skill_words for word in ANSWER_WORD.findall(answer))


def rejection_reason(skill: str, truncated: bool, problem: str, answer: str) -> str | None:
    """Return why a written skill must be dropped, or None if it passes the filter."""
    if not skill:
        return "empty"
    if truncated:
        return "truncated"
    if "\\boxed" in skill:
        return "boxed"
    if contains_answer(skill, answer):
        return "gold_answer"
    if set(NUMBER.findall(problem)) & set(NUMBER.findall(skill)):
        return "problem_number"
    if SPECIFIC_WORDING.search(skill):
        return "specific_wording"
    return None


def load_skill_bank(path: Path) -> list[str]:
    """Load the skills from a bank file, failing if there are none."""
    skills = json.loads(path.read_text())["skills"]
    if not skills:
        raise ValueError(f"{path} contains no skills")
    return skills
