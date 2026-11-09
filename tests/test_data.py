"""Tests for problem IDs, level parsing, and the stratified dev split."""

import random
from collections import Counter

import pytest

from self_derived_skills.data import (
    MathProblem,
    allocate_by_level,
    make_problem_id,
    parse_level,
    row_to_problem,
    select_dev_ids,
    split_by_ids,
)


def make_problems(level_sizes: dict[int, int]) -> list[MathProblem]:
    """Build synthetic problems with the given number of problems per level."""
    return [
        MathProblem(make_problem_id(f"{level}-{i}"), f"{level}-{i}", "", "1", level, "algebra")
        for level, size in level_sizes.items()
        for i in range(size)
    ]


def test_problem_id_ignores_whitespace_only():
    """Reformatted copies of a problem share an ID, and different problems do not."""
    assert make_problem_id("Find  x.\n") == make_problem_id("Find x.")
    assert make_problem_id("Find x.") != make_problem_id("Find y.")


def test_parse_level():
    """Known labels parse, 'Level ?' means unknown, and anything else fails."""
    assert parse_level("Level 3") == 3
    assert parse_level("Level ?") is None
    with pytest.raises(ValueError, match="Unexpected MATH level"):
        parse_level("Level 9")


@pytest.mark.parametrize(
    ("level", "solution"),
    [
        ("Level ?", r"So \boxed{5}."),
        ("Level 3", r"So \boxed 5."),
        ("Level 3", r"So \boxed{}."),
    ],
)
def test_rows_without_a_usable_level_or_answer_are_dropped(level, solution):
    """Rows that could not be graded never become problems."""
    row = {"problem": "Find x.", "solution": solution, "level": level}
    assert row_to_problem(row, "algebra") is None


def test_usable_row_becomes_a_problem():
    """A valid row keeps its level, subject, and extracted gold answer."""
    row = {"problem": "Find x.", "solution": r"So \boxed{\frac{1}{2}}.", "level": "Level 3"}
    problem = row_to_problem(row, "algebra")
    assert (problem.answer, problem.level, problem.subject) == (r"\frac{1}{2}", 3, "algebra")


def test_allocation_sums_to_total_and_is_proportional():
    """Quotas add up exactly and follow the level sizes."""
    quotas = allocate_by_level({1: 564, 2: 1348, 3: 1592, 4: 1690, 5: 2304}, 1000)
    assert sum(quotas.values()) == 1000
    assert quotas == {1: 75, 2: 180, 3: 212, 4: 226, 5: 307}


def test_dev_selection_is_stratified_and_order_independent():
    """The dev set matches the level quotas and does not depend on input order."""
    level_sizes = {1: 100, 2: 300, 3: 600}
    problems = make_problems(level_sizes)
    dev_ids = select_dev_ids(problems, 100, seed=1)

    shuffled = random.Random(0).sample(problems, len(problems))
    assert select_dev_ids(shuffled, 100, seed=1) == dev_ids

    level_of = {problem.problem_id: problem.level for problem in problems}
    assert Counter(level_of[i] for i in dev_ids) == {1: 10, 2: 30, 3: 60}


def test_different_seeds_select_different_dev_sets():
    """The seed controls which problems are selected."""
    problems = make_problems({1: 500})
    assert select_dev_ids(problems, 50, seed=1) != select_dev_ids(problems, 50, seed=2)


@pytest.mark.parametrize("n_dev", [0, 10])
def test_invalid_dev_size_raises(n_dev):
    """A dev set must be non-empty and smaller than the population."""
    with pytest.raises(ValueError, match="n_dev must be"):
        select_dev_ids(make_problems({1: 10}), n_dev, seed=1)


def test_split_partitions_problems():
    """Train and dev are disjoint and together cover every problem."""
    problems = make_problems({1: 20})
    dev_ids = [problem.problem_id for problem in problems[:5]]
    train, dev = split_by_ids(problems, dev_ids)
    assert {p.problem_id for p in dev} == set(dev_ids)
    assert {p.problem_id for p in train}.isdisjoint(dev_ids)
    assert len(train) + len(dev) == len(problems)


def test_split_fails_on_unknown_dev_id():
    """A manifest that does not match the data fails instead of shrinking the dev set."""
    with pytest.raises(ValueError, match="missing from the problems"):
        split_by_ids(make_problems({1: 5}), ["not-a-real-id"])
