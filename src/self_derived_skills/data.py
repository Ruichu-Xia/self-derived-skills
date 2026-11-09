"""Load the MATH datasets at pinned revisions and define the train and dev split."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from self_derived_skills.reward import last_boxed_content

MATH_DATASET = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
# 7,500 rows minus 2 with unknown levels, 2 with a brace-free box, 2 with an empty box, 1 repeat.
MATH_TRAIN_USABLE_SIZE = 7493

MATH_500_DATASET = "HuggingFaceH4/MATH-500"
MATH_500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH_500_SIZE = 500

# Evaluating every checkpoint on the full dev set would take hours on a T4.
DEV_SUBSET_SIZE = 200


@dataclass(frozen=True)
class MathProblem:
    """One MATH problem with its gold answer."""

    problem_id: str
    problem: str
    solution: str
    answer: str
    level: int
    subject: str


def make_problem_id(problem: str) -> str:
    """Return a stable ID derived from the whitespace-normalized problem text."""
    normalized = " ".join(problem.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def parse_level(level: str) -> int | None:
    """Return the integer level from a label like 'Level 3', or None for 'Level ?'."""
    if level == "Level ?":
        return None
    prefix, number = level.split(" ")
    if prefix != "Level" or number not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Unexpected MATH level label: {level!r}")
    return int(number)


def row_to_problem(row: dict[str, str], subject: str) -> MathProblem | None:
    """Convert a raw MATH row, or return None if its level or gold answer is unusable."""
    level = parse_level(row["level"])
    answer = last_boxed_content(row["solution"])
    if level is None or not answer:
        return None
    problem_id = make_problem_id(row["problem"])
    return MathProblem(problem_id, row["problem"], row["solution"], answer, level, subject)


def load_math_train() -> list[MathProblem]:
    """Load the usable, unique problems in MATH train."""
    problems_by_id = {}
    for subject in MATH_SUBJECTS:
        rows = load_dataset(MATH_DATASET, subject, split="train", revision=MATH_REVISION)
        for row in rows:
            problem = row_to_problem(row, subject)
            # Repeats must collapse before splitting, or one problem could land in both splits.
            if problem is not None:
                problems_by_id.setdefault(problem.problem_id, problem)

    problems = list(problems_by_id.values())
    if len(problems) != MATH_TRAIN_USABLE_SIZE:
        raise ValueError(f"Expected {MATH_TRAIN_USABLE_SIZE} MATH problems, got {len(problems)}")
    return problems


def load_math_500() -> list[MathProblem]:
    """Load MATH-500, the held-out final evaluation set."""
    rows = load_dataset(MATH_500_DATASET, split="test", revision=MATH_500_REVISION)
    problems = [
        MathProblem(
            make_problem_id(row["problem"]),
            row["problem"],
            row["solution"],
            row["answer"],
            int(row["level"]),
            row["subject"],
        )
        for row in rows
    ]
    if len(problems) != MATH_500_SIZE:
        raise ValueError(f"Expected {MATH_500_SIZE} MATH-500 problems, got {len(problems)}")
    return problems


def allocate_by_level(level_sizes: dict[int, int], total: int) -> dict[int, int]:
    """Split a total across levels in proportion to their sizes, by largest remainder."""
    population = sum(level_sizes.values())
    exact = {level: total * size / population for level, size in level_sizes.items()}
    quotas = {level: int(share) for level, share in exact.items()}
    by_remainder = sorted(exact, key=lambda level: (quotas[level] - exact[level], level))
    for level in by_remainder[: total - sum(quotas.values())]:
        quotas[level] += 1
    return quotas


def select_dev_ids(problems: Sequence[MathProblem], n_dev: int, seed: int) -> list[str]:
    """Pick dev problem IDs stratified by level, independent of the input order."""
    if not 0 < n_dev < len(problems):
        raise ValueError(f"n_dev must be between 0 and {len(problems)}, got {n_dev}")

    ids_by_level = defaultdict(list)
    for problem in problems:
        ids_by_level[problem.level].append(problem.problem_id)
    quotas = allocate_by_level({level: len(ids) for level, ids in ids_by_level.items()}, n_dev)

    dev_ids = []
    for level, ids in ids_by_level.items():
        ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
        dev_ids.extend(ranked[: quotas[level]])
    return sorted(dev_ids)


def split_by_ids(
    problems: Sequence[MathProblem], dev_ids: Sequence[str]
) -> tuple[list[MathProblem], list[MathProblem]]:
    """Partition problems into (train, dev), failing if any dev ID is not present."""
    dev_id_set = set(dev_ids)
    missing = dev_id_set - {problem.problem_id for problem in problems}
    if missing:
        raise ValueError(f"{len(missing)} dev IDs are missing from the problems")
    train = [problem for problem in problems if problem.problem_id not in dev_id_set]
    dev = [problem for problem in problems if problem.problem_id in dev_id_set]
    return train, dev


def load_math_splits(manifest_path: Path) -> tuple[list[MathProblem], list[MathProblem]]:
    """Load MATH train and return its (train, dev) partition from a split manifest."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != MATH_REVISION:
        raise ValueError(f"Manifest revision {manifest['revision']} is not {MATH_REVISION}")
    return split_by_ids(load_math_train(), manifest["dev_ids"])


def load_eval_problems(split: str, manifest_path: Path) -> list[MathProblem]:
    """Return the problems in an evaluation split: dev, dev-subset, or math500."""
    if split == "math500":
        return load_math_500()
    if split not in {"dev", "dev-subset"}:
        raise ValueError(f"Unknown evaluation split: {split!r}")

    _, dev = load_math_splits(manifest_path)
    if split == "dev":
        return dev
    seed = json.loads(manifest_path.read_text())["seed"]
    subset_ids = set(select_dev_ids(dev, DEV_SUBSET_SIZE, seed))
    return [problem for problem in dev if problem.problem_id in subset_ids]
