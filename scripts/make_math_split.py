"""Write the manifest that fixes the MATH dev set for every run."""

import json
from pathlib import Path

from self_derived_skills.data import MATH_REVISION, load_math_500, load_math_train, select_dev_ids

MANIFEST_PATH = Path("configs/math_split.json")
N_DEV = 1000
SEED = 230


def main() -> None:
    """Select the dev set, confirm it is separate from MATH-500, and write the manifest."""
    problems = load_math_train()
    test_ids = {problem.problem_id for problem in load_math_500()}
    overlap = test_ids & {problem.problem_id for problem in problems}
    if overlap:
        raise ValueError(f"{len(overlap)} MATH-500 problems also appear in MATH train")

    manifest = {
        "revision": MATH_REVISION,
        "seed": SEED,
        "dev_ids": select_dev_ids(problems, N_DEV, SEED),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"Wrote {N_DEV} dev IDs from {len(problems)} problems to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
