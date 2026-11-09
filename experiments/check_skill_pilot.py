"""Compare the untrained model with and without skills, and apply the pilot gate."""

import argparse
import json
from pathlib import Path

from self_derived_skills.metrics import compare_skill_pilot


def main() -> None:
    """Write the comparison of the two pilot conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-skills", type=Path, required=True, help="Directory with summary.json")
    parser.add_argument(
        "--with-skills", type=Path, required=True, help="Directory with summary.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    without = json.loads((args.no_skills / "summary.json").read_text())
    with_skills = json.loads((args.with_skills / "summary.json").read_text())
    result = compare_skill_pilot(without, with_skills)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
