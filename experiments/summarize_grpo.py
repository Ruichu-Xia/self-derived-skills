"""Write the GRPO learning curve by joining the training log with dev evaluations."""

import argparse
import json
from pathlib import Path

from self_derived_skills.metrics import build_learning_curve


def main() -> None:
    """Build the learning curve, save it, and print it as a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    log_history = json.loads((args.run_dir / "train_log.json").read_text())
    evaluations = {
        int(path.parent.name.removeprefix("step-")): json.loads(path.read_text())
        for path in args.run_dir.glob("evals/step-*/metrics.json")
    }
    curve = build_learning_curve(log_history, evaluations)
    (args.run_dir / "learning_curve.json").write_text(json.dumps(curve, indent=2) + "\n")

    print(f"{'step':>5} {'n':>5} {'accuracy':>9} {'std err':>8} {'tokens':>10} {'train s':>9}")
    for row in curve:
        print(
            f"{row['step']:>5} {row['n']:>5} {row['accuracy']:>9.3f} {row['standard_error']:>8.3f} "
            f"{row['sampled_tokens']:>10.0f} {row['training_seconds']:>9.0f}"
        )


if __name__ == "__main__":
    main()
