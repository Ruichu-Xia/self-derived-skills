"""Validate the reward on MATH reference solutions, which should score 1.0."""

import random
from collections import Counter

from self_derived_skills.data import load_math_train
from self_derived_skills.reward import math_reward, parse_answer

SEED = 230
MAX_EXAMPLES = 25


def main() -> None:
    """Report gold parse failures, reference-solution scores, and false matches."""
    problems = load_math_train()
    print(f"Problems: {len(problems)}")
    print(f"By level: {sorted(Counter(problem.level for problem in problems).items())}")
    print(f"By subject: {sorted(Counter(problem.subject for problem in problems).items())}")

    parseable = [problem for problem in problems if parse_answer(problem.answer)]
    unparseable = [problem.answer for problem in problems if not parse_answer(problem.answer)]
    print(f"Gold answers that do not parse: {len(unparseable)}")
    print(f"  Examples: {unparseable[:MAX_EXAMPLES]}")

    failures = [p.answer for p in parseable if math_reward(p.solution, p.answer) != 1.0]
    rate = 1 - len(failures) / len(parseable)
    print(f"Reference solutions scoring 1.0: {rate:.4%} of {len(parseable)}")
    print(f"  Failures: {failures[:MAX_EXAMPLES]}")

    # Pair each solution with another problem's answer. Accepted pairs whose answer text
    # differs show how permissive the grader is, and they should be rare and defensible.
    shuffled = random.Random(SEED).sample(parseable, len(parseable))
    pairs = zip(shuffled, shuffled[1:], strict=False)
    false_matches = [
        (a.answer, b.answer)
        for a, b in pairs
        if a.answer.strip() != b.answer.strip() and math_reward(a.solution, b.answer) == 1.0
    ]
    print(f"Accepted pairs with different answer text: {len(false_matches)}")
    print(f"  Examples: {false_matches[:MAX_EXAMPLES]}")


if __name__ == "__main__":
    main()
