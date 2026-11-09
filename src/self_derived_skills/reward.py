"""Binary MATH reward: does the final boxed answer equal the gold answer."""

from collections.abc import Callable, Sequence
from typing import Any

from math_verify import LatexExtractionConfig, parse, verify

BOXED_PREFIX = "\\boxed{"

# LaTeX-only extraction is deliberate. math_verify's default config also tries plain-expression
# parsing, which passes text to SymPy's eval-based parse_expr, and completions are untrusted.
EXTRACTION_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False)
]


def last_boxed_content(text: str) -> str | None:
    """Return the contents of the last boxed expression, or None if absent or unclosed."""
    start = text.rfind(BOXED_PREFIX)
    if start == -1:
        return None

    content_start = start + len(BOXED_PREFIX)
    index = content_start
    depth = 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            # Skip the escaped character so that \{ and \} do not change the depth.
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
        index += 1
    return None


def parse_answer(answer: str) -> list:
    """Parse answer LaTeX into the objects that verify compares, empty if unparseable."""
    return parse(f"\\boxed{{{answer}}}", extraction_config=EXTRACTION_CONFIG)


def math_reward(completion: str, gold_answer: str) -> float:
    """Return 1.0 if the final boxed answer in the completion equals the gold answer, else 0.0.

    Must run in the main thread, because math_verify enforces its timeouts with signal.alarm.
    """
    gold = parse_answer(gold_answer)
    if not gold:
        raise ValueError(f"Gold answer did not parse: {gold_answer!r}")

    content = last_boxed_content(completion)
    if content is None:
        return 0.0
    predicted = parse_answer(content)
    return float(bool(predicted) and verify(gold, predicted))


def summarize_groups(rewards: Sequence[float], num_generations: int) -> dict[str, int]:
    """Count rollout groups that all fail, all pass, or mix the two."""
    if len(rewards) % num_generations != 0:
        raise ValueError(f"{len(rewards)} rewards do not form groups of {num_generations}")

    counts = {"all_fail": 0, "all_pass": 0, "mixed": 0}
    for start in range(0, len(rewards), num_generations):
        passes = sum(rewards[start : start + num_generations])
        if passes == 0:
            counts["all_fail"] += 1
        elif passes == num_generations:
            counts["all_pass"] += 1
        else:
            counts["mixed"] += 1
    return counts


def make_grpo_reward(
    num_generations: int, initial_sampled_tokens: int = 0
) -> Callable[..., list[float]]:
    """Build the reward function for TRL, which also logs group contrast and sampled tokens."""
    sampled_tokens = initial_sampled_tokens

    def math_accuracy(
        completions: list[list[dict[str, str]]],
        answer: list[str],
        problem_id: list[str],
        completion_ids: list[list[int]],
        log_metric: Callable[[str, float], None],
        **_: Any,
    ) -> list[float]:
        """Score each completion and log how many groups give GRPO a learning signal."""
        nonlocal sampled_tokens
        for start in range(0, len(problem_id), num_generations):
            if len(set(problem_id[start : start + num_generations])) != 1:
                raise ValueError("Completions are not in consecutive groups per problem")

        rewards = [
            math_reward(completion[0]["content"], gold)
            for completion, gold in zip(completions, answer, strict=True)
        ]

        counts = summarize_groups(rewards, num_generations)
        num_groups = sum(counts.values())
        for name, count in counts.items():
            log_metric(f"groups/{name}_frac", count / num_groups)
        sampled_tokens += sum(len(ids) for ids in completion_ids)
        log_metric("tokens/sampled_total", sampled_tokens)
        return rewards

    return math_accuracy
