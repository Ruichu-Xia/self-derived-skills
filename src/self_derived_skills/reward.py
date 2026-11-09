"""Binary MATH reward: does the final boxed answer equal the gold answer."""

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
