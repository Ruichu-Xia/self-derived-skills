"""Prompt formats shared by training, evaluation, and skill writing."""

from collections.abc import Sequence

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SKILL_HEADER = "Here are some general tips that may help:"
# Adapted from NuRL's hint prompt (arXiv 2509.25666, Appendix D). "Problems like this" replaces
# "this problem" because hints are reused across problems, and the last sentence is added so that
# a bank of them fits in every training prompt.
SKILL_INSTRUCTION = (
    "Provide a hint to help a student learn the core concept (e.g. formula, lemma, or necessary "
    "knowledge) needed to solve problems like this. The hint should be concise, to the point, "
    "but high level. Do not include any detailed steps or calculations or the final answer. "
    "Use one sentence."
)


def build_math_messages(problem: str, skills: Sequence[str] = ()) -> list[dict[str, str]]:
    """Return the chat messages that ask the model to solve a MATH problem, with optional skills."""
    content = f"{problem}\n\n{MATH_INSTRUCTION}"
    if skills:
        tips = "\n".join(f"- {skill}" for skill in skills)
        content = f"{SKILL_HEADER}\n{tips}\n\n{content}"
    return [{"role": "user", "content": content}]


def build_skill_messages(problem: str, correct: str, incorrect: str | None) -> list[dict[str, str]]:
    """Return the chat messages that ask the model to write one skill from its own attempts."""
    parts = [f"Problem:\n{problem}", f"A correct attempt:\n{correct}"]
    if incorrect is not None:
        parts.append(f"An incorrect attempt:\n{incorrect}")
    parts.append(SKILL_INSTRUCTION)
    return [{"role": "user", "content": "\n\n".join(parts)}]
