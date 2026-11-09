"""Prompt format shared by training and evaluation."""

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_math_messages(problem: str) -> list[dict[str, str]]:
    """Return the chat messages that ask the model to solve a MATH problem."""
    return [{"role": "user", "content": f"{problem}\n\n{MATH_INSTRUCTION}"}]
