"""Tests for final-answer extraction and the binary MATH reward."""

import pytest

from self_derived_skills.reward import last_boxed_content, math_reward


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no answer here", None),
        (r"so \boxed{5}", "5"),
        (r"first \boxed{1}, finally \boxed{2}", "2"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{\{1,2\}}", r"\{1,2\}"),
        (r"\boxed{5", None),
        (r"\boxed{1} and then \boxed{2", None),
    ],
)
def test_last_boxed_content(text, expected):
    """The final boxed expression is extracted, and an unclosed one counts as no answer."""
    assert last_boxed_content(text) == expected


@pytest.mark.parametrize(
    ("gold", "predicted"),
    [
        ("64", r"64\text{ square feet}"),
        (r"\frac{7}{25}", "0.28"),
        (r"\frac{1}{2}", r"\dfrac{1}{2}"),
        (r"\begin{pmatrix} 4 \\ -3 \end{pmatrix}", r"\begin{pmatrix} 4 \\ -3 \end{pmatrix}"),
        (r"\text{even}", r"\text{even}"),
        ("x^2+2x+1", "(x+1)^2"),
        (r"(-\infty,3)", r"(-\infty,3)"),
        ("1,2", "2,1"),
        (r"2\sqrt{3}", r"\sqrt{12}"),
        ("90^\\circ", "90"),
    ],
)
def test_equivalent_answers_score_one(gold, predicted):
    """Equivalent answers earn full reward across the formats MATH uses."""
    assert math_reward(rf"The answer is \boxed{{{predicted}}}.", gold) == 1.0


@pytest.mark.parametrize(
    ("gold", "predicted"),
    [
        ("3", "4"),
        (r"\frac{1}{2}", r"\frac{1}{3}"),
        (r"\begin{pmatrix} 4 \\ -3 \end{pmatrix}", r"\begin{pmatrix} 4 \\ 3 \end{pmatrix}"),
        (r"\text{even}", r"\text{odd}"),
        ("[2,5]", "(2,5)"),
        ("5", "5,6"),
    ],
)
def test_wrong_answers_score_zero(gold, predicted):
    """Non-equivalent answers earn no reward."""
    assert math_reward(rf"The answer is \boxed{{{predicted}}}.", gold) == 0.0


@pytest.mark.parametrize(
    "completion",
    [
        "The answer is 5.",
        r"\boxed{}",
        r"I think \boxed{5} at first, but the answer is \boxed{6}.",
        r"Truncated after \boxed{5} and then \boxed{",
    ],
)
def test_only_a_complete_final_boxed_answer_counts(completion):
    """Missing, empty, superseded, and truncated answers all score zero."""
    assert math_reward(completion, "5") == 0.0


def test_unparseable_gold_answer_raises():
    """A gold answer that cannot be parsed is a data error, not a zero reward."""
    with pytest.raises(ValueError, match="Gold answer did not parse"):
        math_reward(r"\boxed{5}", "")


def test_completion_never_reaches_expression_eval(monkeypatch):
    """Model text must not reach math_verify's eval-based expression parser."""
    calls = []
    monkeypatch.setattr("math_verify.parser.parse_expr_cached", calls.append)
    completion = r"\boxed{__import__('os').system('true')} or 1+1 = 2"
    assert math_reward(completion, "2") == 0.0
    assert calls == []
