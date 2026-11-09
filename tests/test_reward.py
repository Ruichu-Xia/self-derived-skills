"""Tests for final-answer extraction and the binary MATH reward."""

import pytest

from self_derived_skills.reward import (
    last_boxed_content,
    make_grpo_reward,
    math_reward,
    summarize_groups,
)


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


def test_summarize_groups_separates_all_fail_all_pass_and_mixed():
    """Each group is classified by whether its candidates disagree."""
    rewards = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert summarize_groups(rewards, 2) == {"all_fail": 2, "all_pass": 1, "mixed": 1}


def test_summarize_groups_rejects_incomplete_groups():
    """Rewards that cannot form whole groups indicate a batching bug."""
    with pytest.raises(ValueError, match="do not form groups"):
        summarize_groups([1.0, 0.0, 1.0], 2)


def call_grpo_reward(reward_fn, answers, problem_ids, token_ids, metrics):
    """Call a GRPO reward function the way TRL does, collecting its logged metrics."""
    completions = [[{"role": "assistant", "content": rf"\boxed{{{a}}}"}] for a in answers]
    return reward_fn(
        completions=completions,
        answer=["5"] * len(answers),
        problem_id=problem_ids,
        completion_ids=token_ids,
        log_metric=lambda name, value: metrics.__setitem__(name, value),
        trainer_state=None,
    )


def test_grpo_reward_scores_completions_and_logs_group_contrast():
    """The reward returns one value per completion and logs group shares."""
    metrics = {}
    rewards = call_grpo_reward(
        make_grpo_reward(2),
        answers=["5", "4", "1", "2"],
        problem_ids=["a", "a", "b", "b"],
        token_ids=[[1, 2], [3], [4, 5, 6], [7]],
        metrics=metrics,
    )
    assert rewards == [1.0, 0.0, 0.0, 0.0]
    assert metrics["groups/mixed_frac"] == 0.5
    assert metrics["groups/all_fail_frac"] == 0.5
    assert metrics["groups/all_pass_frac"] == 0.0
    assert metrics["tokens/sampled_total"] == 7


def test_grpo_reward_accumulates_sampled_tokens_across_calls():
    """The sampled-token count is cumulative, which the efficiency metric depends on."""
    reward_fn = make_grpo_reward(2)
    metrics = {}
    call_grpo_reward(reward_fn, ["5", "5"], ["a", "a"], [[1, 2], [3]], metrics)
    call_grpo_reward(reward_fn, ["5", "5"], ["b", "b"], [[4], [5]], metrics)
    assert metrics["tokens/sampled_total"] == 5


def test_grpo_reward_rejects_groups_that_mix_problems():
    """Misaligned groups would corrupt the contrast statistics, so they fail immediately."""
    with pytest.raises(ValueError, match="consecutive groups"):
        call_grpo_reward(make_grpo_reward(2), ["5", "5"], ["a", "b"], [[1], [2]], {})


def test_completion_never_reaches_expression_eval(monkeypatch):
    """Model text must not reach math_verify's eval-based expression parser."""
    calls = []
    monkeypatch.setattr("math_verify.parser.parse_expr_cached", calls.append)
    completion = r"\boxed{__import__('os').system('true')} or 1+1 = 2"
    assert math_reward(completion, "2") == 0.0
    assert calls == []
