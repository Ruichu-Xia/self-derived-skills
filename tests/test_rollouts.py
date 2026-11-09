"""Tests for predicting the problems GRPO samples and for collecting rollouts."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from transformers import BatchEncoding  # noqa: E402

from self_derived_skills.data import MathProblem  # noqa: E402
from self_derived_skills.metrics import load_results  # noqa: E402
from self_derived_skills.rollouts import (  # noqa: E402
    collect_rollouts,
    grpo_problem_order,
    sample_rollouts,
)

PAD, CORRECT, WRONG, EOS = 0, 1, 2, 9
TRAINER_CONFIG = {
    "chat_template_kwargs": {},
    "max_completion_length": 3,
    "num_generations": 2,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 230,
}


class FakeTokenizer:
    """Stand in for a tokenizer, decoding each completion from its first token."""

    pad_token_id = PAD
    eos_token_id = EOS

    def __init__(self):
        """Start with no recorded prompts."""
        self.prompts = []

    def apply_chat_template(self, messages, **_):
        """Record and return the user message as the prompt."""
        self.prompts.append(messages[0]["content"])
        return messages[0]["content"]

    def __call__(self, prompts, **_):
        """Encode every prompt as two placeholder tokens."""
        return BatchEncoding({"input_ids": torch.ones((len(prompts), 2), dtype=torch.long)})

    def decode(self, ids, **_):
        """Return a boxed answer that matches the gold answer only for the CORRECT token."""
        return r"\boxed{5}" if ids[0] == CORRECT else r"\boxed{4}"


class FakeModel:
    """Stand in for a model, returning fixed completions and recording generate's arguments."""

    device = "cpu"

    def __init__(self, completions):
        """Store the completion token rows to return."""
        self.completions = completions
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        """Return the expanded prompts followed by the fixed completions."""
        self.kwargs = kwargs
        prompts = input_ids.repeat_interleave(kwargs["num_return_sequences"], dim=0)
        return torch.cat([prompts, torch.tensor(self.completions[: len(prompts)])], dim=1)


def make_problem(name):
    """Build a problem whose gold answer is 5."""
    return MathProblem(
        problem_id=name, problem=name, solution="", answer="5", level=1, subject="algebra"
    )


def test_a_longer_run_extends_the_same_problem_order():
    """Skills built for a short run stay valid if the run is extended."""
    short = grpo_problem_order(100, 8, 3, seed=230)
    long = grpo_problem_order(100, 8, 5, seed=230)
    assert long[:24] == short
    assert len(set(long)) == 40


def test_problem_order_rejects_more_than_one_pass():
    """The order is only predicted within the first pass over the data."""
    with pytest.raises(ValueError, match="more than one pass"):
        grpo_problem_order(100, 8, 13, seed=230)


def test_sample_rollouts_groups_scores_and_trims_completions():
    """Consecutive completions form each problem's group, cut at the end-of-text token."""
    model = FakeModel(
        [
            [CORRECT, EOS, PAD],
            [WRONG, WRONG, WRONG],
            [WRONG, EOS, PAD],
            [CORRECT, EOS, PAD],
        ]
    )
    first, second = sample_rollouts(
        model, FakeTokenizer(), [make_problem("a"), make_problem("b")], TRAINER_CONFIG
    )
    assert first["problem_id"] == "a"
    assert [rollout["correct"] for rollout in first["rollouts"]] == [True, False]
    assert [rollout["truncated"] for rollout in first["rollouts"]] == [False, True]
    assert [rollout["completion_tokens"] for rollout in first["rollouts"]] == [1, 3]
    assert [rollout["correct"] for rollout in second["rollouts"]] == [False, True]
    assert model.kwargs["do_sample"] is True
    assert model.kwargs["top_k"] == 0


def test_sample_rollouts_puts_skills_in_the_prompt():
    """The pilot's with-skills condition samples from prompts that contain the skills."""
    tokenizer = FakeTokenizer()
    model = FakeModel([[CORRECT, EOS, PAD]] * 2)
    sample_rollouts(model, tokenizer, [make_problem("a")], TRAINER_CONFIG, ["Check your work."])
    assert "- Check your work." in tokenizer.prompts[0]


def test_collect_rollouts_saves_every_problem_across_batches(tmp_path):
    """Each batch is written out, with the time spent on each problem."""
    path = tmp_path / "rollouts.jsonl"
    problems = [make_problem(name) for name in "abc"]
    model = FakeModel([[CORRECT, EOS, PAD]] * 4)
    collect_rollouts(model, FakeTokenizer(), problems, 0, path, TRAINER_CONFIG, batch_size=2)
    records = load_results(path)
    assert [record["problem_id"] for record in records] == ["a", "b", "c"]
    assert all(len(record["rollouts"]) == 2 and record["seconds"] >= 0 for record in records)
