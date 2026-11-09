"""Sample rollouts from the untrained model, which later become the source of skills."""

from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any

from transformers import set_seed
from trl.trainer.utils import RepeatSampler

from self_derived_skills.data import MathProblem
from self_derived_skills.evaluate import generate_completions
from self_derived_skills.metrics import append_results
from self_derived_skills.prompts import build_math_messages
from self_derived_skills.reward import math_reward


def grpo_problem_order(
    num_problems: int, problems_per_step: int, num_steps: int, seed: int
) -> list[int]:
    """Return the dataset indices that GRPO samples over num_steps, in order.

    This asks TRL's own sampler. Its order depends only on the dataset size and seed, so a longer
    run extends a shorter one, and adding skills to the prompts does not change the problems drawn.
    """
    needed = problems_per_step * num_steps
    sampler = RepeatSampler(
        range(num_problems), mini_repeat_count=1, batch_size=problems_per_step, seed=seed
    )
    order = list(islice(sampler, needed))
    if len(order) != needed:
        raise ValueError(f"{num_steps} steps need more than one pass over {num_problems} problems")
    return order


def sample_rollouts(
    model: Any,
    tokenizer: Any,
    problems: Sequence[MathProblem],
    trainer_config: dict[str, Any],
    skills: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Sample a group of scored completions per problem, with GRPO's sampling settings."""
    num_samples = trainer_config["num_generations"]
    completions = generate_completions(
        model,
        tokenizer,
        [build_math_messages(problem.problem, skills) for problem in problems],
        trainer_config["chat_template_kwargs"],
        trainer_config["max_completion_length"],
        do_sample=True,
        num_return_sequences=num_samples,
        temperature=trainer_config["temperature"],
        top_p=trainer_config["top_p"],
        top_k=trainer_config["top_k"],
    )
    if len(completions) != len(problems) * num_samples:
        raise ValueError(f"Expected {num_samples} completions for each of {len(problems)} problems")

    records = []
    for index, problem in enumerate(problems):
        group = completions[index * num_samples : (index + 1) * num_samples]
        rollouts = [
            {**item, "correct": math_reward(item["completion"], problem.answer) == 1.0}
            for item in group
        ]
        records.append(
            {
                "problem_id": problem.problem_id,
                "level": problem.level,
                "subject": problem.subject,
                "rollouts": rollouts,
            }
        )
    return records


def collect_rollouts(
    model: Any,
    tokenizer: Any,
    problems: Sequence[MathProblem],
    first_index: int,
    path: Path,
    trainer_config: dict[str, Any],
    batch_size: int,
    skills: Sequence[str] = (),
) -> None:
    """Sample rollouts for each problem, saving after each batch so that a run can resume.

    Each batch is seeded from its position in the full problem order, given by first_index for the
    first problem here, so that a resumed run stays reproducible.
    """
    for start in range(0, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        set_seed(trainer_config["seed"] + first_index + start)
        started = perf_counter()
        records = sample_rollouts(model, tokenizer, batch, trainer_config, skills)
        seconds = (perf_counter() - started) / len(batch)
        append_results(path, [{**record, "seconds": seconds} for record in records])
        print(f"Sampled {first_index + start + len(batch)} problems", flush=True)
