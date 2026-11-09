"""Evaluate a model on MATH problems with one greedy attempt per problem."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from self_derived_skills.data import MathProblem
from self_derived_skills.metrics import append_results, load_results
from self_derived_skills.prompts import build_math_messages
from self_derived_skills.reward import math_reward


@torch.inference_mode()
def generate_and_score(
    model: Any,
    tokenizer: Any,
    problems: Sequence[MathProblem],
    template_kwargs: dict[str, Any],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate one greedy completion per problem and score it with the training reward."""
    prompts = [
        tokenizer.apply_chat_template(
            build_math_messages(problem.problem),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        for problem in problems
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    inputs = inputs.to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    results = []
    for problem, ids in zip(problems, completion_ids, strict=True):
        eos_positions = (ids == tokenizer.eos_token_id).nonzero()
        truncated = len(eos_positions) == 0
        length = len(ids) if truncated else int(eos_positions[0])
        completion = tokenizer.decode(ids[:length], skip_special_tokens=True)
        results.append(
            {
                "problem_id": problem.problem_id,
                "level": problem.level,
                "subject": problem.subject,
                "correct": math_reward(completion, problem.answer) == 1.0,
                "completion_tokens": length,
                "truncated": truncated,
                "completion": completion,
            }
        )
    return results


def evaluate_problems(
    model: Any,
    tokenizer: Any,
    problems: Sequence[MathProblem],
    results_path: Path,
    template_kwargs: dict[str, Any],
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Evaluate every problem, saving after each batch so that an interrupted run can resume."""
    completed = {result["problem_id"] for result in load_results(results_path)}
    remaining = [problem for problem in problems if problem.problem_id not in completed]
    print(f"{len(completed)} problems already done, {len(remaining)} remaining", flush=True)

    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        results = generate_and_score(model, tokenizer, batch, template_kwargs, max_new_tokens)
        append_results(results_path, results)
        print(f"Evaluated {start + len(batch)} of {len(remaining)}", flush=True)

    results = load_results(results_path)
    if {result["problem_id"] for result in results} != {p.problem_id for p in problems}:
        raise ValueError(f"{results_path} does not cover exactly the requested problems")
    return results
