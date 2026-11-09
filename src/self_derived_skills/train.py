"""Load the model and build the GRPO trainer."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from self_derived_skills.data import MathProblem
from self_derived_skills.prompts import build_math_messages
from self_derived_skills.reward import make_grpo_reward


def load_model_and_tokenizer(model_config: dict[str, str]) -> tuple[Any, Any]:
    """Load the text-only model and its tokenizer at the pinned revision."""
    name, revision = model_config["name"], model_config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=model_config["dtype"]
    )
    return model, tokenizer


def make_train_dataset(problems: Sequence[MathProblem]) -> Dataset:
    """Build the dataset whose columns TRL and the reward function read."""
    rows = [
        {
            "prompt": build_math_messages(problem.problem),
            "answer": problem.answer,
            "problem_id": problem.problem_id,
        }
        for problem in problems
    ]
    return Dataset.from_list(rows)


def build_trainer(
    config: dict[str, Any],
    problems: Sequence[MathProblem],
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    max_steps: int,
) -> GRPOTrainer:
    """Build a LoRA GRPO trainer from a loaded config."""
    args = GRPOConfig(output_dir=str(output_dir), max_steps=max_steps, **config["trainer"])
    return GRPOTrainer(
        model=model,
        args=args,
        train_dataset=make_train_dataset(problems),
        processing_class=tokenizer,
        reward_funcs=make_grpo_reward(args.num_generations),
        peft_config=LoraConfig(**config["lora"]),
    )
