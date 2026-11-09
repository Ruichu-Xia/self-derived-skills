"""Check that the model and TRL GRPO work on this machine, and time GRPO steps."""

import argparse
import functools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import TrainerCallback
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

from self_derived_skills.data import load_math_splits
from self_derived_skills.prompts import build_math_messages
from self_derived_skills.run_record import write_run_record
from self_derived_skills.train import build_trainer, load_model_and_tokenizer

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
PADDING_PROBE_PROBLEM = "Find the area in square feet of a square with a perimeter of 32 feet."
PADDING_LENGTHS = (1, 16, 64)
GREEDY_TOKENS = 40
TOP_TOKENS = 50


class StepTimer(TrainerCallback):
    """Record the wall-clock duration of each optimizer step."""

    def __init__(self) -> None:
        """Start with no recorded steps."""
        self.started_at = 0.0
        self.step_seconds: list[float] = []

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Mark the start of an optimizer step."""
        self.started_at = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Record the duration of the finished optimizer step."""
        self.step_seconds.append(time.perf_counter() - self.started_at)


@torch.inference_mode()
def check_left_padding(model: Any, tokenizer: Any, device: str, template_kwargs: dict) -> dict:
    """Measure how much left padding changes next-token log-probs and greedy output."""
    messages = build_math_messages(PADDING_PROBE_PROBLEM)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    mask = torch.ones_like(input_ids)
    generate_kwargs = {
        "max_new_tokens": GREEDY_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    base_logps = model(input_ids=input_ids, attention_mask=mask).logits[0, -1].log_softmax(-1)
    top_tokens = base_logps.topk(TOP_TOKENS).indices
    base_output = model.generate(input_ids=input_ids, attention_mask=mask, **generate_kwargs)
    base_new_tokens = base_output[0, input_ids.shape[1] :]

    results = {}
    for length in PADDING_LENGTHS:
        pad = torch.full((1, length), tokenizer.pad_token_id, device=device)
        padded_ids = torch.cat([pad, input_ids], dim=1)
        padded_mask = torch.cat([torch.zeros_like(pad), mask], dim=1)

        logps = (
            model(input_ids=padded_ids, attention_mask=padded_mask).logits[0, -1].log_softmax(-1)
        )
        output = model.generate(input_ids=padded_ids, attention_mask=padded_mask, **generate_kwargs)
        new_tokens = output[0, padded_ids.shape[1] :]
        results[str(length)] = {
            "max_abs_logprob_diff_top_tokens": (logps - base_logps)[top_tokens].abs().max().item(),
            "greedy_output_matches": torch.equal(new_tokens, base_new_tokens),
        }
    return results


def timed_reward(reward_fn: Any, timer: StepTimer, samples: dict[str, list]) -> Any:
    """Wrap a reward function to time generation and scoring and keep one completion per call."""

    @functools.wraps(reward_fn)
    def wrapper(*args: Any, **kwargs: Any) -> list[float]:
        """Call the reward function and record its timing and a sample completion."""
        started_at = time.perf_counter()
        samples["generation_seconds"].append(started_at - timer.started_at)
        rewards = reward_fn(*args, **kwargs)
        samples["reward_seconds"].append(time.perf_counter() - started_at)
        samples["completion_lengths"].extend(len(ids) for ids in kwargs["completion_ids"])
        samples["completions"].append(kwargs["completions"][0][0]["content"])
        return rewards

    return wrapper


def main() -> None:
    """Run the checks and write a summary to the output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    use_cpu = config["trainer"].get("use_cpu", False) or not torch.cuda.is_available()
    device = "cpu" if use_cpu else "cuda"
    device_name = "cpu" if use_cpu else torch.cuda.get_device_name(0)
    write_run_record(args.output_dir, config, device_name)
    print(f"Device: {device_name}", flush=True)

    kernels = {
        "flash_linear_attention": is_flash_linear_attention_available(),
        "causal_conv1d": is_causal_conv1d_available(),
    }
    if not use_cpu and not kernels["flash_linear_attention"]:
        raise RuntimeError("flash-linear-attention is missing, so timings would mislead")

    train_problems, _ = load_math_splits(Path(config["split_manifest"]))
    model, tokenizer = load_model_and_tokenizer(config["model"])
    model.to(device)
    template_kwargs = config["trainer"]["chat_template_kwargs"]

    rendered = tokenizer.apply_chat_template(
        build_math_messages("Find x."),
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not rendered.endswith(EMPTY_THINK_BLOCK):
        raise ValueError(f"Thinking is not disabled in the prompt: {rendered!r}")

    summary: dict[str, Any] = {
        "device": device_name,
        "kernels": kernels,
        "model_class": type(model).__name__,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "left_padding": check_left_padding(model, tokenizer, device, template_kwargs),
    }
    print(json.dumps(summary["left_padding"], indent=2), flush=True)

    trainer = build_trainer(
        config, train_problems, model, tokenizer, args.output_dir / "trainer", args.max_steps
    )
    timer = StepTimer()
    samples: dict[str, list] = {
        "generation_seconds": [],
        "reward_seconds": [],
        "completion_lengths": [],
        "completions": [],
    }
    trainer.add_callback(timer)
    trainer.reward_funcs[0] = timed_reward(trainer.reward_funcs[0], timer, samples)
    if torch.cuda.is_available() and not use_cpu:
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))

    # The first step includes one-time setup, so timing estimates exclude it.
    steady_seconds = timer.step_seconds[1:] or timer.step_seconds
    median_step = statistics.median(steady_seconds)
    lengths = samples["completion_lengths"]
    summary["grpo"] = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "step_seconds": timer.step_seconds,
        "median_step_seconds": median_step,
        "steps_per_hour": 3600 / median_step,
        "generation_seconds": samples["generation_seconds"],
        "reward_seconds": samples["reward_seconds"],
        "mean_completion_tokens": statistics.mean(lengths),
        "max_completion_tokens": max(lengths),
        "log_history": trainer.state.log_history,
        "peak_memory_gb": 0.0 if use_cpu else torch.cuda.max_memory_allocated() / 2**30,
        "adapter_files": sorted(path.name for path in adapter_dir.iterdir()),
        "first_completion": samples["completions"][0],
    }
    summary_path = args.output_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"PROBE_SUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()
