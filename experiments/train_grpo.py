"""Train GRPO, with or without skills, resuming from the latest checkpoint if one exists."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.import_utils import is_flash_linear_attention_available

from self_derived_skills.data import load_math_splits
from self_derived_skills.run_record import write_run_record
from self_derived_skills.skills import load_skill_bank
from self_derived_skills.train import build_trainer, load_model_and_tokenizer


class TrainingLogWriter(TrainerCallback):
    """Rewrite the full training log after each logged step, so that it survives a disconnect."""

    def __init__(self, path: Path) -> None:
        """Store the log path."""
        self.path = path

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Write the training log."""
        self.path.write_text(json.dumps(state.log_history, indent=1) + "\n")


def sampled_tokens_in_checkpoint(checkpoint: str | None) -> int:
    """Return the cumulative sampled tokens recorded in a checkpoint, or 0 for a new run."""
    if checkpoint is None:
        return 0
    state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
    logged = [row for row in state["log_history"] if "tokens/sampled_total" in row]
    return int(logged[-1]["tokens/sampled_total"])


def main() -> None:
    """Train, then save the final adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--skill-bank", type=Path, help="Skills for training prompts, if any")
    args = parser.parse_args()

    final_adapter = args.output_dir / "final_adapter"
    if final_adapter.exists():
        print(f"Training is already complete: {final_adapter}")
        return

    config = yaml.safe_load(args.config.read_text())
    save_steps = config["trainer"]["save_steps"]
    if args.max_steps % save_steps != 0:
        raise ValueError(f"max_steps must be a multiple of save_steps ({save_steps})")

    use_gpu = torch.cuda.is_available()
    if use_gpu and not is_flash_linear_attention_available():
        raise RuntimeError("flash-linear-attention is missing, so training would be very slow")
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(checkpoint_dir))
    skills = load_skill_bank(args.skill_bank) if args.skill_bank else []
    run_settings = {"max_steps": args.max_steps, "skills": skills}
    write_run_record(args.output_dir, {**config, **run_settings}, device_name)
    print(f"Device: {device_name}. Resuming from: {checkpoint}", flush=True)

    # The seed must be set before the trainer creates the adapter's random initial weights.
    set_seed(config["trainer"]["seed"])
    train_problems, _ = load_math_splits(Path(config["split_manifest"]))
    model, tokenizer = load_model_and_tokenizer(config["model"])
    trainer = build_trainer(
        config,
        train_problems,
        model,
        tokenizer,
        checkpoint_dir,
        args.max_steps,
        initial_sampled_tokens=sampled_tokens_in_checkpoint(checkpoint),
        skills=skills,
    )
    trainer.add_callback(TrainingLogWriter(args.output_dir / "train_log.json"))
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(final_adapter))
    print(f"Saved final adapter to {final_adapter}")


if __name__ == "__main__":
    main()
