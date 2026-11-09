"""Sample rollouts from the untrained model on the problems that GRPO will train on."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers.utils.import_utils import is_flash_linear_attention_available

from self_derived_skills.data import load_math_splits
from self_derived_skills.metrics import load_results, summarize_rollouts
from self_derived_skills.rollouts import collect_rollouts, grpo_problem_order
from self_derived_skills.run_record import write_run_record
from self_derived_skills.skills import load_skill_bank
from self_derived_skills.train import load_model_and_tokenizer


def main() -> None:
    """Sample any rollouts still missing, then write a summary of all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="GRPO steps to cover")
    parser.add_argument("--batch-size", type=int, default=4, help="Problems per generate call")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N problems")
    parser.add_argument("--limit", type=int, help="Sample only the first N problems after --skip")
    parser.add_argument("--skill-bank", type=Path, help="Skills to add to the prompts, if any")
    args = parser.parse_args()

    rollouts_path = args.output_dir / "rollouts.jsonl"
    summary_path = args.output_dir / "summary.json"
    # A summary from a shorter run must not stand in for this one if it fails partway.
    summary_path.unlink(missing_ok=True)

    config = yaml.safe_load(args.config.read_text())
    trainer_config = config["trainer"]
    problems_per_step = (
        trainer_config["per_device_train_batch_size"]
        * trainer_config["gradient_accumulation_steps"]
        // trainer_config["num_generations"]
    )
    train_problems, _ = load_math_splits(Path(config["split_manifest"]))
    order = grpo_problem_order(
        len(train_problems), problems_per_step, args.max_steps, trainer_config["seed"]
    )
    problems = [train_problems[index] for index in order][args.skip :][: args.limit]
    skills = load_skill_bank(args.skill_bank) if args.skill_bank else []

    done = {record["problem_id"] for record in load_results(rollouts_path)}
    remaining = [problem for problem in problems if problem.problem_id not in done]
    first_index = len(problems) - len(remaining)
    print(f"{first_index} problems already done, {len(remaining)} remaining", flush=True)

    if remaining:
        use_gpu = torch.cuda.is_available()
        if use_gpu and not is_flash_linear_attention_available():
            raise RuntimeError("flash-linear-attention is missing, so sampling would be very slow")
        device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
        run_settings = {
            "max_steps": args.max_steps,
            "skip": args.skip,
            "limit": args.limit,
            "skills": skills,
        }
        write_run_record(args.output_dir, {**config, "rollouts": run_settings}, device_name)

        model, tokenizer = load_model_and_tokenizer(config["model"])
        model.to("cuda" if use_gpu else "cpu")
        model.eval()
        collect_rollouts(
            model,
            tokenizer,
            remaining,
            first_index,
            rollouts_path,
            trainer_config,
            args.batch_size,
            skills,
        )

    wanted = {problem.problem_id for problem in problems}
    records = [record for record in load_results(rollouts_path) if record["problem_id"] in wanted]
    if {record["problem_id"] for record in records} != wanted:
        raise ValueError(f"{rollouts_path} does not cover exactly the requested problems")

    summary = summarize_rollouts(records)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
