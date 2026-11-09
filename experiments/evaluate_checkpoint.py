"""Evaluate the base model or a LoRA adapter on a MATH split, with no skills in the prompt."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers.utils.import_utils import is_flash_linear_attention_available

from self_derived_skills.data import load_eval_problems
from self_derived_skills.evaluate import evaluate_problems
from self_derived_skills.metrics import summarize_results
from self_derived_skills.run_record import write_run_record
from self_derived_skills.train import load_model_and_tokenizer


def main() -> None:
    """Run the evaluation and write per-problem results and summary metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "dev-subset", "math500"], required=True)
    parser.add_argument("--adapter", type=Path, help="LoRA adapter; omit for the base model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N problems")
    args = parser.parse_args()

    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"Already evaluated: {metrics_path}")
        return

    config = yaml.safe_load(args.config.read_text())
    use_gpu = torch.cuda.is_available()
    if use_gpu and not is_flash_linear_attention_available():
        raise RuntimeError("flash-linear-attention is missing, so evaluation would be very slow")
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"

    run_settings = {"split": args.split, "adapter": str(args.adapter), "limit": args.limit}
    write_run_record(args.output_dir, {**config, "evaluation": run_settings}, device_name)

    problems = load_eval_problems(args.split, Path(config["split_manifest"]))[: args.limit]
    model, tokenizer = load_model_and_tokenizer(config["model"])
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to("cuda" if use_gpu else "cpu")
    model.eval()

    results = evaluate_problems(
        model,
        tokenizer,
        problems,
        args.output_dir / "results.jsonl",
        config["trainer"]["chat_template_kwargs"],
        config["trainer"]["max_completion_length"],
        args.batch_size,
    )
    metrics = {**run_settings, "device": device_name, **summarize_results(results)}
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
