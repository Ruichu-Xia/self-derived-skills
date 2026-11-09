"""Write skills from the untrained model's rollouts, filter them, and build the global bank."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers.utils.import_utils import is_flash_linear_attention_available

from self_derived_skills.data import load_math_splits
from self_derived_skills.evaluate import generate_completions
from self_derived_skills.metrics import load_results
from self_derived_skills.prompts import build_skill_messages
from self_derived_skills.run_record import write_run_record
from self_derived_skills.skill_bank import select_central_skills
from self_derived_skills.skills import rejection_reason, select_attempts
from self_derived_skills.train import load_model_and_tokenizer


def main() -> None:
    """Write every candidate skill and the global bank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True, help="rollouts.jsonl to read")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    bank_path = args.output_dir / "global_bank.json"
    if bank_path.exists():
        print(f"Bank already built: {bank_path}")
        return

    config = yaml.safe_load(args.config.read_text())
    skills_config = config["skills"]
    use_gpu = torch.cuda.is_available()
    if use_gpu and not is_flash_linear_attention_available():
        raise RuntimeError("flash-linear-attention is missing, so writing would be very slow")
    device = "cuda" if use_gpu else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_record(args.output_dir, config, device_name)

    train_problems, _ = load_math_splits(Path(config["split_manifest"]))
    problems_by_id = {problem.problem_id: problem for problem in train_problems}
    records = load_results(args.rollouts)
    sources, messages = [], []
    for record in records:
        attempts = select_attempts(record["rollouts"])
        if attempts is not None:
            problem = problems_by_id[record["problem_id"]]
            sources.append(problem)
            messages.append(build_skill_messages(problem.problem, *attempts))
    print(f"Writing skills for {len(messages)} of {len(records)} problems", flush=True)

    model, tokenizer = load_model_and_tokenizer(config["model"])
    model.to(device)
    model.eval()
    started = perf_counter()
    written = []
    for start in range(0, len(messages), args.batch_size):
        written += generate_completions(
            model,
            tokenizer,
            messages[start : start + args.batch_size],
            config["trainer"]["chat_template_kwargs"],
            skills_config["max_skill_tokens"],
            do_sample=False,
        )
    writing_seconds = perf_counter() - started

    candidates = []
    for problem, completion in zip(sources, written, strict=True):
        skill = completion["completion"].strip()
        candidates.append(
            {
                "problem_id": problem.problem_id,
                "level": problem.level,
                "subject": problem.subject,
                "skill": skill,
                "completion_tokens": completion["completion_tokens"],
                "rejected": rejection_reason(
                    skill, completion["truncated"], problem.problem, problem.answer
                ),
            }
        )
    lines = "".join(json.dumps(candidate) + "\n" for candidate in candidates)
    (args.output_dir / "candidates.jsonl").write_text(lines)
    kept = [candidate for candidate in candidates if candidate["rejected"] is None]

    embedder = SentenceTransformer(
        skills_config["embedding_model"],
        revision=skills_config["embedding_revision"],
        device=device,
    )
    embeddings = embedder.encode([c["skill"] for c in kept], normalize_embeddings=True)
    chosen = select_central_skills(
        embeddings, skills_config["bank_size"], config["trainer"]["seed"]
    )
    entries = [{**kept[index], "cluster_size": size} for index, size in chosen]
    skills = [entry["skill"] for entry in entries]

    bank = {
        "skills": skills,
        "entries": entries,
        "num_problems": len(records),
        "num_candidates": len(candidates),
        "num_kept": len(kept),
        "rejected": Counter(c["rejected"] for c in candidates if c["rejected"] is not None),
        "bank_subjects": Counter(entry["subject"] for entry in entries),
        "bank_tokens": len(tokenizer("\n".join(skills))["input_ids"]),
        "writing_tokens": sum(c["completion_tokens"] for c in candidates),
        "writing_seconds": writing_seconds,
    }
    bank_path.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n")
    print(json.dumps(bank, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
