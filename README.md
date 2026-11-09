# Self-Derived Skills

Code for the Stanford CS230 project *Learning Better with Self-Derived Skills: Reinforcement
Learning for Small Language Models*. The project tests whether conditioning GRPO on procedural
skills written by the model itself lets Qwen3.5-0.8B reach a given held-out performance with fewer
sampled tokens and less compute.

The repository currently contains the MATH data split, prompt format, binary reward, and the code
and Colab notebooks for the two baselines, the untrained model and standard GRPO, plus skill
construction and GRPO with the global skill bank. Clustered retrieval, per-problem conditioning, and
the shuffled control are still to come. See [docs/DESIGN.md](docs/DESIGN.md) for the design and the
reasoning behind it.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run pytest
```

To enable the lint and formatting hooks, run `uv run pre-commit install`. Add `--extra train` to
`uv sync` to install the training stack.

## Running on Colab

Each notebook runs from start to finish with **Runtime > Run all**, and resumes from where it
stopped if the session disconnects. Results are saved to Google Drive by default.

| Notebook | Produces |
| :--- | :--- |
| `notebooks/baseline_inference.ipynb` | Untrained-model accuracy on MATH dev and MATH-500 |
| `notebooks/baseline_grpo.ipynb` | Standard GRPO training, learning curve, and final dev accuracy |
| `notebooks/skill_pipeline.ipynb` | Rollouts, the global skill bank, and the pilot that gates RL |
| `notebooks/skill_grpo.ipynb` | GRPO with the global bank in training prompts, evaluated without it |
| `notebooks/colab_runner.ipynb` | A short probe of model checks and GRPO step time |

## Data checks

These scripts need network access to download the pinned datasets.

```bash
uv run python scripts/make_math_split.py    # rewrites configs/math_split.json
uv run python scripts/check_math_reward.py  # validates the reward on reference solutions
```

## Datasets

Dataset text is never committed. `configs/math_split.json` stores only the hashed problem IDs that
define the dev set.

| Dataset | Source | Revision | Use | License |
| :--- | :--- | :--- | :--- | :--- |
| MATH train | `EleutherAI/hendrycks_math` | `21a5633` | RL training and dev set | Disputed, see below |
| MATH-500 | `HuggingFaceH4/MATH-500` | `6e4ed1a` | Final evaluation | None on the card, source repository is MIT |

The MATH code repository is MIT-licensed, but the original `hendrycks/competition_math` dataset was
disabled on Hugging Face after a January 2025 takedown notice from Art of Problem Solving. A mirror
remaining online does not settle the dataset's legal status.

## Layout

```text
configs/      Split manifest and run settings
docs/         Design document
experiments/  Entry points for runs
notebooks/    Colab runner
scripts/      Data preparation and validation
src/          The self_derived_skills package
tests/        Unit tests, which need no network or GPU
```
