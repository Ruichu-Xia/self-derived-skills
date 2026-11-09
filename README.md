# Self-Derived Skills

Code for the Stanford CS230 project *Learning Better with Self-Derived Skills: Reinforcement
Learning for Small Language Models*. The project tests whether conditioning GRPO on procedural
skills written by the model itself lets Qwen3.5-0.8B reach a given held-out performance with fewer
sampled tokens and less compute.

The repository currently contains the MATH data split, prompt format, binary reward, GRPO trainer
setup, and a probe that checks the model and times training on a GPU. Evaluation code will be added
next. See [docs/DESIGN.md](docs/DESIGN.md) for the design and the reasoning behind it.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run pytest
```

To enable the lint and formatting hooks, run `uv run pre-commit install`. Add `--extra train` to
`uv sync` to install the training stack.

## Running on Colab

Open `notebooks/colab_runner.ipynb` in Colab and follow its first cell. Training needs a GPU. Run
records and adapters are saved to Google Drive.

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
