# Design

## Goal

Measure whether GRPO on Qwen3.5-0.8B learns more efficiently when its training prompts include
procedural skills that the model derived from its own rollouts. The primary evaluation omits
skills, so a gain counts only if the policy keeps it.

## Data

MATH train is loaded at a pinned revision. Of its 7,500 rows, 7,493 are usable:

| Removed | Rows | Reason |
| :--- | ---: | :--- |
| Unknown level | 2 | Labeled `Level ?`, so they cannot be stratified |
| Brace-free box | 2 | Written as `\boxed 2`, which the answer extractor does not read |
| Empty box | 2 | The solution contains `\boxed{}`, so there is no gold answer |
| Repeated problem | 1 | Must collapse so one problem cannot land in both splits |

The dev set is 1,000 problems, stratified by level and fixed in `configs/math_split.json` before
any skill extraction or training. The remaining 6,493 problems are for skill extraction and RL.

| Level | 1 | 2 | 3 | 4 | 5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Usable problems | 564 | 1,348 | 1,590 | 1,690 | 2,301 |
| Dev problems | 75 | 180 | 212 | 226 | 307 |

MATH-500 is reserved for final evaluation and shares no problem with train or dev after whitespace
normalization. The split script fails if that stops being true. Near-duplicates are not detected.

## Reward

The reward is 1.0 if the last boxed expression in a completion is equivalent to the gold answer
under `math_verify`, and 0.0 otherwise. It is strictly binary because GRPO normalizes rewards by
the group's standard deviation, which would inflate any small partial credit.

- **Only the final boxed answer counts.** A missing, empty, or unclosed final box scores 0.0. A
  completion cut off after a complete box is scored on that box, unless the cut falls inside a
  later box.
- **Extraction is LaTeX-only.** The default `math_verify` config also tries plain-expression
  parsing, which sends text to SymPy's eval-based `parse_expr`. Completions are untrusted.
- **The reward runs in the main thread.** `math_verify` enforces its timeouts with `signal.alarm`.
- **An unparseable gold answer raises an error.** It is a data bug and should not become a zero.

`scripts/check_math_reward.py` validates this on real data. All 7,493 reference solutions score
1.0. When 7,492 solutions are paired with another problem's answer, 2 pairs with different answer
text are accepted: `6\%` for `6` and `3:1` for `3`.

## Prompt

Training and evaluation share one zero-shot user message: the problem, followed by the instruction
to reason step by step and box the final answer.

## Model

Qwen3.5-0.8B is loaded with `AutoModelForCausalLM`, which gives the text-only model. Thinking is
disabled through the chat template, which prefills an empty think block.

Its linear-attention layers need the `flash-linear-attention` kernel on a GPU, because transformers
otherwise falls back to a far slower reference implementation. The probe stops if the kernel is
missing on a GPU, so that timings cannot silently describe the slow path. The optional
`causal-conv1d` kernel is not installed, because it must be compiled with `nvcc`.

## Training

The standard GRPO baseline uses TRL's `GRPOTrainer` with a LoRA adapter and the settings in
`configs/grpo_math.yaml`. TRL's default variant differs from the original GRPO: it normalizes the
loss by active tokens across the batch (`loss_type: dapo`) and has no KL penalty (`beta: 0`).

Training runs in float32 on every GPU, so that results do not depend on which GPU Colab assigns.
For bf16 inputs, TRL's gradient kernel needs a mixed-precision matrix product that PyTorch only
supports on GPUs with compute capability 8.0 or higher, which excludes Colab's T4.

The reward function also logs what TRL does not report directly: the share of groups that all
fail, all pass, or are mixed, and the cumulative number of sampled completion tokens. Only mixed
groups give GRPO a policy gradient.

Each step samples 8 problems with 4 completions each. The trainer saves a checkpoint every 20 steps,
and a restarted run resumes from the latest one. The sampled-token count resumes with it, so that
token budgets stay comparable across interrupted runs.

## Evaluation

Every checkpoint gets one greedy attempt per problem, with no skills in the prompt, and is scored
with the training reward. Results are saved after each batch, so that an interrupted evaluation
resumes without repeating finished problems.

| Set | Size | Used for |
| :--- | ---: | :--- |
| Dev subset | 200 | Learning curve across checkpoints, stratified by level |
| Dev | 1,000 | Untrained model and final adapter |
| MATH-500 | 500 | Untrained model now, trained models only for the final report |

The subset exists because evaluating every checkpoint on the full dev set would take hours on a T4.

## Skills

A skill-conditioned run differs from the standard GRPO baseline only in its training prompts, which
gain a block of tips above the problem. The untrained model writes the skills once, before any
training, and they stay fixed. Evaluation still omits them. The global bank is built as follows:

1. **Collect rollouts.** The untrained model samples four completions per problem, with GRPO's
   sampling settings and token limit, on exactly the problems GRPO will train on. TRL's sampler
   order depends only on the dataset size and seed, so those problems are known in advance. A longer
   run extends the same order, and adding skills to the prompts does not change the problems drawn.
2. **Write skills.** For each problem with a correct completion, the model reads its first correct
   attempt and its first incorrect attempt that finished, then writes a one-sentence hint on the
   core concept, with wording adapted from NuRL's hint prompt. Problems where every completion fails
   produce no skill, because a 0.8B model told only that an attempt was wrong is likely to invent a
   diagnosis.
3. **Filter.** A skill is dropped if it is unfinished or contains a boxed expression, the gold
   answer, a number from the problem, or wording that refers to the specific problem. Dropped skills
   are not rewritten.
4. **Select.** Skills are embedded with `all-mpnet-base-v2` and clustered into 20 groups with
   k-means. The bank is the most central skill from each of the 10 largest clusters. It reuses the
   model's own sentences, because a 0.8B model asked to merge them could lose their content.
5. **Pilot.** On the 200 training problems after those GRPO trains on, the untrained model is sampled
   with and without the bank. RL runs only if skills do not lower accuracy and they raise the share
   of mixed groups.

Clustered retrieval, per-problem conditioning, and the shuffled control are planned but not built.

## Run records

Each invocation writes a timestamped `run-*.json` with its command, resolved config, Git commit,
dataset revisions, device, and installed package versions.
