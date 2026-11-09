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

- **Only the final boxed answer counts.** A missing, empty, or unclosed final box scores 0.0, so
  truncated completions cannot earn credit from an earlier answer.
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
