# Minimal-replication notebooks

Three self-contained Jupyter notebooks, one per probe, for anyone who wants
to see the mechanism behind each result and run a small version of it
themselves without setting up the full pipeline (`lm_eval`, multi-stage
shell scripts, cached activation directories, etc.).

| Notebook | Probe | What it does |
|---|---|---|
| [`01_refusal_direction_ablation.ipynb`](01_refusal_direction_ablation.ipynb) | 1 | Extracts a refusal direction (Arditi et al. diff-in-means), selects a layer via bypass/induce/KL scoring, ablates it, and scores WMDP-Bio accuracy against matched-control and random-direction baselines. |
| [`02_junk_direction_ablation.ipynb`](02_junk_direction_ablation.ipynb) | 2 | Extracts `û = normalize(mean(a_unlearned) - mean(a_base))` directly from forget-domain text, ablates it on RMU, and scores WMDP-Bio accuracy — the project's one clean positive recovery result. |
| [`03_unrelated_sft_recovery.ipynb`](03_unrelated_sft_recovery.ipynb) | 3 | Runs QLoRA SFT on GSM8K and compares WMDP-Bio / MMLU / GSM8K accuracy before and after. |

## What "minimal" means here

Every step is **inlined** — activation collection, hooks, direction math,
QLoRA setup, forced-choice scoring — nothing is imported from `scripts/`,
`junk_direction_ablation/`, or `wmdp_sft_recovery/`. That's deliberate: the
point is to let you read the actual implementation in one place and step
through it, not to call into the production scripts as a black box.

To keep runtime and setup low, each notebook trades scale for speed
relative to the full pipeline: smaller prompt/example/eval counts, a
single layer or position instead of the full sweep, and a hand-rolled
single-token forced-choice scorer instead of `lm_eval`. Each notebook's
first cell has a table spelling out exactly what's reduced. **Numbers you
get from these notebooks are for understanding the mechanism, not for
citing as a replication of the paper's numbers** — for those, run the
scripts named in each notebook's closing cell (and see the top-level
[README](../README.md) / [results.md](../results.md)).

## Requirements

- A GPU. Notebooks 1 and 2 load Llama-3-8B in bf16 (~16GB); notebook 3
  loads it in 4-bit NF4 via `bitsandbytes` (~6-8GB) for QLoRA.
- `pip install torch transformers datasets accelerate pandas numpy` for
  notebooks 1 and 2; add `peft bitsandbytes` for notebook 3.
- Hugging Face access to `meta-llama/Meta-Llama-3-8B-Instruct` (gated,
  near-universally auto-approved) for notebook 2's base-model reference.
- Optional: access to the gated `cais/wmdp-bio-forget-corpus` for
  notebook 2's headline number — it falls back to the public
  `cais/wmdp-corpora` cyber-forget corpus if you don't have it yet.

Each notebook runs top-to-bottom in a few minutes to ~20 minutes on a
single modern GPU (notebook 3's SFT step is the slowest; lower
`N_SFT_EXAMPLES` for a quicker smoke test).
