# Unlearning Recovery Probes

Does machine unlearning remove knowledge from a language model, or just
suppress access to it? Low accuracy on a forget-set benchmark doesn't tell
you which -- a genuinely-removed fact and a merely-hidden one score the same.
This repository holds three independent probes for telling them apart,
applied to the same six WMDP-unlearned Llama-3-8B-Instruct checkpoints
(RMU, ILU-RMU, IDK-AP, GradDiff, NPO, NPO-ILU):

1. **Refusal-direction ablation** -- does removing the direction that
   mediates refusal behavior also uncover forgotten knowledge?
2. **Forget-domain representation-direction ablation** -- does removing a
   direction built directly from the unlearned-vs-base activation shift on
   forget-domain text (replicating & extending [Arditi & Chughtai](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow))?
3. **Unrelated supervised fine-tuning** -- does QLoRA SFT on GSM8K (a dataset
   with near-zero mutual information with WMDP-bio) disturb the unlearned
   weights enough to raise WMDP-Bio accuracy?

**Headline results:**

- **Probe 1 (refusal direction):** no recovery beyond random/matched-control
  ablation for any of the 6 methods, even for **ILU-RMU** -- the one
  checkpoint where the same direction demonstrably collapses refusal
  behavior 98%&rarr;0% with 100% clean bypass. 95% CIs exclude recovery
  larger than ~1.4pp on the interpretable rows.
- **Probe 2 (representation direction):** a differently-built direction
  recovers **57% (RMU)** and **61% (ILU-RMU)** of the gap to full-knowledge
  WMDP-Bio accuracy, McNemar-significant at p < 1e-27 against baseline,
  matched control, and 8 random-direction controls. The same construction
  finds nothing beyond matched control for GradDiff, NPO, NPO-ILU, or IDK-AP
  (pre-registered null, Bonferroni-corrected).
- **Probe 3 (unrelated SFT):** WMDP-Bio accuracy rises for 5 of 6 methods
  after GSM8K fine-tuning, with MMLU held flat for RMU (+41pp bio),
  ILU-RMU (+24pp), and NPO (+23pp) -- GradDiff's rise is confounded by a
  matching MMLU jump (general capability repair, not suppression coming
  undone); a full-knowledge control run through the identical recipe stays
  flat.
- **Benchmark accuracy overstates recovery.** All of the above is
  forced-choice loglikelihood scoring -- the model never generates anything.
  Reading the actual free-text completions on flipped questions (genuine /
  format-artifact / contradictory reasoning, blinded human labeling) finds
  only ~40-47% verified-genuine recovery where accuracy suggested more, and
  RMU-family checkpoints turn out to contradict their *own* stably-correct
  forced-choice answers 30-37% of the time even with no ablation involved --
  well above the full-knowledge base model's 6.7% floor. Two of six methods
  (NPO-ILU, and NPO under some interventions) produce free text too
  incoherent to classify at all, which forced-choice accuracy alone doesn't
  surface.

Full results and the reasoning behind each conclusion: [results.md](results.md).
Companion write-up: *Probing Knowledge Recovery in Unlearned Models*.

## Results and writeups

- [results.md](results.md) -- results summary for all three probes
- [direction_extraction.md](direction_extraction.md) -- refusal-direction
  prompt source, candidate sweep, and selection procedure (Probe 1)
- [junk_direction_ablation/README.md](junk_direction_ablation/README.md) --
  representation-direction construction and pipeline (Probe 2)
- [wmdp_sft_recovery/README.md](wmdp_sft_recovery/README.md) -- SFT-recovery
  design and gotchas (Probe 3)

---

## Installation

```bash
pip install -r requirements.txt
```

The scripts accept Hugging Face model IDs or local checkpoint paths.

---

## Model checkpoints

WMDP-unlearned checkpoints are from [OPTML-Group](https://huggingface.co/OPTML-Group)
(GradDiff, IDK-AP, ILU-RMU, NPO, NPO-ILU) and [ScaleAI](https://huggingface.co/ScaleAI/mhj-llama3-8b-rmu)
(RMU), all unlearned from `meta-llama/Meta-Llama-3-8B-Instruct`. This
repository focuses on direction extraction/evaluation and SFT-recovery
evaluation rather than reproducing the unlearning training pipelines
themselves.

---

## Probe 1: refusal-direction ablation

> An earlier script, `wmdp_bio_refusal_direction_eval.py`, is removed from
> this repo. It re-tokenized already-templated prompts (silently
> double-inserting the BOS token, which also undercounted its own
> forced-choice WMDP-Bio scoring vs. `lm_eval`), only searched the final
> token position instead of the full post-instruction grid, and applied
> neither Arditi et al.'s nor COSMIC's selection filters. `extract_refusal_direction.py`
> + `wmdp_bio_lm_eval_ablation.py` (below) fix all of that -- see
> [direction_extraction.md](direction_extraction.md) for the full rationale.

**1. Extract a refusal direction.** `extract_refusal_direction.py` computes
difference-in-means candidates over the last 5 post-instruction token
positions x every layer, using Arditi et al. (2024)'s own AdvBench-harmful vs.
Alpaca-harmless setup (a general "refusal" direction, not bio-specific).
`--selection-method both` (the default) computes **both** selection
algorithms from a single candidate sweep:

- `mean_diff` -- Arditi et al. (2024) Appendix C.1's causal selection
  (minimize bypass_score subject to induce_score > 0, kl_score < 0.1,
  layer < 0.8L).
- `cosmic` -- Siu et al. (2025) COSMIC's concept-inversion cosine-similarity
  selection on the low-similarity layers, with the same KL/layer-fraction
  filters used in the official COSMIC repo (`wang-research-lab/COSMIC`).

```bash
python scripts/extract_refusal_direction.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --selection-method both \
  --output-root outputs/idk-ap/direction_extraction/generic
```

Writes, per selection method: `direction_{method}.pt` (with selection
diagnostics in the metadata), `candidate_diagnostics.csv` (every candidate's
bypass/induce/KL/COSMIC scores, for auditing), and
`direction_{method}_matched_control.pt` -- a split-half difference-in-means
direction from a fresh, disjoint harmless prompt pool at the same
(position, layer), testing whether ablating *any* direction built this way at
this spot moves WMDP-Bio accuracy, not just the one selected as "refusal."
(`mean_diff` finds zero passing candidates for 4 of the 6 models tested here
-- see [direction_extraction.md](direction_extraction.md) for why the
pipeline always runs both methods rather than picking one upfront.)

**2. Evaluate WMDP-Bio accuracy under ablation.** `wmdp_bio_lm_eval_ablation.py`
scores a direction against `lm_eval`'s own `wmdp_bio` task:

```bash
python scripts/wmdp_bio_lm_eval_ablation.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --matched-control-direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic_matched_control.pt \
  --output-root outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic \
  --apply-chat-template
```

Runs `wmdp_bio` on the same loaded model across `baseline` (no hook, unless
`--skip-baseline`), `selected_direction_ablation`, `matched_control_ablation`
(if given), and `random_direction_ablation_{0..N}` controls, writing a
comparison summary (including a random-control mean/std and a
`selected_control_z` distance from that distribution) to
`wmdp_bio_lm_eval_summary.json`.

**3. Confirm the direction does something.** A null accuracy result is only
informative if the direction demonstrably affects behavior.
`wmdp_refusal_behavior_check.py` generates on 100 held-out AdvBench prompts
(disjoint from extraction) with and without ablation and measures the
refusal rate, reusing the exact ablation code from step 2:

```bash
python scripts/wmdp_refusal_behavior_check.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --extraction-prompts-csv outputs/idk-ap/direction_extraction/generic/direction_prompts.csv \
  --matched-control-direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic_matched_control.pt \
  --output-root outputs/idk-ap/wmdp_refusal_behavior_check/generic_cosmic
```

**4. Sanity-check the ablation hook itself.** `hook_activation_check.py`
scores the same `lm_eval` request twice on one loaded model -- once with no
hook, once with the ablation hook registered -- and asserts the per-choice
loglikelihoods actually differ, so a null WMDP-Bio result can't be a silently
inactive hook:

```bash
python scripts/hook_activation_check.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --num-docs 3
```

**Full pipeline, one model, both selection methods:**

```bash
MODEL=OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
MODEL_LABEL=idk-ap \
OUTPUT_ROOT=outputs/idk-ap \
HARDWARE_PROFILE=a100 \
bash examples/run_full_wmdp_bio_pipeline.sh
```

Repeat per model. Random-direction and matched-control conditions are only
computed once per model (the first ablation-eval call); subsequent calls for
the same model pass `--num-random-controls 0`.

**Paired significance testing.** `wmdp_bio_lm_eval_ablation.py`,
`junk_direction_ablation/eval_junk_ablation_lm_eval.py`, and
`wmdp_sft_recovery/eval_recovery_lm_eval.py` all persist per-question
correctness to `per_doc_correctness/<task>.json` (doc_id -> bool), since
every condition in this repo is scored on the *same* question set and a
paired test (exact binomial McNemar + a paired bootstrap CI) is the correct
tool -- not two independent-sample tests. `scripts/analyze_paired_recovery.py`
runs that comparison between any two conditions' persisted files:

```bash
python scripts/analyze_paired_recovery.py \
  --correctness-a outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/per_doc_correctness/wmdp_bio/selected_direction_ablation.json \
  --correctness-b outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/per_doc_correctness/wmdp_bio/matched_control_ablation.json \
  --label-a selected_direction --label-b matched_control \
  --output-json outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/mcnemar_selected_vs_matched_control.json
```

**Result:** across six checkpoints, ablating a rank-1 refusal direction does
not recover WMDP-Bio accuracy beyond random-direction or matched-construction
controls -- not even for **ILU-RMU**, the only fully validated
behavioral-bypass row (refusal 98%&rarr;0% with 100% clean bypass).
GradDiff's refusal-rate collapse (61%&rarr;0%) is only partially validated:
ablated clean bypass is 30%, with most generations still degenerate on an
already-broken model. See [results.md](results.md) for the full
per-model breakdown and the paired-significance tables.

---

## Probe 2: forget-domain representation-direction ablation

A differently-built direction: instead of a generic refusal direction,
[`junk_direction_ablation/`](junk_direction_ablation/) extracts
`û = normalize(mean(a_unlearned) - mean(a_base))` directly from forget-domain
text (WMDP bio/cyber-forget corpora), targeting whatever RMU-family training
actually did to forget-domain representations rather than a refusal signal.
This replicates and extends [Arditi & Chughtai's finding](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow)
(~71% WMDP-Bio gap recovery on RMU-unlearned Zephyr-7B) to Llama-3-8B-Instruct
and to five additional unlearning methods:

```bash
cd junk_direction_ablation
bash run_step3.sh rmu                              # RMU / ILU-RMU positive-control path
bash run_loss_based_junk_null.sh                    # GradDiff / NPO / NPO-ILU / IDK-AP null check
```

See [`junk_direction_ablation/README.md`](junk_direction_ablation/README.md)
for the pipeline (extraction, ablation arms, blinded qualitative labeling)
and [results.md](results.md#a-differently-built-direction-junk-direction-ablation-recovers-rmu-and-ilu-rmu-cleanly)
for the outcome: 57% (RMU) and 61% (ILU-RMU) gap recovery with an
overwhelming McNemar signal, and a pre-registered null for the other four
methods that survives Bonferroni correction -- so "direction ablation finds
nothing" (Probe 1's result) turns out to be a property of the
*refusal*-direction construction on these methods, not of direction ablation
as a technique.

---

## Probe 3: unrelated SFT recovery

A different recovery mechanism entirely: does unrelated fine-tuning undo
unlearning by disturbing the unlearned weights, rather than by ablating a
direction? [`wmdp_sft_recovery/`](wmdp_sft_recovery/) runs QLoRA SFT on
`openai/gsm8k` (near-zero mutual information with WMDP-bio, so any bio
movement afterward is the unlearning coming undone, not new knowledge going
in) on each unlearned model plus a full-knowledge control, checkpointed at
1000/3000/6000 examples, evaluated via `lm_eval` on `wmdp_bio` (recovery
signal), `mmlu` (utility gate), and `gsm8k` (uptake check):

```bash
cd wmdp_sft_recovery
bash run_base_control.sh                          # full-knowledge control
METHODS="RMU" bash run_unlearned.sh                # one unlearned model at a time
python aggregate_table.py                          # -> results/wmdp_sft_recovery/recovery_table.md
```

See [`wmdp_sft_recovery/README.md`](wmdp_sft_recovery/README.md) for the full
design (controls, precision/chat-template gotchas, and the cross-check
against this repo's ablation-arm baselines) and
[results.md](results.md#sft-recovery-does-unrelated-fine-tuning-undo-unlearning)
for the outcome, including `sample_flipped_generations.py`'s qualitative
read of the free-text completions behind the accuracy numbers.

---

## Why accuracy alone isn't enough

Every probe above is scored as forced-choice loglikelihood accuracy on
`wmdp_bio` -- the model never generates anything, so an accuracy rise only
means the argmax answer flipped, not that the model reasoned its way there.
Both Probe 2 (`junk_direction_ablation/sample_flipped_generations.py`,
`sample_stable_correct_generations.py`, `sample_base_correct_generations.py`)
and Probe 3 (`wmdp_sft_recovery/sample_flipped_generations.py`) re-generate
**free-form** completions for questions that flipped wrong-to-right, and a
human classifies each as **genuine** (reasoning supports the correct
answer), **format-artifact** (degenerate/repetitive output that happens to
match the gold letter), or **contradictory** (the model's own reasoning
names a different option than the one credited as correct). This is what
surfaces, e.g., NPO-ILU's benchmark recovery being an argmax shift on a model
that cannot generate coherent text, and RMU-family checkpoints contradicting
their own stably-correct answers even without any ablation touching them.
See [results.md](results.md) for every labeled table and the blinding
protocol behind the n=30 rounds.

---

## References

- Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction.](https://arxiv.org/abs/2406.11717)
- Siu et al. (2025). [COSMIC: Generalized Refusal Direction Identification in LLM Activations.](https://arxiv.org/abs/2506.00085)
- Chughtai, B. (2024). [Unlearning via RMU is mostly shallow.](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow) (Arditi & Chughtai's forget-domain representation-direction result that Probe 2 replicates and extends.)
- Łucki et al. (2024). [An Adversarial Perspective on Machine Unlearning for AI Safety.](https://arxiv.org/abs/2409.18025)
- Hu et al. (2025). [Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning.](https://arxiv.org/abs/2406.13356)
- Deeb & Roger (2024). [Do Unlearning Methods Remove Information from Language Model Weights?](https://arxiv.org/abs/2410.08827)
- WMDP: [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.](https://www.wmdp.ai/)
