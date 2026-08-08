# Junk-direction ablation (Probe 2)

Replicates and extends [Arditi & Chughtai's finding](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow)
that RMU-family unlearning can be partly undone by ablating a *forget-domain
representation* direction, rather than the refusal-behavior direction used in
[`scripts/extract_refusal_direction.py`](../scripts/extract_refusal_direction.py)
(Probe 1). See [`../results.md`](../results.md#a-differently-built-direction-junk-direction-ablation-recovers-rmu-and-ilu-rmu-cleanly)
for the full results and discussion; this file documents the pipeline.

## The direction

For each forget-domain corpus (WMDP bio-forget, cyber-forget), extract

```
û = normalize(mean(a_unlearned) - mean(a_base))
```

-- the residual-stream vector separating the unlearned model's activations
from the base model's, pooled over the last token, computed directly on
forget-domain text (not on refusal prompts at all). This targets whatever
RMU-family training actually did to forget-domain representations, rather
than a generic refusal signal. Ablating û_bio (+ û_cyber if available)
together, at the layer both `wmdp-bio` and `wmdp-cyber` were unlearned on,
recovered 57% (RMU) and 61% (ILU-RMU) of the gap to full-knowledge WMDP-Bio
accuracy in this repo's runs, against matched-control and 8x random-direction
controls, both McNemar-significant at p < 1e-27.

## Layout

| File | Role |
|---|---|
| `PREDICTION.md` | Pre-registered prediction, signed before looking at RMU/ILU-RMU results |
| `PREDICTION_LOSS_BASED.md` | Pre-registered null + promotion rule for the loss-based methods (GradDiff/NPO/NPO-ILU/IDK-AP) |
| `PREREGISTER_LOSS_BASED_SWEEP.md` | Pre-registration for the 16-configuration layer/pooling sweep behind `DIAGNOSTIC_LAYERS` |
| `ablation_lib.py` | Shared hooks, multi-direction ablation, `lm_eval` helpers |
| `extract_junk_directions.py` | Probe-set construction + direction extraction (6 layer/pooling variants) |
| `eval_junk_ablation_lm_eval.py` | Runs `wmdp_bio` + `mmlu` across baseline / junk / matched-control / random-control arms, persists per-doc correctness |
| `sample_flipped_generations.py` | Free-text generation on wrong→right flips, for the genuine/format-artifact/contradictory read |
| `sample_stable_correct_generations.py` | Ablated-model null: free-text on docs both baseline and junk-ablation already get right |
| `sample_base_correct_generations.py` | Base-model null: free-text on docs the base model gets right under forced-choice scoring |
| `calibrate_letter_proxy.py` | Automatic first-stated-letter proxy vs. hand labels (P0a) -- an earlier, since-superseded shortcut; kept for the record, see caveat in `results.md` |
| `build_blind_p0c_batch.py` / `plan_blind_p0c_ids.py` / `unblind_p0c_batch.py` | Blinding harness for the n=30 genuine/format-artifact/contradictory labeling protocol |
| `chat_template_artifact_check.py` | Checks whether ScaleAI's broken RMU chat template (renders to ~2 tokens) confounds any reported number |
| `npo_ablated_degeneracy_check.py` | Quantifies NPO / NPO-ILU baseline generation degeneracy referenced in `results.md` |
| `analyze_stats.py` | Exact binomial McNemar + paired bootstrap CI on persisted per-doc correctness |
| `sweep_*.py` | The loss-based layer/pooling sweep (extraction, evaluation, sanity checks, reporting) for GradDiff/NPO/NPO-ILU/IDK-AP |
| `run_step3.sh` | End-to-end orchestration for RMU / ILU-RMU |
| `run_loss_based_junk_null.sh` | End-to-end orchestration for the loss-based null (GradDiff/NPO/NPO-ILU/IDK-AP) |

## Arms (per model)

1. `baseline`
2. `junk_direction_ablation` -- û_bio (+ û_cyber if present), ablated together
3. `matched_control_ablation` -- same estimator on retain/wikitext text instead of forget-domain text
4. `random_direction_ablation_{0..7}`
5. `base_model` (optional full-knowledge reference)

## Quick start (RMU / ILU-RMU)

```bash
cd junk_direction_ablation

# 0. Sign the pre-registration
#    edit PREDICTION.md

# 1. Extract directions (bio + cyber + retain; layers 6/7/8 x last/mean)
python extract_junk_directions.py \
  --output-root ../outputs/junk_direction \
  --models rmu,ilu-rmu \
  --n-chunks 150 \
  --max-length 512 \
  --hardware-profile manual \
  --dtype bfloat16 \
  --batch-size 2 \
  --gpu-memory 22GiB \
  --skip-cyber-if-missing

# 2. Smoke all 6 variants on RMU (pick winner)
python eval_junk_ablation_lm_eval.py \
  --model-label rmu \
  --directions-root ../outputs/junk_direction/directions \
  --sweep-variants \
  --limit 64 \
  --skip-base-model \
  --num-random-controls 2 \
  --hardware-profile manual \
  --dtype bfloat16 \
  --batch-size auto:2 \
  --gpu-memory 22GiB \
  --output-root ../outputs/junk_direction/smoke/rmu

# Record winner in PREDICTION.md, then full eval:
WINNER=$(python -c "import json; print(json.load(open('../outputs/junk_direction/smoke/rmu/selected_variant.json'))['winner'])")

python eval_junk_ablation_lm_eval.py \
  --model-label rmu \
  --directions-root ../outputs/junk_direction/directions \
  --variant "$WINNER" \
  --num-random-controls 8 \
  --hardware-profile manual \
  --dtype bfloat16 \
  --batch-size auto:2 \
  --gpu-memory 22GiB \
  --output-root ../outputs/junk_direction/eval/rmu_${WINNER}

# 3. Stats
python analyze_stats.py \
  --eval-root ../outputs/junk_direction/eval/rmu_${WINNER} \
  --baseline-acc-reference 0.281 \
  --base-model-acc-reference 0.731

# 4. Free-text on 15 flips
python sample_flipped_generations.py \
  --model-label rmu \
  --directions-root ../outputs/junk_direction/directions \
  --variant "$WINNER" \
  --baseline-correctness ../outputs/junk_direction/eval/rmu_${WINNER}/per_doc_correctness/wmdp_bio/baseline.json \
  --junk-correctness ../outputs/junk_direction/eval/rmu_${WINNER}/per_doc_correctness/wmdp_bio/junk_direction_ablation.json \
  --num-samples 15 \
  --output-jsonl ../outputs/junk_direction/eval/rmu_${WINNER}/flipped_generations.jsonl \
  --label-template-path ../outputs/junk_direction/eval/rmu_${WINNER}/flipped_label_sheet.json

# 5. Ablated null: always-correct docs (baseline AND junk), still under junk ablation
python sample_stable_correct_generations.py \
  --model-label rmu \
  --directions-root ../outputs/junk_direction/directions \
  --variant "$WINNER" \
  --baseline-correctness ../outputs/junk_direction/eval/rmu_${WINNER}/per_doc_correctness/wmdp_bio/baseline.json \
  --junk-correctness ../outputs/junk_direction/eval/rmu_${WINNER}/per_doc_correctness/wmdp_bio/junk_direction_ablation.json \
  --num-samples 30 \
  --seed 20260802 \
  --output-jsonl ../outputs/junk_direction/eval/rmu_${WINNER}/stable_correct_generations.jsonl
```

Or: `bash run_step3.sh rmu` (see script for flags).

The blinded genuine/format-artifact/contradictory labeling round (n=30 per
arm, identity sealed) is planned with `plan_blind_p0c_ids.py`, packaged with
`build_blind_p0c_batch.py`, and scored after labeling with
`unblind_p0c_batch.py` -- see the "Qualitative check" subsection of
[`../results.md`](../results.md) for the protocol and results.

## Loss-based null (GradDiff / NPO / NPO-ILU / IDK-AP)

These four were not in the original positive-control path (Arditi & Chughtai
tested RMU only). Trimmed protocol:

- Pre-register in `PREDICTION_LOSS_BASED.md` (null + promotion rule) **before** looking.
- Extract at **both** diagnostic layers per model (divergence + norm-spike,
  from a separate layer-divergence probe -- see `ablation_lib.DIAGNOSTIC_LAYERS`),
  `last_token` pooling only.
- Arms: baseline, junk, matched, 3 randoms. `wmdp_bio` only. No MMLU, no free-text, no smoke.
- NPO / NPO-ILU: include and flag as degenerate; do not interpret (see
  `npo_ablated_degeneracy_check.py`).

```bash
# Sign PREDICTION_LOSS_BASED.md first, then:
bash run_loss_based_junk_null.sh
# or a subset:
MODELS=graddiff,idk-ap bash run_loss_based_junk_null.sh
```

Flags already supported without the wrapper:

```bash
python extract_junk_directions.py ... --models graddiff,npo,npo-ilu,idk-ap \
  --use-diagnostic-layers --positions last_token

python eval_junk_ablation_lm_eval.py --model-label graddiff \
  --variants layer6_last_token,layer30_last_token \
  --tasks wmdp_bio --num-random-controls 3 --skip-base-model ...
```

Result: pre-registered null holds for all four -- no cell meets the
promotion rule (junk-matched >=5pp **and** McNemar p<0.05 **and** junk above
the rank-matched random mean), even after a Bonferroni correction across the
8 tests run. Junk-direction recovery tracks the RMU-family representational
bump specifically; it is not a generic artifact of ablating any
forget-minus-base difference. Full table in
[`../results.md`](../results.md#a-differently-built-direction-junk-direction-ablation-recovers-rmu-and-ilu-rmu-cleanly).

The 16-configuration layer x pooling sweep behind `DIAGNOSTIC_LAYERS` (the
divergence/norm-spike diagnostic layers per loss-based method) is
pre-registered in `PREREGISTER_LOSS_BASED_SWEEP.md` and implemented in
`sweep_extract_base_activations.py` / `sweep_run_model.py` /
`sweep_sanity_check.py` / `sweep_report.py`.

## Notes

- **No chat template** by default (ScaleAI's RMU checkpoint ships a broken
  chat template that renders message content to ~2 tokens; raw
  `Question:`/`Answer:`-style text matches how RMU was trained).
  `chat_template_artifact_check.py` verifies this isn't silently confounding
  any reported number.
- **Cyber:** OPTML/ScaleAI checkpoints are named WMDP (not WMDP-bio).
  Extraction requires `forget_cyber` by default. Only set
  `--skip-cyber-if-missing` and `--allow-bio-only` if you deliberately accept
  the under-recovery confound of a bio-only direction.
- **Bio-forget corpus** is gated (`cais/wmdp-bio-forget-corpus`) or from
  `wmdp-corpora.zip` -> `data/wmdp/wmdp-corpora/bio-forget-corpus.jsonl`.
- **Padding:** extraction uses left padding (last-token pool); `lm_eval` uses
  **right** padding. Do not mix these.
- **Layer index:** residual stream **after** decoder block L, matching the
  convention used throughout this repo.
- **MMLU tanks:** re-run eval with `--ablation-mode resid_pre_only`.
- **Smoke winner** is chosen by `junk - baseline` recovery (and must beat
  matched control when available), not raw junk accuracy -- see the "smoke
  favored a different variant than the full run used" caveat in
  [`../results.md`](../results.md).

## Decision rule (from `PREDICTION.md`)

RMU junk-ablation WMDP-bio accuracy: **>45%** clear replication ·
**35-45%** partial · **<32%** failure (do not debug blind -- the published
comparison point is Zephyr-7B RMU, not this Llama-3-8B checkpoint).
