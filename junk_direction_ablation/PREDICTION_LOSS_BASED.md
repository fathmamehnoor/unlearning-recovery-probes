# Junk-direction null — GradDiff / NPO / NPO-ILU / IDK-AP (pre-registration)

**Write this down before looking at junk-ablation results for these four.**

## Setup

- Models: GradDiff, NPO, NPO-ILU, IDK-AP (OPTML-Group Llama-3-8B WMDP instruct checkpoints)
- Extraction: û = normalize(mean(a_unl) − mean(a_base)) on forget_bio (+ forget_cyber if present) and retain
- Pooling: `last_token` only (matches the RMU/ILU-RMU headline variant; no smoke sweep)
- Layers: **both** diagnostic nominees from the layer-divergence probe (not layer 7):

| Model | Divergence layer | Norm-spike layer |
|---|---|---|
| GradDiff | 6 | 30 |
| NPO | 31 | 15 |
| NPO-ILU | 2 | 30 |
| IDK-AP | 2 | 30 |

- Ablation arms (trimmed): `baseline`, `junk_direction_ablation`, `matched_control_ablation`, `random_direction_ablation_{0,1,2}`
- Eval: `wmdp_bio` only, `log_samples=True`, per-doc correctness persisted
- **No MMLU.** Nothing to gate if bio does not move.
- **No free-text.** Only meaningful on flips; a null has no flips worth reading.
- **No smoke sweep.** Eight runs = 4 models × 2 layers.

## Prediction (null)

**No recovery beyond matched control at either diagnostic layer, for any of the four models.**

Operationally: junk − matched is within noise; McNemar junk vs matched is non-significant at conventional α = 0.05.

## Degenerate models (include, flag, do not interpret)

NPO and NPO-ILU are degenerate / uninterpretable on every prior arm in this project. They are included so the panel is not cherry-picked. Flag both the same way as elsewhere; **read nothing into either result.**

## Promotion rule (spend full budget only on hits)

If **any** model × layer shows:

1. junk − matched **≳ 5 pp** on WMDP-bio, **and**
2. McNemar junk vs matched **significant** (exact binomial, α = 0.05), **and**
3. junk accuracy **above** the rank-matched random-control mean (guards the 2D-junk vs 1D-matched confound),

then **stop treating that cell as a control** and re-run it under the full RMU protocol (MMLU utility gate + free-text flip read). Otherwise leave it as a trimmed null.

**Required for a conclusive null:** `forget_bio` **and** `forget_cyber` directions both present. Bio-only ablation is an under-recovery confound — do not interpret.

## Signed before results

- Date: 2026-08-02
- Operator: Mehnoor (via Cursor agent; completed on Vast instance 46556203)
- Notes: Trimmed null; dual diagnostic layers; last_token; bio+cyber required; NPO/NPO-ILU flagged degenerate.

## Outcome (filled after looking)

**Null holds. No promotion triggers.** Full table in
`local_outputs/junk_direction_loss_based/SUMMARY.md` and
`unlearning_direction_evaluation/results.md` (junk-direction section).
Largest junk−matched gaps: NPO layer 15 (+4.2pp, p=0.014, degenerate) and
IDK-AP layer 2 (+2.0pp, p=0.011) — both under the 5pp bar.
