# Results

Summary of three recovery probes applied to the same six WMDP-unlearned
Llama-3-8B-Instruct checkpoints: whether ablating a refusal direction
recovers WMDP-Bio knowledge, whether ablating a differently-built
forget-domain *representation* direction does, and whether unrelated
fine-tuning does. See [direction_extraction.md](direction_extraction.md) for
how the refusal directions were built and selected, and
[junk_direction_ablation/README.md](junk_direction_ablation/README.md) for
how the representation directions were built and selected. (An earlier TOFU
direction-ablation arm and a since-fixed WMDP-Bio scorer were tested during
this project and have been removed from the codebase; this file reflects the
current, corrected pipeline only.)

## WMDP-Bio direction-ablation

`lm_eval`'s `wmdp_bio` task (n = 1273), scored across `baseline`,
`selected_direction_ablation`, `matched_control_ablation`, and 8
`random_direction_ablation` conditions on the same loaded model
(`scripts/wmdp_bio_lm_eval_ablation.py`). Six WMDP-unlearned Llama-3-8B-Instruct
checkpoints (OPTML-Group's GradDiff, IDK-AP, ILU-RMU, NPO, NPO-ILU, and
ScaleAI's RMU).

| model | method | selected acc | matched-control acc | random-control mean (±std) |
|---|---|---|---|---|
| GradDiff | cosmic | 27.0% | 27.1% | 26.9% (±0.6%) |
| IDK-AP | cosmic | 26.8% | 27.1% | 27.2% (±0.1%) |
| ILU-RMU | mean_diff | 32.3% | 32.2% | 32.1% (±0.6%) |
| ILU-RMU | cosmic | 32.8% | 32.7% | *(shares controls above)* |
| NPO | cosmic | 24.5% | 24.7% | 24.3% (±0.2%) |
| NPO-ILU | cosmic | 27.5% | 26.8% | 27.1% (±0.3%) |
| RMU | mean_diff | 26.3% | 26.2% | 26.1% (±0.2%) |
| RMU | cosmic | 28.0% | 25.5% | *(shares controls above)* |

**A note on cross-arm baseline comparisons.** The columns above are all
post-ablation/control accuracies -- the no-hook `baseline` condition isn't
shown, and is *not* the same number as any of them. For ILU-RMU it's 31.7%
(`mcnemar_recheck_results/outputs/ilu-rmu/wmdp_bio_lm_eval/{generic_cosmic,generic_mean_diff}/`,
identical in both direction runs, as expected since they load the same
model). That's noticeably below the 34.0% (SFT arm, step-0) and 34.3%
(junk-direction arm) baselines used elsewhere in this document for the same
model on the same task -- **but this isn't noise, and it isn't a
wrong-model bug.** Every run's own `run_config.json`/`arm_config.json`
confirms the model id is `OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct` in
all three arms (never the base Llama checkpoint). The difference is
`apply_chat_template`: this arm (`scripts/wmdp_bio_lm_eval_ablation.py`)
defaults to chat-template **on**; the SFT-recovery arm and the
junk-direction arm both run with it explicitly **off** (a convention adopted
after discovering ScaleAI's RMU ships a broken chat template that renders to
~2 tokens -- see the SFT-recovery section below -- and kept for all
RMU-family checkpoints for consistency). The two chat-template-off baselines agree with
each other to within 0.3pp (comfortably inside this task's own noise band --
random-control spread elsewhere in this document runs ±0.4-0.7pp on
n=1273), while the chat-template-on baseline sits ~2.3-2.6pp below both,
consistently -- a real, systematic prompt-formatting effect, not run-to-run
variance. This does **not** affect any statistic reported above or in the
junk-direction/SFT sections: every McNemar test, z-score, and recovery
fraction in this document is computed entirely within one arm against that
same arm's own baseline, so numerator and denominator always share the same
template setting. It only matters if you compare a raw accuracy number from
this table directly against one from a different section -- which this
document does not do, but a reader skimming across sections could easily do
by eye, so: don't treat "ILU-RMU's WMDP-Bio baseline" as one fixed number
across this file. RMU's refusal-direction run also used
`apply_chat_template: true` (confirmed from its `run_config.json`), so the
same offset almost certainly applies there too, though the local RMU runs
were invoked with `--skip-baseline` and don't have a persisted `baseline`
row to confirm the exact number.

**Robustness (ILU-RMU, template-off).** The ILU-RMU refusal null was re-run
with the chat template off, matching the junk and SFT arms
(`ilu_rmu_template_off_results/`; directions reused, not re-extracted;
pre-registered in `ilu_rmu_template_off_preregister.md`). Baseline moved to
34.2%, confirming the template offset. The direction still recovered nothing
beyond matched control (mean_diff McNemar selected vs matched p = 0.13;
cosmic p = 0.92; both vs baseline also non-significant). So the contrast
between the refusal null and the junk recovery on this model is not an
artifact of prompt formatting. Template-on remains primary for this arm,
since that is the format the direction was extracted in (extraction
chat-formatted, this eval not — a stated limitation of the robustness check).

No selected direction cleared its own random-control distribution by more
than noise (the largest deviation, IDK-AP's, is in the wrong direction --
ablation scores *below* the random-control mean, not above). Both selection
algorithms agree with each other in every model where both produced a
direction (see [direction_extraction.md](direction_extraction.md) for why
`mean_diff` only produced one for RMU and ILU-RMU).

Control-distribution overlap is suggestive but not a significance test on
its own. Where per-question correctness was persisted, `scripts/analyze_paired_recovery.py`
gives the rigorous version -- exact binomial McNemar on the same ~1273
questions, since `selected_direction_ablation` and `matched_control_ablation`
are paired, not independent, samples. Available for 3 of the 8 rows above:

| model | method | vs. matched-control | vs. baseline |
|---|---|---|---|
| GradDiff | cosmic | p = 0.49 (n10+n01=52) | p = 0.79 (n10+n01=59) |
| ILU-RMU | cosmic | p = 0.78 (n10+n01=118) | p = 0.22 (n10+n01=54) |
| ILU-RMU | mean_diff | p = 0.70 (n10+n01=109) | p = 0.63 (n10+n01=69) |

All six comparisons are far from significant. Paired-bootstrap 95% CIs on the
accuracy delta (selected − matched; 10k resamples, n=1273) are:

| model | method | Δ selected−matched | 95% CI |
|---|---|---|---|
| GradDiff | cosmic | −0.5pp | [−1.6, +0.6]pp |
| ILU-RMU | cosmic | −0.3pp | [−2.0, +1.4]pp |
| ILU-RMU | mean_diff | −0.4pp | [−2.0, +1.2]pp |

Every interval crosses zero. Across these three rows the highest upper bound
is **+1.4pp** — so we exclude refusal-direction recovery above about 1.4pp on
the models where the behavioral sensor is even partially live. That is a
bounded null, not an empty one: the same ILU-RMU checkpoint recovers ~24pp
under junk-direction ablation below. McNemar is reported only for these three
interpretable rows. The other four (IDK-AP, NPO, NPO-ILU, RMU) are **not**
unfinished: IDK-AP is already at a refusal floor (91% clean-bypass before any
ablation), NPO and NPO-ILU are degenerate at baseline, and RMU's directions
here do not cleanly bypass refusal either (behavioral table below) — pairing
McNemar on those flat accuracy numbers would dress up a non-result. They stay
on the control-distribution table only.

### The direction still does something: behavioral bypass check

`scripts/wmdp_refusal_behavior_check.py` checks whether the same directions
that show no WMDP-Bio recovery actually bypass refusal-like behavior on 100
held-out AdvBench harmful prompts (disjoint from extraction). This is what
makes the null result above informative rather than a dead sensor.

| model | method | baseline refusal rate | ablated refusal rate | ablated clean-bypass rate |
|---|---|---|---|---|
| ILU-RMU | mean_diff | 98% | 0% | 100% |
| RMU | mean_diff | 70% | 72% | 13% |
| RMU | cosmic | 70% | 62% | 32% |
| GradDiff | cosmic | 61% | 0% | 30%\*\*\* |
| IDK-AP | cosmic | 0%\* | 0% | 92% |
| NPO | cosmic | 5%\*\* | 0% | 6% |
| NPO-ILU | cosmic | 0%\*\* | 0% | 2% |

\* IDK-AP already complies with harmful prompts at baseline (91% clean-bypass
before any ablation) -- it doesn't exhibit refusal-phrase behavior to bypass.
\*\* NPO and NPO-ILU are heavily degenerate at baseline (94% / 100% of
generations are repetition/garbage per the check's degeneracy flag), so
refusal rate isn't a meaningful signal for these two either way.
\*\*\* GradDiff is only **partially** validated as a behavioral-bypass row.
Refusal phrases do collapse 61%&rarr;0% (matched control unchanged), but
ablated clean-bypass is only 30% -- the other 70% are degenerate under the
same scorer, on a model whose baseline was already 77% degenerate and whose
MMLU sits at chance (24.9%). Hand-reading 20 ablated generations confirms
most are repetition / spaced-token junk rather than coherent compliance.
Refusal-substring disappearance on an already-broken model is weak evidence
that the direction is doing targeted refusal work.

**ILU-RMU** is the only fully validated refusal row: ablating its selected
direction collapses refusal 98%&rarr;0% with 100% clean bypass (hand-checked
on 20 baseline + 20 ablated generations; labels hold), and matched-control
ablation leaves refusal unchanged -- yet WMDP-Bio accuracy stays within noise
of the random-control distribution. That is the central dissociation in this
arm: **the direction demonstrably gates refusal-like behavior, but not access
to the forgotten bio knowledge.** Suppressing surface refusal and suppressing
the underlying knowledge do not appear to share a mechanism exposed by a
single linear direction. GradDiff's refusal-rate collapse is real but does
not earn the same "clean behavioral effect" claim; treat it as supporting
color, not a second fully validated within-model dissociation.

### A differently-built direction: junk-direction ablation recovers RMU and ILU-RMU cleanly

The refusal direction above is selected to gate refusal-*behavior* (COSMIC /
mean-diff over harmful-vs-harmless prompt pairs). `junk_direction_ablation/`
tests a differently-constructed, more targeted direction instead: for each
forget-domain corpus (WMDP bio-forget, cyber-forget), extract
`û = normalize(mean(a_unlearned) - mean(a_base))` -- the residual-stream
vector separating the unlearned model's activations from the base model's,
pooled over the last token, directly on forget-domain text (not on refusal
prompts at all). This targets whatever RMU-family training actually did to
forget-domain representations, rather than a generic refusal signal.

Result at layer 7, last-token pooling (`outputs/junk_direction/eval/{rmu,ilu-rmu}_layer7_last_token/`,
n = 1273, rank-2 ablation of the bio+cyber û's together):

| model | condition | WMDP-Bio acc | vs. junk (McNemar exact p) |
|---|---|---|---|
| RMU | baseline | 28.0% | p = 6.7e-49 |
| RMU | matched-control (retain-domain û) | 28.6% | p = 1.2e-46 |
| RMU | random-direction controls (8x, rank-2 matched) | 27.9% (±0.4%) | all p < 1e-46 |
| RMU | **junk_direction_ablation** | **53.9%** | -- |
| RMU | base model (full-knowledge reference) | 73.2% | -- |
| ILU-RMU | baseline | 34.3% | p = 1.8e-37 |
| ILU-RMU | matched-control (retain-domain û) | 37.8% | p = 1.2e-28 |
| ILU-RMU | random-direction controls (8x, rank-2 matched) | 34.5% (±0.7%) | all p < 1e-27 |
| ILU-RMU | **junk_direction_ablation** | **57.9%** | -- |
| ILU-RMU | base model (full-knowledge reference) | 73.2% | -- |

junk_control_z = 59-61 for RMU, ~32 for ILU-RMU (junk accuracy vs. each
model's own random-control distribution) -- not remotely comparable to the
noise-level deviations in the refusal-direction table above. MMLU stays flat
for both (RMU 62.0%&rarr;63.6%, +1.6pp; ILU-RMU 62.3%&rarr;62.3%, -0.04pp), so
this isn't general capability damage being repaired. Recovered accuracy is
57% of the RMU-to-base-model gap and 61% of the ILU-RMU-to-base-model gap. A
4-variant smoke sweep for RMU (layers 6-7 x last-token/mean-over-positions,
`outputs/junk_direction/smoke/rmu/`) found the same clear separation
from baseline/random at every variant tested (43.8%-54.7%), so this isn't a
one-off lucky pick for RMU at least.

This directly complicates the "no recovery beyond chance" reading of the
refusal-direction arm above, **for both RMU and ILU-RMU**: a direction built
to target the forget-domain representational shift RMU-family training
induces recovers most of the way to full-knowledge performance, even though a
direction built to target refusal behavior (same models, same task, same eval
harness) does not. Read together with the SFT-recovery arm below -- where RMU
and ILU-RMU also show the cleanest suppression-not-removal signatures of all
six methods -- this is a second, independent probe pointing at the same
conclusion: for these two methods specifically, the bio knowledge is
suppressed, not removed, and is reachable by an intervention that targets the
right subspace.

**Qualitative check, both models -- manually labeled (P0c protocol).**
Pilot n=15 free-form completions per junk-ablated model
(`outputs/junk_direction/eval/{rmu,ilu-rmu}_layer7_last_token/p0c_manual_labels.json`)
looked like they told different stories (RMU cleaner; ILU more contradictory),
but Fisher exact on that split was noise (genuine p≈0.47; contradictory
p≈0.26). Those pilots are retained only as directional history:

| model (pilot n=15) | genuine | format-artifact | contradictory |
|---|---|---|---|
| RMU (junk-direction) | 9/15 (60%) | 2/15 (13%) | 4/15 (27%) |
| ILU-RMU (junk-direction) | 6/15 (40%) | 1/15 (7%) | 8/15 (53%) |
| RMU (SFT arm, for comparison) | ~2-3/10 (20-30%) | 7/10 (70%) | 1/10 (10%) |
| ILU-RMU (SFT arm, for comparison) | 5/15 (33%) | 9/15 (60%) | 1/15 (7%) |

**Blinded n=30×3 (pre-registered follow-up).** Pool 30 RMU junk flips + 30
ILU-RMU junk flips + 30 base-model forced-choice-correct free-text
completions (`Meta-Llama-3-8B-Instruct`), shuffle with identity sealed, label
under the same genuine / format-artifact / contradictory protocol, then
unblind (`outputs/junk_direction/blind_p0c_n30/`,
`blind_p0c_results.{json,md}`). Prior n=15 docs were reused inside each junk
arm; re-labels matched the pilot labels 30/30 (κ=1.0).

| arm | genuine | format-artifact | contradictory |
|---|---|---|---|
| RMU junk | 12/30 (40%) | 4/30 (13%) | 14/30 (47%) |
| ILU-RMU junk | 13/30 (43%) | 3/30 (10%) | 14/30 (47%) |
| Base (forced-choice correct) | 28/30 (93%) | 0/30 (0%) | 2/30 (7%) |

Fisher exact RMU vs ILU: genuine p=1.00; contradictory p=1.00. There is
**no detectable qualitative gap between RMU and ILU-RMU** at n=30 — the
pilot "two pictures" claim does not survive. Both junk arms sit at ~40%
verified genuine and ~47% contradictory in this flip-only sheet.

The base-model row is a useful free-text-vs-forced-choice floor: on questions
the base model already "knows" under loglikelihood scoring, free-text
contradiction is only **2/30 ≈ 6.7%** (Wilson 95% CI ~1.8–21%). That shows
~47% contradiction is not "just how free-text scoring always looks." **It is
not, however, the right null for whether junk ablation *adds* contradiction
on the docs it flips** — that comparison needs the same ablated model on
docs it does *not* flip. The next subsection runs that control; the load-bearing
claim about flip-specific confabulation is settled there, not here. What this
sheet alone supports: McNemar/z significance is real, but only ~40% of sampled
wrong→right flips look like explanation-verified recovery; ~47% are
confidently-wrong reasoning the forced-choice scorer happens to credit.
Junk-direction recovery still looks cleaner than RMU's SFT-arm read (mostly
format-artifact), but that is a within-RMU probe contrast, not an RMU-vs-ILU
story.

**A sharper control: is contradiction specific to the flip, or baseline to
the ablated model?** The base-model null above (6.7%) controls for the
free-text-vs-forced-choice gap in general, but it's a different model
(full-knowledge, never ablated). A tighter control asks the same question
on the *ablated* model itself: on docs where junk-direction ablation changes
nothing at all -- `baseline==True AND junk_direction_ablation==True`, both
conditions already correct -- how often does RMU's/ILU-RMU's own free text
still contradict the letter both conditions agree on?

An earlier draft compared the blinded flip-arm ~47% contradiction rate to a
~30% letter-proxy rate on 30 stable-correct docs per model. That was a
**protocol mismatch** (hand labels vs automatic first-stated-letter) and is
withdrawn. The fix: pool the 30+30 flipped docs with the 30+30
stable-correct docs, shuffle with identity sealed, and label all 120 under
the same genuine / format-artifact / contradictory protocol
(`outputs/junk_direction/blind_p0c_n30_stable/`,
`blind_p0c_results.{json,md}`). Flip-arm re-labels agree with the prior
blind sheet on 55/60 docs (disagreements are mostly contradictory →
format-artifact).

| arm (blinded n=30) | genuine | format-artifact | contradictory |
|---|---|---|---|
| RMU junk (flip) | 12/30 (40%) | 7/30 (23%) | 11/30 (37%) |
| ILU-RMU junk (flip) | 14/30 (47%) | 3/30 (10%) | 13/30 (43%) |
| RMU stable-correct | 19/30 (63%) | 0/30 (0%) | 11/30 (37%) |
| ILU-RMU stable-correct | 18/30 (60%) | 3/30 (10%) | 9/30 (30%) |

Same-protocol flip vs stable contradiction: RMU **37% vs 37%** (Fisher
p=1.00); ILU-RMU **43% vs 30%** (Fisher p=0.42). Neither gap is significant.
The letter proxy's ~30% was in the right ballpark for ILU-RMU's manual
stable rate and slightly low for RMU (37%), but the load-bearing claim was
never the absolute level — it was the **gap** to the flip arm. Under matched
blinding that gap collapses: RMU shows none; ILU-RMU's 13pp gap is noise at
n=30.

So the same-model read is sharper than the earlier proxy draft suggested.
These RMU-family checkpoints already contradict their own stably-correct
forced-choice credit ~30–37% of the time with no flip involved — well above
the base model's 6.7% floor — and the flip-arm contradiction rates in this
pooled sheet (37%/43%) are not detectably higher. The ~47% figure from the
first flip-only blind batch remains a valid within-batch description of that
sheet, but **do not subtract letter-proxy 30% from it**. Apples-to-apples,
most of the free-text/forced-choice disagreement on flipped docs looks like
baseline RMU-family confabulation rather than ablation-induced
confabulation on the docs it flips.

**Loss-based junk null (GradDiff / NPO / NPO-ILU / IDK-AP).** Pre-registered in
`junk_direction_ablation/PREDICTION_LOSS_BASED.md`: no recovery
beyond matched control at either diagnostic layer (divergence + norm-spike
from the layer-divergence probe), for any of the four. Trimmed protocol —
baseline / junk / matched / 3 randoms, `wmdp_bio` only, no MMLU, no free-text,
`last_token` pooling (`outputs/junk_direction_loss_based/`).

| model | layer (role) | baseline | junk | matched | junk−matched | McNemar p | random mean (±std) |
|---|---|---|---|---|---|---|---|
| GradDiff | 6 (div) | 26.6% | 26.9% | 26.3% | +0.6pp | 0.36 | 26.3% (±1.2%) |
| GradDiff | 30 (spike) | 26.6% | 24.4% | 26.2% | −1.7pp | 0.31 | 26.3% (±1.2%) |
| NPO\* | 31 (div) | 26.9% | 27.0% | 26.7% | +0.3pp | 0.72 | 26.7% (±0.2%) |
| NPO\* | 15 (spike) | 26.9% | 30.1% | 25.8% | +4.2pp | 0.014 | 26.7% (±0.2%) |
| NPO-ILU\* | 2 (div) | 27.1% | 27.0% | 26.5% | +0.6pp | 0.64 | 27.0% (±0.3%) |
| NPO-ILU\* | 30 (spike) | 27.1% | 29.1% | 27.0% | +2.1pp | 0.099 | 27.0% (±0.3%) |
| IDK-AP | 2 (div) | 35.0% | 36.2% | 34.2% | +2.0pp | 0.011 | 34.9% (±0.6%) |
| IDK-AP | 30 (spike) | 35.0% | 35.6% | 34.6% | +0.9pp | 0.45 | 34.9% (±0.6%) |

\*NPO and NPO-ILU are degenerate / uninterpretable on every prior arm in this
document — included so the panel is not cherry-picked; **read nothing into
either result.**

Pre-registered null holds: **no cell meets the promotion rule** (junk−matched
≳5pp **and** McNemar p&lt;0.05 **and** junk above rank-matched random mean).
Eight McNemar tests were run; a Bonferroni α = 0.05/8 = 0.00625 threshold
leaves **none** significant — including the two nominally small-p cells (NPO
layer 15 p=0.014; IDK-AP layer 2 p=0.011), which also sit below the 5pp bar
(and NPO is flagged degenerate anyway). GradDiff is flat at both layers. So
unlike RMU / ILU-RMU, extracting û at both layers the diagnostic nominated
for each loss-based method recovers nothing beyond matched control. That
contrast is the useful one: junk-direction recovery is **not** a generic
artifact of ablating any forget−base difference; it tracks the RMU-family
representational bump that these four methods do not share.

**Caveats, before this goes further:** (1) The full-run RMU/ILU-RMU variant
(`layer7_last_token`) was picked from a layer-divergence probe, not from the
RMU smoke sweep's own accuracy ranking -- the smoke actually favored
`layer6_mean_over_positions` (54.7%) numerically before being killed
mid-sweep, so that variant is worth a full run too (no smoke sweep has been
run for ILU-RMU at all). The loss-based null used `last_token` only at both
diagnostic layers (no pooling sweep). (2) blinded n=30 still has modest power
for small between-model gaps; the point estimate here is near-identical rates,
so the null finding is not an underpowered miss of a large effect. (3) the
persisted `lm_eval_summary.json` for the RMU full run has every condition/task
row duplicated (byte-identical values, consistent with the eval having been
invoked twice without dedup) -- cosmetic, doesn't change the statistics above
(recomputed independently here from the underlying per-doc-correctness files
and de-duplicated random-control values), but worth cleaning up before this
is relied on elsewhere. The ILU-RMU run does not have this duplication.

## SFT recovery: does unrelated fine-tuning undo unlearning?

A different mechanism, tested by [`wmdp_sft_recovery/`](wmdp_sft_recovery/):
unrelated QLoRA SFT on `openai/gsm8k` (near-zero mutual information with
WMDP-bio), applied to six WMDP-unlearned methods plus a full-knowledge
control (base `meta-llama/Meta-Llama-3-8B-Instruct`), evaluated at
0/1000/3000/6000 examples seen via `lm_eval` (`wmdp_bio`, `mmlu`, `gsm8k`).
Full table:
[`wmdp_sft_recovery/results/wmdp_sft_recovery/recovery_table.md`](wmdp_sft_recovery/results/wmdp_sft_recovery/recovery_table.md).

| model | WMDP-bio: baseline &rarr; peak | &Delta; acc | &Delta; correct-option prob | &Delta; MMLU (utility gate) |
|---|---|---|---|---|
| RMU | 28.1% &rarr; 69.5% | +41.4pp | +0.326 | -3.2pp |
| NPO | 26.8% &rarr; 49.3% | +22.5pp | +0.140 | -1.8pp |
| ILU-RMU | 34.0% &rarr; 57.8% | +23.8pp | +0.149 | -5.9pp |
| GradDiff | 26.4% &rarr; 54.1% | +27.7pp | +0.126 | **+29.8pp** |
| NPO-ILU | 27.2% &rarr; 44.3% | +17.1pp | +0.081 | +2.4pp |
| IDK-AP | 35.2% &rarr; 40.4% | +5.2pp | +0.021 | -6.2pp |
| Full-knowledge control | 73.1% &rarr; 71.5% (no rise) | -1.6 to -2.0pp | ~-0.06 | -4.2 to -5.8pp |

Reading rule: bio up with MMLU flat means real recovery (suppression, not
removal); bio and MMLU up together means general SFT churn, not hidden
knowledge resurfacing. That rule is necessary but not sufficient -- it's
argmax multiple-choice accuracy, and the qualitative check below found that
matters.

- **RMU** recovers essentially to full-knowledge-model territory with MMLU
  flat -- the cleanest suppression-not-removal signature in this arm by the
  numbers, and its correct-option-probability delta (+0.326) leads every
  other model by more than 2x. (The free-text read below shows this number
  is mostly, not entirely, genuine.)
- **NPO** and **ILU-RMU** show substantial, MMLU-flat recovery (+21-24pp on
  bio, MMLU within -6pp) by the same metric, though less completely than
  RMU -- see below for what the underlying generations actually look like.
- **GradDiff**'s bio jump is confounded: MMLU rises by nearly as much
  (+29.8pp, from a near-chance 24.9% baseline). This reads as SFT repairing a
  broadly degraded/incoherent baseline rather than bio-specific suppression --
  exactly the case the reading rule is designed to catch. The ablation arm
  above only partially validates GradDiff behaviorally (refusal 61%&rarr;0%
  but clean bypass only 30%, with 70% ablated generations still degenerate),
  so there is no tension to resolve: both arms see a broadly broken model,
  and neither gives GradDiff a clean within-model dissociation.
- **NPO-ILU** sits in between by the numbers: a bio rise (+17.1pp, still
  climbing at the last checkpoint tested) alongside a modest MMLU increase
  (+2.4pp) -- not the clean flat-utility signature of RMU/NPO/ILU-RMU, but
  far short of GradDiff's confound. Read this one with the most caution of
  all six: see below, its free-text generations are not just
  artifact-affected but wholly incoherent, which puts what this accuracy
  number is even measuring in question.
- **IDK-AP** shows only a small rise (+5.2pp, probability flat at +0.02) while
  MMLU *declines* -- no clean recovery signature. This project's own probes
  can't distinguish genuine removal from suppression they can't reach, so the
  fair reading is that IDK-AP resists the interventions that recover the
  others, not that it was "removed."
- The **full-knowledge control** stays flat to slightly declining on
  WMDP-bio under the same SFT recipe, confirming the rises above are
  attributable to the unlearning coming undone rather than GSM8K SFT
  manufacturing bio knowledge on its own.

Ranking by correct-option-probability delta -- the cleanest discriminator,
since it can rise before the argmax answer flips -- is roughly
**RMU > NPO &asymp; ILU-RMU > GradDiff (confounded) > NPO-ILU &gt;&gt; IDK-AP &asymp; control (flat)**.

### Qualitative verification: reading the flipped generations

Everything above is forced-choice argmax accuracy on `wmdp_bio` -- the model
never generates anything; `lm_eval` compares loglikelihoods over the four
answer letters. An accuracy rise only means the argmax flipped, not that the
model is reasoning correctly about the biology. `wmdp_sft_recovery/sample_flipped_generations.py`
checks this directly: for questions that flipped wrong-at-baseline &rarr;
right-after-SFT, it re-generates a **free-form** (not forced-choice)
completion from the SFT-recovered model and a human reads it. Samples were
read for all four models with an MMLU-flat-or-near-flat recovery signature,
classifying each into: **genuine** (the free text coherently reasons to the
correct option), **format-artifact** (degenerate/repetitive output that
happens to argmax-match the gold letter without real content), or
**contradictory** (the model's own reasoning names a *different* option than
the one the forced-choice scorer credited as correct).

| model | n read | genuine | format-artifact | contradictory |
|---|---|---|---|---|
| RMU | 10 | ~2-3 | 7 (`####`-looping) | 1 (doc 706) |
| ILU-RMU | 15 | 5 | 9 | 1 (doc 1167) |
| NPO | 15 | 7 | 5 | 3 (docs 31, 1055, 139) |
| NPO-ILU | 15 | 0 | 15 | 0\* |

\* Every one of NPO-ILU's 15 samples is unreadable -- not `####`-style
repetition of real words, but incoherent token salad (e.g. `"...constitclincoincoincoincoincoin..."`,
`"...survivablility reporter/reportamer survivabl survivabl..."`). This is
categorically different from the other three: there's no free-text content
to classify as genuine or contradictory because the model isn't producing
language at that checkpoint. It's consistent with the ablation arm's
behavioral check, where NPO-ILU showed ~100% degeneracy at baseline too.

Two things follow. First, **none of the four "recovering" models is clean**.
Even the best case (NPO, 7/15 genuine) has three sampled documents where the
model's own stated reasoning argues for a specific wrong option on a
question the scorer marked correct -- a stronger and more direct artifact
than RMU's single doc 706 (20% of NPO's 15-doc sample vs. RMU's 10%), and it
means some unknown share of NPO's full accuracy rise reflects the scorer
crediting an answer the model didn't actually reason its way to (these
samples are small, 10-15 docs per model, so treat the fractions as
directional, not precise). Roughly a third to two-thirds of flipped documents
across RMU/ILU-RMU/NPO
are format-artifact rather than reasoning, and the true "verified genuine
recovery" fraction is closer to a quarter to a half of the reported accuracy
delta for each of these three, not the whole of it. Second, **NPO-ILU's
+17.1pp is not interpretable as recovery at all** -- it's an argmax shift on
a model that cannot generate coherent text, and should be read as noise
until (if ever) shown otherwise.

None of this erases the finding -- RMU, ILU-RMU, and NPO still show a real,
above-noise rise in genuine, coherently-reasoned correct answers on
previously-missed questions, which a purely degenerate or purely
lucky-argmax model would not produce. But "recovery, verified on generated
explanations" is the accurate description, not the raw accuracy delta, and
the ordering by *verified* recovery likely compresses the gap between RMU
and the other two rather than preserving the 2x gap the probability-delta
ranking above suggests.

### Is `####`-looping a property of the GSM8K recipe itself, or specific to the damaged models?

The table above attributes 7/10 of RMU's format-artifact completions to
`####`-looping -- but those are free-form completions on **WMDP-bio**
questions from a model that was SFT'd on GSM8K, so it's worth asking whether
this GSM8K-shaped SFT recipe just makes any model loop on `####`, independent
of what it's being asked. If so, the RMU-specific "hollow recovery" reading
would need softening.

A free, zero-compute check first: the full-knowledge control's own `gsm8k`
score *falls* across its own GSM8K SFT checkpoints (74.0%&rarr;71.0%&rarr;62.5%&rarr;68.0%,
worst at checkpoint-examples-3000) -- backwards for a model being fine-tuned
directly on the distribution it's scored on. `lm_eval` reports this via two
filters, `exact_match,strict-match` (requires the literal `#### <number>`
ending) and `exact_match,flexible-extract` (takes the last number emitted) --
if the strict filter were choking on a formatting quirk while flexible still
credited correct reasoning, that would explain the drop as a parsing
artifact. They are **identical at every control checkpoint** in the persisted
`lm_eval_results.json` (e.g. 0.625/0.625 at step 3000), which already rules
that out.

To settle it, `wmdp_sft_recovery/sample_gsm8k_free_text.py` re-runs `lm_eval`'s
actual `gsm8k` task (same seed, same 200-doc cap, same no-chat-template
config that produced the scored rows) but keeps the raw per-doc generations
instead of discarding them, and dumps a random 20 of the 200 scored docs
(control, checkpoint-examples-3000 -- the worst point) for a hand read
(`results/wmdp_sft_recovery/control_gsm8k_step3000_free_text.jsonl`; this run
replicated 0.64/0.64, matching the original 0.625/0.625 within noise).

**None of the 20 show `####`-looping, repetition, or any other degenerate
output.** Every completion is coherent prose ending in a single, correctly
placed `#### N`. 14/20 are correct; the other 6 are genuine arithmetic/reasoning
mistakes -- e.g. dropping a term in a three-way subtraction (doc 28: `60-20=40`
instead of `60-20-15=25`), misapplying a percentage-of-percentage pension
formula (doc 62), losing track of remaining hours in a multi-leg rate problem
(doc 8: answers 275 instead of 45), and inventing a "day 2/day 3" split that
isn't in the problem (doc 151). So the control's own GSM8K dip is real, if
modest, arithmetic damage from the QLoRA nudge -- not a parsing artifact and
not degenerate generation.

That matters for reading the table above two ways. First, since the
*undamaged* model given the *identical* SFT recipe does not loop on `####`
anywhere in 20 samples, `####`-looping is not a generic side effect of
GSM8K-QLoRA-SFT in general -- it takes a model whose weights are already
perturbed (RMU's noise injection) to produce it. That points the format-artifact
rate in RMU's bio table back at RMU's own damaged state, not at the SFT recipe,
which is the reading the "hollow recovery" framing needs to hold up. Second,
it does **not** make RMU's recovery numbers look stronger than currently
claimed: this check was run on the control's in-domain GSM8K completions, not
on RMU's own GSM8K completions or its out-of-domain bio completions, so it
narrows what `####`-looping *isn't* (a universal recipe artifact) without
directly re-examining what it *is* for RMU specifically. RMU's own
GSM8K-checkpoint completions are the natural next check if this needs closing
further -- flagged as a possible follow-up, not run here.

## Takeaway

Two different probes of the same question -- did unlearning remove
knowledge, or just suppress access to it? -- point the same direction but at
different resolutions. Rank-1 *refusal*-direction ablation, even where it
demonstrably bypasses refusal behavior, finds no WMDP-Bio recovery beyond
chance for any of 6 methods — McNemar + bootstrap CIs bound the null to
≲1.4pp recovery on the interpretable GradDiff/ILU-RMU rows; the other four
rows are non-results by construction (refusal floor / degenerate /
non-bypass), not missing tests. But direction construction
matters: a differently-built direction that targets the forget-domain
representational shift directly (junk-direction ablation, above) recovers
both RMU and ILU-RMU to more than half the gap to full-knowledge performance
(57% and 61% respectively) with an overwhelming McNemar signal (p < 1e-27
against baseline, matched control, and every random control, for both models)
-- so "direction ablation finds nothing" is a property of the
refusal-direction construction on the methods tested, not of direction
ablation as a technique; RMU-family forgetting is reachable by *some* linear
intervention, just not the refusal-gating one. The blinded qualitative read
tempers this evenly across both models: only ~40–47% of sampled
wrong-to-right flips look like explanation-verified recovery (Fisher exact
p≈0.8 between RMU and ILU-RMU in the pooled sheet), with the rest
contradictory or format-artifact. The right baseline for that flip-arm
contradiction rate is not the full-knowledge base model (6.7%) and not a
letter-proxy on stable-correct docs, but the same ablated checkpoints on
docs ablation doesn't touch, labeled under the **same** blinded protocol:
there, RMU and ILU-RMU already contradict their own stably-correct answer
37% / 30% of the time, and flip-vs-stable gaps are null (RMU 37% vs 37%;
ILU-RMU 43% vs 30%, Fisher p=0.42). So these checkpoints are
self-contradictory a lot in general, and ablation's marginal contribution
to free-text contradiction on the docs it flips is not detectable at n=30
once the protocol is matched. Unrelated SFT, a much less targeted
intervention, does recover bio performance for most methods, gated by
whether utility (MMLU) moves with it *and* by whether the recovered answers
hold up as free-text explanations rather than argmax noise or
format-artifact repetition -- which, on inspection, only part of each
model's reported accuracy delta does. Low forget-set performance alone does
not distinguish genuine knowledge removal from suppression, and a null
result from one recovery probe (direction ablation) does not generalize to
another (fine-tuning) -- and, as the qualitative checks above show, a
positive result from a recovery probe doesn't automatically mean what its
headline number suggests either, in either direction ablation or SFT.
Catching that in one's own data, rather than reporting the accuracy jump and
stopping, is the more defensible standard this project tries to hold itself
to.

