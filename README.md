# Probing Knowledge Recovery in Unlearned Models

Does machine unlearning remove knowledge from a language model, or does it only suppress access to it?

This repository contains the code for three experiments that test whether knowledge can be recovered from unlearned models. All three experiments use the same six WMDP-unlearned Llama-3-8B-Instruct checkpoints: RMU, ILU-RMU, IDK-AP, GradDiff, NPO, and NPO-ILU.

The three recovery probes are:

1. **Refusal-direction ablation:** Does removing a direction that controls refusal behavior recover forgotten knowledge?
2. **Forget-set representation-direction ablation:** Does removing a direction constructed from the difference between the base and unlearned model on forget-set prompts recover knowledge?
3. **Unrelated supervised fine-tuning:** Does fine-tuning on unrelated data cause the model to recover knowledge that was removed by unlearning?

The experiments suggest that different recovery probes reveal different behavior. In particular, some checkpoints show increased WMDP-Bio accuracy after representation-direction ablation or unrelated fine-tuning, while refusal-direction ablation does not recover knowledge in the one checkpoint where the refusal direction could be cleanly identified.

The experiments also show that WMDP forced-choice accuracy is not enough to establish genuine knowledge recovery. Some of the apparent recovery comes from contradictory or degenerate generations.

See `results.md` for the full results and analysis.

## Results

### 1. Refusal-direction ablation

A clean refusal direction could only be extracted from the ILU-RMU checkpoint. Ablating this direction reduced the refusal rate from 98% to 0%, but did not recover WMDP-Bio accuracy.

The other checkpoints did not give a clean refusal direction. RMU only partially reduced refusal, IDK-AP already had almost no refusal behavior, and GradDiff, NPO, and NPO-ILU produced responses that were too degenerate to interpret reliably.

Across the six checkpoints, refusal-direction ablation did not recover WMDP-Bio accuracy beyond the matched and random controls.

See `direction_extraction.md` for the direction extraction and selection procedure.

### 2. Forget-set representation-direction ablation

This experiment follows the approach of [Arditi & Chughtai](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow). The direction is constructed from the mean activation difference between the base and unlearned models on forget-set prompts.

For the Llama-3-8B-Instruct checkpoints, this recovered a substantial part of the WMDP-Bio accuracy gap for RMU and ILU-RMU:

* **RMU:** 57% of the gap recovered
* **ILU-RMU:** 61% of the gap recovered
* **NPO:** 64% of the gap recovered, but the model's generations were highly degenerate
* **IDK-AP:** 19% of the gap recovered
* **GradDiff:** no confirmed recovery
* **NPO-ILU:** no confirmed recovery

The RMU and ILU-RMU results replicate the qualitative finding from previous work. The experiment also extends the test to the other unlearning methods.

The NPO result is difficult to interpret because the model produces incoherent text even when its WMDP accuracy increases.

See `junk_direction_ablation/README.md` for the full pipeline.

### 3. Unrelated supervised fine-tuning

The third experiment tests a different recovery mechanism. Instead of modifying an activation direction, the unlearned models are fine-tuned on GSM8K using QLoRA.

WMDP-Bio accuracy increased for five of the six methods.

For RMU, ILU-RMU, and NPO, WMDP-Bio accuracy increased while MMLU accuracy remained roughly flat. This is consistent with recovery that is specific to the forget set.

GradDiff was different. Its WMDP-Bio accuracy increased, but MMLU accuracy increased by a similar amount. This suggests that the WMDP improvement was caused by a general increase in model capability rather than recovery of the unlearned knowledge.

NPO-ILU also showed an increase in WMDP accuracy, but its generated responses were incoherent, making the result difficult to interpret as genuine recovery.

See `wmdp_sft_recovery/README.md` for the SFT setup and results.

### 4. Benchmark accuracy is not enough

The recovery experiments above use WMDP forced-choice accuracy. This measures the log-likelihood assigned to the four answer choices, but the model does not generate an explanation.

To check whether the apparent recovery corresponded to meaningful behavior, I generated free-text responses for questions whose answers changed after an intervention.

The responses were classified as:

* **Genuine recovery:** the reasoning supports the correct answer.
* **Format artifact:** the model selects the correct answer, but the generated text does not support it.
* **Contradictory reasoning:** the explanation contradicts the selected answer or is incoherent.

Only about 40 to 47% of the examined recovered responses showed genuine reasoning supporting the correct answer.

The qualitative evaluation also showed that some contradictions were already present in the unlearned checkpoints. This means that benchmark accuracy can change without the model demonstrating coherent knowledge of the answer.

This is particularly important for NPO and NPO-ILU, where free-text generation reveals degeneration that is not apparent from forced-choice accuracy alone.

See `results.md` for the qualitative results and examples.

## Repository structure

* `results.md` -- summary of the results from all three probes
* `direction_extraction.md` -- refusal-direction extraction, candidate sweep, and selection procedure
* `junk_direction_ablation/README.md` -- forget-set representation-direction extraction and evaluation
* `wmdp_sft_recovery/README.md` -- unrelated SFT recovery setup and evaluation
* `notebooks/` -- self-contained notebooks showing the main steps of each probe
* `scripts/` -- evaluation and analysis scripts
* `examples/` -- example scripts for running the full pipelines

The notebooks are intended to make the experiments easier to follow without having to trace the full script pipeline. Each notebook contains the main steps for that probe, including activation extraction, direction construction, ablation, SFT setup, and evaluation.

---

## Installation

```bash
pip install -r requirements.txt
```

The scripts accept either Hugging Face model IDs or local checkpoint paths.

---

## Model checkpoints

The experiments use six WMDP-unlearned checkpoints, all starting from `meta-llama/Meta-Llama-3-8B-Instruct`.

Five checkpoints are from [OPTML-Group](https://huggingface.co/OPTML-Group):

* GradDiff
* IDK-AP
* ILU-RMU
* NPO
* NPO-ILU

The RMU checkpoint is from [ScaleAI](https://huggingface.co/ScaleAI/mhj-llama3-8b-rmu).

The repository focuses on evaluating these checkpoints rather than reproducing the unlearning training procedures.

---

# Probe 1: Refusal-direction ablation

One possible explanation for knowledge suppression is that unlearning hides the knowledge behind refusal behavior.

[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) found that refusal behavior in language models can be mediated by a single direction in activation space. If the same mechanism explains the behavior of an unlearned model, removing the refusal direction should allow the model to access the forgotten knowledge.

The experiment therefore has two parts:

1. Extract and validate a refusal direction.
2. Ablate the direction and measure WMDP-Bio accuracy.

The direction is only used for the recovery experiment if it first passes a behavioral validation test.

## Extracting the refusal direction

`extract_refusal_direction.py` computes difference-in-means directions at every layer and at the last five post-instruction token positions.

The prompts come from the same general setup used by Arditi et al., with harmful AdvBench prompts and harmless Alpaca prompts. The direction is therefore a general refusal direction rather than a direction specifically constructed from WMDP-Bio.

The script supports two selection methods:

* `mean_diff` -- the causal selection procedure from Arditi et al.
* `cosmic` -- the concept-inversion selection procedure from COSMIC.

Both methods are run from the same candidate sweep.

```bash
python scripts/extract_refusal_direction.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --selection-method both \
  --output-root outputs/idk-ap/direction_extraction/generic
```

For each selection method, the script writes:

* `direction_{method}.pt` -- the selected direction
* `candidate_diagnostics.csv` -- diagnostics for every candidate
* `direction_{method}_matched_control.pt` -- a matched control direction constructed at the same layer and position

The diagnostics include bypass, induce, KL, and COSMIC scores.

The matched control tests whether ablating any direction constructed in the same way at the selected layer and position changes WMDP-Bio accuracy.

## Evaluate WMDP-Bio under ablation

`wmdp_bio_lm_eval_ablation.py` evaluates the selected direction using the same `lm_eval` WMDP-Bio task used for the baseline results.

```bash
python scripts/wmdp_bio_lm_eval_ablation.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --matched-control-direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic_matched_control.pt \
  --output-root outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic \
  --apply-chat-template
```

The evaluation compares:

* baseline
* selected-direction ablation
* matched-control ablation
* random-direction ablations

The results are saved in `wmdp_bio_lm_eval_summary.json`.

## Check whether the direction affects refusal

A null WMDP result is only useful if the direction actually changes the behavior it is supposed to control.

`wmdp_refusal_behavior_check.py` generates responses to 100 held-out AdvBench prompts and measures the refusal rate with and without ablation.

```bash
python scripts/wmdp_refusal_behavior_check.py \
  --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
  --model-label idk-ap \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --extraction-prompts-csv outputs/idk-ap/direction_extraction/generic/direction_prompts.csv \
  --matched-control-direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic_matched_control.pt \
  --output-root outputs/idk-ap/wmdp_refusal_behavior_check/generic_cosmic
```

The prompts used for this evaluation are separate from the prompts used to extract the direction.

## Check the ablation hook

`hook_activation_check.py` checks that the ablation hook actually changes the model's per-choice log-likelihoods.

```bash
python scripts/hook_activation_check.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --direction-path outputs/idk-ap/direction_extraction/generic/direction_cosmic.pt \
  --num-docs 3
```

This provides a basic sanity check that a null WMDP result is not caused by an inactive hook.

## Run the full pipeline

For one model, the full pipeline can be run with:

```bash
MODEL=OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \
MODEL_LABEL=idk-ap \
OUTPUT_ROOT=outputs/idk-ap \
HARDWARE_PROFILE=a100 \
bash examples/run_full_wmdp_bio_pipeline.sh
```

The pipeline should be run separately for each checkpoint.

Random-direction and matched-control conditions only need to be computed once for each model. Later runs can use `--num-random-controls 0`.

## Paired significance testing

All three evaluation pipelines save per-question correctness because every condition is evaluated on the same WMDP questions.

This allows the recovery conditions to be compared using paired tests rather than independent-sample tests.

`scripts/analyze_paired_recovery.py` compares two saved correctness files using an exact McNemar test and a paired bootstrap confidence interval.

```bash
python scripts/analyze_paired_recovery.py \
  --correctness-a outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/per_doc_correctness/wmdp_bio/selected_direction_ablation.json \
  --correctness-b outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/per_doc_correctness/wmdp_bio/matched_control_ablation.json \
  --label-a selected_direction \
  --label-b matched_control \
  --output-json outputs/idk-ap/wmdp_bio_lm_eval/generic_cosmic/mcnemar_selected_vs_matched_control.json
```

## Result

Across the six checkpoints, refusal-direction ablation did not recover WMDP-Bio accuracy beyond the matched or random controls.

ILU-RMU was the only checkpoint where a refusal direction could be cleanly validated. Ablating it reduced refusal from 98% to 0%, but WMDP-Bio accuracy did not increase.

GradDiff also showed a large reduction in refusal after ablation, but its generations were too degenerate to interpret the result as a clean behavioral bypass.

See `results.md` for the per-model results and statistical comparisons.

---

# Probe 2: Forget-set representation-direction ablation

The second probe uses a different type of direction.

Instead of extracting a general refusal direction, it constructs a direction directly from the difference between the base and unlearned models on forget-set prompts:

```text
u = normalize(mean(a_unlearned) - mean(a_base))
```

This follows the approach used by [Arditi & Chughtai](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow), who found substantial recovery after ablating a forget-domain representation direction in an RMU-unlearned Zephyr-7B model.

Here the same idea is tested on Llama-3-8B-Instruct and extended to the other unlearning methods.

```bash
cd junk_direction_ablation

bash run_step3.sh rmu
bash run_loss_based_junk_null.sh
```

The pipeline includes:

* direction extraction
* layer and pooling sweeps
* direction ablation
* matched and random controls
* WMDP-Bio evaluation
* qualitative evaluation of changed generations

The main result was:

* **RMU:** 57% gap recovery
* **ILU-RMU:** 61% gap recovery
* **NPO:** 64% gap recovery, but with highly degenerate generations
* **IDK-AP:** 19% gap recovery
* **GradDiff:** no confirmed recovery
* **NPO-ILU:** no confirmed recovery

The RMU and ILU-RMU results are consistent with the earlier finding from Arditi & Chughtai. The experiment also shows that the same construction can produce smaller or ambiguous effects for other unlearning methods.

See `junk_direction_ablation/README.md` for the full pipeline and `results.md` for the results.

---

# Probe 3: Unrelated SFT recovery

The third probe tests whether unrelated fine-tuning can recover forgotten knowledge.

The unlearned models are fine-tuned on GSM8K using QLoRA. GSM8K is unrelated to WMDP-Bio, so the experiment tests whether changing the model through unrelated training can restore its performance on the forget set.

Each unlearned model is evaluated at 1000, 3000, and 6000 training examples.

A full-knowledge base model is also fine-tuned with the same procedure. This provides a control for changes in WMDP-Bio accuracy caused by the fine-tuning process itself.

The models are evaluated on:

* `wmdp_bio` for recovery
* `mmlu` for general capability
* `gsm8k` for training-data uptake

```bash
cd wmdp_sft_recovery

bash run_base_control.sh
METHODS="RMU" bash run_unlearned.sh
python aggregate_table.py
```

WMDP-Bio accuracy increased for five of the six unlearned models.

For RMU, ILU-RMU, and NPO, WMDP-Bio accuracy increased while MMLU remained roughly flat. This is consistent with a recovery effect that is more specific to the forget set.

GradDiff showed a different pattern. Its WMDP-Bio accuracy increased, but MMLU increased by a similar amount. This makes it difficult to distinguish recovery from a general increase in model capability.

NPO-ILU also showed an increase in WMDP-Bio accuracy, but its generated text was incoherent.

See `wmdp_sft_recovery/README.md` for the full setup and `results.md` for the results.

---

# Why benchmark accuracy is not enough

All three probes use forced-choice WMDP-Bio accuracy. The model is scored on the likelihood it assigns to the four answer choices.

The model does not need to generate a coherent explanation.

This means that an increase in accuracy does not necessarily mean that the model has recovered the underlying knowledge.

To investigate this, the repository includes scripts that generate free-text responses for questions whose answers changed after an intervention.

The responses are classified as:

* **Genuine recovery:** the reasoning supports the correct answer.
* **Format artifact:** the model selects the correct answer, but the generated text does not support it.
* **Contradictory reasoning:** the generated reasoning contradicts the selected answer or is incoherent.

The same evaluation is also performed on questions that remain correct before and after the intervention. This helps determine how much of the contradictory behavior is caused by the intervention and how much is already present in the checkpoint.

Only about 40 to 47% of the examined changed responses showed genuine reasoning supporting the correct answer.

This qualitative evaluation is particularly important for NPO and NPO-ILU, where forced-choice accuracy can increase even though the model produces incoherent free-text responses.

The relevant scripts are:

```text
junk_direction_ablation/
    sample_flipped_generations.py
    sample_stable_correct_generations.py
    sample_base_correct_generations.py

wmdp_sft_recovery/
    sample_flipped_generations.py
```

See `results.md` for the qualitative results and the blinding procedure.

---

# References

* Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction.](https://arxiv.org/abs/2406.11717)
* Siu et al. (2025). [COSMIC: Generalized Refusal Direction Identification in LLM Activations.](https://arxiv.org/abs/2506.00085)
* Chughtai, B. (2024). [Unlearning via RMU is mostly shallow.](https://www.lesswrong.com/posts/6QYpXEscd8GuE7BgW/unlearning-via-rmu-is-mostly-shallow)
* Łucki et al. (2024). [An Adversarial Perspective on Machine Unlearning for AI Safety.](https://arxiv.org/abs/2409.18025)
* Hu et al. (2025). [Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning.](https://arxiv.org/html/2406.13356v1)
* Deeb & Roger (2024). [Do Unlearning Methods Remove Information from Language Model Weights?](https://arxiv.org/abs/2410.08827)
* WMDP. [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.](https://www.wmdp.ai/)
