#!/usr/bin/env bash
# Orchestrate Step 3 for one model label (rmu | ilu-rmu).
# Usage: bash run_step3.sh rmu
set -euo pipefail

MODEL_LABEL="${1:?usage: run_step3.sh <rmu|ilu-rmu>}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
OUT="${OUT:-$REPO/outputs/junk_direction}"
PROFILE="${PROFILE:-manual}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM="${GPU_MEM:-22GiB}"
BATCH_EXTRACT="${BATCH_EXTRACT:-2}"
BATCH_EVAL="${BATCH_EVAL:-auto:2}"
N_CHUNKS="${N_CHUNKS:-150}"
SMOKE_LIMIT="${SMOKE_LIMIT:-64}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "=== PREDICTION.md must be signed before interpreting results ==="

echo "=== [1/5] extract directions ==="
# Do NOT pass --skip-cyber-if-missing by default: missing û_cyber under-recovers
# and makes weak results inconclusive. Set SKIP_CYBER=1 only as an explicit override.
EXTRACT_EXTRA=()
if [[ "${SKIP_CYBER:-0}" == "1" ]]; then
  EXTRACT_EXTRA+=(--skip-cyber-if-missing)
fi
EVAL_EXTRA=()
if [[ "${ALLOW_BIO_ONLY:-0}" == "1" ]]; then
  EVAL_EXTRA+=(--allow-bio-only)
fi

python extract_junk_directions.py \
  --output-root "$OUT" \
  --models "$MODEL_LABEL" \
  --n-chunks "$N_CHUNKS" \
  --max-length 512 \
  --hardware-profile "$PROFILE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_EXTRACT" \
  --gpu-memory "$GPU_MEM" \
  "${EXTRACT_EXTRA[@]}"

echo "=== [2/5] smoke sweep variants ==="
python eval_junk_ablation_lm_eval.py \
  --model-label "$MODEL_LABEL" \
  --directions-root "$OUT/directions" \
  --sweep-variants \
  --limit "$SMOKE_LIMIT" \
  --skip-base-model \
  --num-random-controls 2 \
  --hardware-profile "$PROFILE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_EVAL" \
  --gpu-memory "$GPU_MEM" \
  --output-root "$OUT/smoke/$MODEL_LABEL" \
  "${EVAL_EXTRA[@]}"

WINNER="$(python -c "import json; print(json.load(open('$OUT/smoke/$MODEL_LABEL/selected_variant.json'))['winner'])")"
echo "Smoke winner: $WINNER"
echo "Record this in PREDICTION.md before reading the full eval."

EVAL_ROOT="$OUT/eval/${MODEL_LABEL}_${WINNER}"
echo "=== [3/5] full eval on $WINNER ==="
python eval_junk_ablation_lm_eval.py \
  --model-label "$MODEL_LABEL" \
  --directions-root "$OUT/directions" \
  --variant "$WINNER" \
  --num-random-controls 8 \
  --hardware-profile "$PROFILE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_EVAL" \
  --gpu-memory "$GPU_MEM" \
  --output-root "$EVAL_ROOT" \
  "${EVAL_EXTRA[@]}"

REF_UNL="0.281"
if [[ "$MODEL_LABEL" == "ilu-rmu" ]]; then
  REF_UNL="0.340"
fi

echo "=== [4/5] stats ==="
python analyze_stats.py \
  --eval-root "$EVAL_ROOT" \
  --baseline-acc-reference "$REF_UNL" \
  --base-model-acc-reference 0.731

echo "=== [5/5] free-text flips ==="
python sample_flipped_generations.py \
  --model-label "$MODEL_LABEL" \
  --directions-root "$OUT/directions" \
  --variant "$WINNER" \
  --baseline-correctness "$EVAL_ROOT/per_doc_correctness/wmdp_bio/baseline.json" \
  --junk-correctness "$EVAL_ROOT/per_doc_correctness/wmdp_bio/junk_direction_ablation.json" \
  --num-samples 15 \
  --hardware-profile "$PROFILE" \
  --dtype "$DTYPE" \
  --gpu-memory "$GPU_MEM" \
  --output-jsonl "$EVAL_ROOT/flipped_generations.jsonl" \
  --label-template-path "$EVAL_ROOT/flipped_label_sheet.json"

echo "Done. Eval root: $EVAL_ROOT"
echo "Next: manually label flipped_generations.jsonl with genuine|format-artifact|contradictory"
