#!/usr/bin/env python
"""Read raw free-text GSM8K completions from an SFT checkpoint.

Why this exists: the full-knowledge-control's own GSM8K accuracy FALLS across
its own GSM8K SFT checkpoints (74.0% -> 71.0% -> 62.5% -> 68.0%, worst at
checkpoint-examples-3000) -- backwards for a model being fine-tuned directly on
the distribution it is scored on. The free, zero-compute check is to compare
lm_eval's `exact_match,strict-match` (requires the literal "#### <number>"
ending) against `exact_match,flexible-extract` (takes the last number in the
output) in the persisted lm_eval_results.json for each control checkpoint --
if flexible rises while strict falls, the model reasoned fine and just stopped
emitting the exact terminator, a formatting artifact, not real damage. That
check was already run by hand: the two metrics are IDENTICAL at every control
checkpoint (0/1000/3000/6000), which rules out the formatting-regex
explanation -- both metrics agree the score really fell.

That leaves reading raw completions to tell apart:
  (a) correct reasoning with a non-numeric/unparseable ending both regexes
      miss identically (still a measurement artifact, just not the strict-vs-
      flexible one),
  (b) "####"-looping or other repetition degeneracy (genuine SFT-induced
      degenerate generation), or
  (c) actually-wrong arithmetic (the SFT damaged the model -- the bad case).

Every arm's own "did the fine-tune take" check (RMU, IDK-AP, ILU-RMU, NPO,
GradDiff) reuses this exact GSM8K SFT recipe and eval config, so whatever this
reveals about the control generalizes to reading their gsm8k rows too.

This script re-runs lm_eval's own `gsm8k` task through the SAME model-loading
path as eval_recovery_lm_eval.py (so the LoRA loads identically) and the SAME
seeds/limit/fewshot/chat-template settings actually used to produce the scored
rows (see run_base_control.sh -> common.sh: seed 42, gsm8k capped at the first
200 test docs via --task-limits gsm8k:200, num_fewshot left at lm_eval's task
default, --no-chat-template). Unlike eval_recovery_lm_eval.py, which discards
`res["samples"]` after computing aggregate stats, this script keeps the raw
per-doc generations (`resps`) and dumps a random sample of them to a jsonl for
manual reading, together with each doc's strict-match / flexible-extract
correctness so you can read exactly the docs that were scored.

Usage (control, checkpoint-examples-3000 -- the worst point):
    python sample_gsm8k_free_text.py \\
      --model-name meta-llama/Meta-Llama-3-8B-Instruct \\
      --adapter-path results/wmdp_sft_recovery/adapters/base_llama3_8b_instruct_gsm8k6000_seed42/checkpoint-examples-3000 \\
      --num-samples 20 \\
      --output-jsonl results/wmdp_sft_recovery/control_gsm8k_step3000_free_text.jsonl

Usage (no-SFT baseline, for comparison -- omit --adapter-path):
    python sample_gsm8k_free_text.py \\
      --model-name meta-llama/Meta-Llama-3-8B-Instruct \\
      --num-samples 20 \\
      --output-jsonl results/wmdp_sft_recovery/control_gsm8k_step0_free_text.jsonl

Reusable for any other arm's checkpoint by pointing --model-name/--adapter-path
at that model's adapter (e.g. the RMU or IDK-AP GSM8K checkpoints) to check
whether the same "####"-looping shows up there too.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_recovery_lm_eval import build_model, require_lm_eval  # noqa: E402

try:
    from lm_eval import simple_evaluate
except ModuleNotFoundError:
    simple_evaluate = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Base model repo/path (same base the adapter was trained on top of).")
    parser.add_argument("--adapter-path", default=None, help="GSM8K LoRA checkpoint to load on top. Omit to read the no-SFT baseline.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--eval-limit", type=int, default=200,
                         help="Must match GSM8K_LIMIT in common.sh (the doc cap actually scored). "
                              "lm_eval's --limit takes the FIRST N docs of the task's doc iterator, not a random "
                              "sample -- this reproduces that same prefix so sampling below draws only from docs "
                              "that were actually scored.")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for choosing which of the --eval-limit scored docs to read (independent of the eval's own seed).")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Match eval_recovery_lm_eval.py's EVAL_LOAD_IN_4BIT for this run if it wasn't bf16-everywhere.")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=False)
    parser.add_argument("--eval-seed", type=int, default=42, help="Must match --seed used for the scored run (random/numpy/torch/fewshot seeds).")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def group_samples_by_doc(raw_samples: List[dict]) -> Dict[str, dict]:
    """lm_eval's log_samples emits one entry per (doc, filter) pair for
    generate_until tasks like gsm8k -- e.g. one entry tagged filter=strict-match
    and one tagged filter=flexible-extract per doc, sharing the same raw
    `resps` (filters only post-process the same generation differently).
    Merge those back into one record per doc so a human reads one completion
    with both filters' verdicts attached, instead of the same text twice."""
    grouped: Dict[str, dict] = {}
    for entry in raw_samples:
        doc_id = str(entry.get("doc_id", entry.get("doc", {}).get("idx", len(grouped))))
        rec = grouped.setdefault(doc_id, {"doc_id": doc_id, "doc": entry.get("doc"), "resps": entry.get("resps"),
                                           "target": entry.get("target"), "per_filter": {}, "raw_entries": []})
        rec["raw_entries"].append(entry)
        filter_name = entry.get("filter", entry.get("filter_name", f"entry_{len(rec['per_filter'])}"))
        filter_info = {k: v for k, v in entry.items() if k not in ("doc", "resps", "doc_id")}
        rec["per_filter"][filter_name] = filter_info
    return grouped


def main() -> None:
    args = parse_args()
    require_lm_eval()
    if simple_evaluate is None:
        raise ModuleNotFoundError("lm_eval is not installed -- pip install lm_eval on the GPU box first.")

    lm = build_model(args)
    print(f"Running lm_eval gsm8k (limit={args.eval_limit}, num_fewshot=default, "
          f"apply_chat_template={args.apply_chat_template}, seed={args.eval_seed}), log_samples=True ...")
    res = simple_evaluate(
        model=lm,
        tasks=[args.task],
        num_fewshot=None,
        limit=args.eval_limit,
        log_samples=True,
        apply_chat_template=args.apply_chat_template,
        random_seed=args.eval_seed,
        numpy_random_seed=args.eval_seed,
        torch_random_seed=args.eval_seed,
        fewshot_random_seed=args.eval_seed,
    )

    metrics = res["results"].get(args.task, {})
    print("Aggregate metrics for this run (should match/replicate the original scored row within noise):")
    for k, v in metrics.items():
        if "exact_match" in k:
            print(f"  {k}: {v}")

    raw_samples = res.get("samples", {}).get(args.task)
    if not raw_samples:
        raise RuntimeError(f"lm_eval returned no samples for task '{args.task}' -- log_samples may not have worked "
                            "with this lm_eval version; check res['samples'].keys() by hand.")
    grouped = group_samples_by_doc(raw_samples)
    print(f"{len(grouped)} scored docs recovered from this run's samples.")

    rng = random.Random(args.sample_seed)
    doc_ids = sorted(grouped.keys(), key=lambda k: (len(k), k))
    n = min(args.num_samples, len(doc_ids))
    sampled_ids = rng.sample(doc_ids, n)

    records = []
    for doc_id in sampled_ids:
        rec = grouped[doc_id]
        doc = rec.get("doc") or {}
        resps = rec.get("resps")
        completion = None
        if resps and isinstance(resps, list) and resps and isinstance(resps[0], list) and resps[0]:
            completion = resps[0][0]
        correctness = {}
        for filter_name, info in rec["per_filter"].items():
            for k, v in info.items():
                if k.startswith("exact_match"):
                    correctness[f"{filter_name}::{k}"] = v
        record = {
            "doc_id": doc_id,
            "question": doc.get("question"),
            "gold_answer": rec.get("target"),
            "generated_completion": completion,
            "correctness_by_filter": correctness,
        }
        if completion is None:
            record["_raw_entries_fallback"] = rec["raw_entries"]
        records.append(record)
        print("=" * 80)
        print(f"doc_id={doc_id}  correctness={correctness}")
        print(f"Q: {doc.get('question')}")
        print(f"gold: {rec.get('target')}")
        print("-" * 40)
        print(completion if completion is not None else "(could not extract raw completion -- see _raw_entries_fallback in the jsonl)")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {len(records)} completions to {output_path} -- read these by hand for: correct reasoning with an "
          "unparseable ending (measurement artifact), '####'-looping/repetition (degenerate generation), or wrong "
          "arithmetic (real damage).")


if __name__ == "__main__":
    main()
