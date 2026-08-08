"""Cheap check: is the `@nate@nate@nate...` free-text collapse seen in
npo_ablated_degeneracy_check.py (and in the original
npo_degeneracy_domain_check.py before it) a prompt-format artifact -- every
check so far has used a plain "Question:/Answer:" prompt, never
tokenizer.apply_chat_template -- or a property of the checkpoint that
survives switching to its own chat template?

## Why this exists

The reader's obvious objection to the main result: maybe the plain-text
prompt format itself is what's broken (Llama-3-Instruct was RLHF'd on chat-
formatted data; feeding it a bare completion-style prompt is off-distribution
for an Instruct checkpoint), and `@nate` is a symptom of that mismatch, not
of anything ablation-relevant. Indirect evidence against this already exists
(MMLU under the same plain, no-template format scores ~53% for NPO --
reasonable, well above chance -- which is hard to reconcile with "the format
itself produces garbage"), but that's an inference from a different task
(loglikelihood-scored MCQ, not free generation) on a different observable.
This script gets the direct answer: generate with the template ON and read
what comes out.

## Two additions on top of the minimal version of this check

1. **Two models, not one.** `ablation_lib.DEGENERATE_MODELS = {"npo",
   "npo-ilu"}` -- both are flagged uninterpretable-by-prior-arm, not just
   NPO. Running only NPO here would let a chat-template finding get read as
   a property of one checkpoint when the claim on the table is broader.
   Runs both, one model load at a time (never two checkpoints resident at
   once), same 20 docs for each.
2. **Split 10 forget-domain (bio) / 10 retain-domain**, not 20 forget-domain.
   The main run's whole point was keeping a global-collapse hypothesis
   (checkpoint/format-wide, content-independent) distinguishable from a
   selective one (tied to what was actually unlearned) by including a
   retain-domain arm; dropping to bio-only here under the new format would
   quietly lose that distinction for the one variable (templating) this
   script is testing. Bio docs are the same `flipped` set's first 10 ids
   from the completed run's `doc_manifest.json` (already-verified real
   WMDP-bio content); retain docs are that same manifest's first 10 `retain`
   ids (wmdp-cyber/-chem, content NPO/NPO-ILU were never unlearned on) --
   reusing the persisted, already-correctness-diffed doc ids rather than
   resampling, so these are the exact same documents the main run already
   characterized under the plain format, just re-run under the template.

## Design choices carried over from this session's main check (each one
fixed a real bug found along the way -- see npo_ablated_degeneracy_check.py
and the FINDINGS.md write-up for how each was caught)

- **Tokenizer: BASE_MODEL (meta-llama/Meta-Llama-3-8B-Instruct), not the
  unlearned checkpoint's own.** NPO's own tokenizer ships `pad_token_id`
  outside the base model's vocab range, which silently corrupted an earlier
  padded-batch run in this project. This script sidesteps padding entirely
  (see below), so that specific failure mode can't recur here either way --
  but BASE_MODEL is used regardless, for the same "keep tokenization
  identical to every other eval in this project" reason
  (eval_junk_ablation_lm_eval.py's own convention), and because it's the
  tokenizer already confirmed (this session) to carry a complete, working
  Llama-3-Instruct chat template; an unlearned checkpoint's own repo isn't
  guaranteed to have re-saved one.
- **No batching -- one document at a time.** `degeneracy_domain_check.
  generate_batch` re-tokenizes its prompt strings with `tokenizer(...,
  add_special_tokens=True)`, which is correct for plain text (no BOS yet)
  but WRONG for a chat-templated string: `apply_chat_template` already
  emits `<|begin_of_text|>` itself, so routing its output through
  `generate_batch` would silently double the BOS token on every prompt --
  exactly the kind of single-line bug that would make this check's answer
  meaningless without ever raising an error. Tokenizing via
  `apply_chat_template(..., tokenize=True, return_tensors="pt")` directly
  and generating one sequence at a time sidesteps both that and the
  pad-token issue above (no padding is needed at batch size 1). n=40 total
  completions (2 models x 20 docs) at batch size 1 is still cheap -- this
  is explicitly not the main run's statistics-bearing pipeline.
- Greedy decoding (`do_sample=False, num_beams=1`), same `--max-new-tokens`
  (default 256, matching the main run) for every doc and every model.
- Message content is `degeneracy_domain_check.format_prompt`'s output,
  UNCHANGED (including its trailing "Answer:" line) wrapped as a single user
  turn with `add_generation_prompt=True`. Keeping the text identical to the
  plain-format condition and only changing how it's wrapped isolates
  templating as the one variable this script is testing -- not a rewritten,
  chat-native prompt that would reintroduce a content confound.

## What this script does NOT do

No degeneracy-rate statistics, no decision rule, no ablation arms. n=20 per
model is a "read it yourself" check, not a powered comparison -- the
deliverable is the raw completions (persisted, never discarded) plus the
automatic flags for a first pass, exactly as the main run's own docstring
insists: read a sample by hand before trusting any rate, and here there
isn't even a rate being trusted, just eyes on the text.

## Usage

    python chat_template_artifact_check.py \\
      --doc-manifest ../local_outputs/npo_ablated_degeneracy_check/L6_last_token/doc_manifest.json \\
      --output-dir ../local_outputs/npo_chat_template_artifact_check \\
      --hardware-profile manual --dtype bfloat16
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ablation_lib import (
    BASE_MODEL,
    MODELS,
    apply_hardware_profile,
    load_model_and_tokenizer,
    release_memory,
    unload_model,
    write_json,
)

_DEGEN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "npo_degeneracy_domain_check"
    / "degeneracy_domain_check.py"
)
if not _DEGEN_MODULE_PATH.exists():
    raise FileNotFoundError(f"Expected degeneracy_domain_check.py at {_DEGEN_MODULE_PATH}.")
_spec = importlib.util.spec_from_file_location("degeneracy_domain_check", _DEGEN_MODULE_PATH)
_degen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_degen)  # type: ignore[union-attr]

format_prompt = _degen.format_prompt
load_wmdp_bio_docs = _degen.load_wmdp_bio_docs
load_retain_docs = _degen.load_retain_docs
get_terminators = _degen.get_terminators
is_degenerate_legacy = _degen.is_degenerate_legacy
is_degenerate_strict = _degen.is_degenerate_strict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-labels", default="npo,npo-ilu",
                         help="Both are DEGENERATE_MODELS in ablation_lib.py -- keep both by default "
                              "so a finding here isn't a claim about one checkpoint.")
    parser.add_argument("--doc-manifest", required=True,
                         help="doc_manifest.json from the completed npo_ablated_degeneracy_check.py run -- "
                              "reuses its persisted flipped/retain doc ids for direct before/after "
                              "comparison against that run's plain-format completions on the same docs.")
    parser.add_argument("--retain-jsonl", default=_degen.RETAIN_JSONL_DEFAULT)
    parser.add_argument("--num-bio-docs", type=int, default=10)
    parser.add_argument("--num-retain-docs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="manual")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    apply_hardware_profile(args)
    args.model_label_list = [m.strip() for m in args.model_labels.split(",") if m.strip()]
    for m in args.model_label_list:
        if m not in MODELS:
            raise ValueError(f"Unknown model label {m!r}; choices are {list(MODELS)}")
    return args


def build_doc_items(args: argparse.Namespace) -> List[dict]:
    manifest = json.loads(Path(args.doc_manifest).read_text(encoding="utf-8"))
    bio_ids = sorted(manifest["ids"]["flipped"], key=int)[: args.num_bio_docs]
    retain_ids = list(manifest["ids"]["retain"])[: args.num_retain_docs]
    if len(bio_ids) < args.num_bio_docs:
        raise RuntimeError(f"Manifest has only {len(bio_ids)} flipped ids, need {args.num_bio_docs}.")
    if len(retain_ids) < args.num_retain_docs:
        raise RuntimeError(f"Manifest has only {len(retain_ids)} retain ids, need {args.num_retain_docs}.")

    bio_docs = load_wmdp_bio_docs()
    bio_by_id = {d["source_id"]: d for d in bio_docs}
    retain_pool = load_retain_docs(args.retain_jsonl)
    retain_by_id = {d["source_id"]: d for d in retain_pool}

    items: List[dict] = []
    for sid in bio_ids:
        if sid not in bio_by_id:
            raise RuntimeError(f"bio doc id {sid} from manifest not found in load_wmdp_bio_docs() -- "
                                "doc-id alignment with the main run may have diverged.")
        doc = bio_by_id[sid]
        f = format_prompt(doc["question"], doc["choices"], doc["answer"])
        if f is None:
            raise RuntimeError(f"format_prompt failed for bio doc {sid}")
        items.append({"domain": "bio", "source_id": sid, "question": doc["question"],
                      "choices": doc["choices"], **f})
    for sid in retain_ids:
        if sid not in retain_by_id:
            raise RuntimeError(f"retain doc id {sid} from manifest not found in load_retain_docs() -- "
                                "retain_test.jsonl may not match the one the main run used.")
        doc = retain_by_id[sid]
        f = format_prompt(doc["question"], doc["choices"], doc["answer"])
        if f is None:
            raise RuntimeError(f"format_prompt failed for retain doc {sid}")
        items.append({"domain": "retain", "source_id": sid, "question": doc["question"],
                      "choices": doc["choices"], **f})
    return items


def generate_one_chat(model, tokenizer, chat_input_ids, max_new_tokens: int, terminators: List[int]) -> str:
    input_device = model.get_input_embeddings().weight.device
    input_ids = chat_input_ids.to(input_device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
        )
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def run_model(model_label: str, items: List[dict], args: argparse.Namespace, output_dir: Path) -> List[dict]:
    model_id = MODELS[model_label]
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_label}: {model_id} (tokenizer: {BASE_MODEL})")
    model, tokenizer = load_model_and_tokenizer(
        model_id, dtype=args.dtype, device_map=args.device_map, trust_remote_code=args.trust_remote_code,
        gpu_memory=args.gpu_memory, cpu_memory=args.cpu_memory,
        tokenizer_id=BASE_MODEL, padding_side="left",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"tokenizer for {BASE_MODEL} has no chat_template -- this check needs one. "
            "Unexpected for an Instruct checkpoint; investigate before proceeding."
        )
    terminators = get_terminators(tokenizer)

    rows: List[dict] = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item["prompt"]}]
        # NOTE: apply_chat_template(tokenize=True, return_tensors="pt") returns a
        # BatchEncoding, not a bare tensor, on this transformers version -- indexing
        # it with [0] directly (as older API docs suggest) yields a tokenizers.Encoding
        # object, not a token-id row, and blows up in tokenizer.decode(). Pull
        # ["input_ids"] out explicitly; confirmed (this session) to give a [1, seq_len]
        # tensor with a single BOS and correct chat structure.
        chat_encoding = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        chat_input_ids = chat_encoding["input_ids"]
        chat_prompt_str = tokenizer.decode(chat_input_ids[0], skip_special_tokens=False)
        completion = generate_one_chat(model, tokenizer, chat_input_ids, args.max_new_tokens, terminators)
        row = {
            "model_label": model_label,
            "domain": item["domain"],
            "source_id": item["source_id"],
            "question": item["question"],
            "choices": item["choices"],
            "gold_letter": item["gold_letter"],
            "chat_prompt": chat_prompt_str,
            "completion": completion,
            "degenerate_legacy": is_degenerate_legacy(completion),
            "degenerate_strict": is_degenerate_strict(completion),
        }
        rows.append(row)
        print(f"  [{time.strftime('%H:%M:%S')}] {model_label} {item['domain']} {item['source_id']}: "
              f"{i + 1}/{len(items)}  legacy={row['degenerate_legacy']} strict={row['degenerate_strict']}  "
              f"completion[:60]={completion[:60]!r}")

    unload_model(model)
    release_memory()

    out_path = output_dir / f"{model_label}_chat_template_generations.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    for domain in ("bio", "retain"):
        d_rows = [r for r in rows if r["domain"] == domain]
        if not d_rows:
            continue
        strict = sum(r["degenerate_strict"] for r in d_rows)
        legacy = sum(r["degenerate_legacy"] for r in d_rows)
        unique = len({r["completion"] for r in d_rows})
        print(f"[{time.strftime('%H:%M:%S')}] {model_label}/{domain}: n={len(d_rows)} "
              f"strict={strict}/{len(d_rows)} legacy={legacy}/{len(d_rows)} unique_completions={unique}")
    write_json(output_dir / f"{model_label}_chat_template_summary.json", {
        "model_label": model_label, "n": n,
        "strict_degenerate": sum(r["degenerate_strict"] for r in rows),
        "legacy_degenerate": sum(r["degenerate_legacy"] for r in rows),
        "unique_completions": len({r["completion"] for r in rows}),
        "by_domain": {
            domain: {
                "n": len([r for r in rows if r["domain"] == domain]),
                "strict_degenerate": sum(r["degenerate_strict"] for r in rows if r["domain"] == domain),
                "legacy_degenerate": sum(r["degenerate_legacy"] for r in rows if r["domain"] == domain),
            }
            for domain in ("bio", "retain")
        },
    })
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] building doc items from {args.doc_manifest}")
    items = build_doc_items(args)
    bio_ids = [it["source_id"] for it in items if it["domain"] == "bio"]
    retain_ids = [it["source_id"] for it in items if it["domain"] == "retain"]
    print(f"[{time.strftime('%H:%M:%S')}] {len(items)} docs: bio={bio_ids} retain={retain_ids}")
    write_json(output_dir / "doc_ids.json", {"bio": bio_ids, "retain": retain_ids})

    all_rows: Dict[str, List[dict]] = {}
    for model_label in args.model_label_list:
        all_rows[model_label] = run_model(model_label, items, args, output_dir)

    print(f"\n[{time.strftime('%H:%M:%S')}] === summary (read the actual completions before trusting this) ===")
    for model_label, rows in all_rows.items():
        strict = sum(r["degenerate_strict"] for r in rows)
        unique = len({r["completion"] for r in rows})
        print(f"  {model_label:10s} n={len(rows)} strict_degenerate={strict}/{len(rows)} "
              f"unique_completions={unique}")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {output_dir}/<model>_chat_template_generations.jsonl -- "
          f"read every completion by hand, this is a {sum(len(r) for r in all_rows.values())}-row "
          "qualitative check, not a powered statistic.")


if __name__ == "__main__":
    main()
