"""Free-text check on WMDP-Bio docs that flipped wrong→right under junk ablation.

Generates open-ended completions from the ablated unlearned model on N flipped
stems. Manual three-way labels (genuine / format-artifact / contradictory)
match the SFT-arm standard in unlearning_direction_evaluation/results.md.

Usage:

    python sample_flipped_generations.py \\
      --model-label rmu \\
      --directions-root outputs/junk_direction/directions \\
      --variant layer7_mean_over_positions \\
      --baseline-correctness outputs/junk_direction/eval/rmu_.../per_doc_correctness/wmdp_bio/baseline.json \\
      --junk-correctness outputs/junk_direction/eval/rmu_.../per_doc_correctness/wmdp_bio/junk_direction_ablation.json \\
      --num-samples 15 \\
      --output-jsonl outputs/junk_direction/eval/rmu_.../flipped_generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from lm_eval.tasks import TaskManager, get_task_dict
except ModuleNotFoundError:
    TaskManager = None
    get_task_dict = None

from ablation_lib import (
    BASE_MODEL,
    MODELS,
    all_layer_all_token_ablation,
    apply_hardware_profile,
    load_model_and_tokenizer,
    require_torch,
    set_seed,
    unload_model,
    unit_vector,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-label", required=True, choices=list(MODELS))
    parser.add_argument("--directions-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--forget-domains", default="forget_bio,forget_cyber")
    parser.add_argument("--baseline-correctness", required=True)
    parser.add_argument("--junk-correctness", required=True)
    parser.add_argument("--task", default="wmdp_bio")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="4090")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--label-template-path",
        default="",
        help="Optional path to write an empty labeling sheet (doc_id -> label) for manual annotation.",
    )
    parser.add_argument(
        "--allow-bio-only",
        action="store_true",
        help="Allow generation with only û_bio if forget_cyber directions are missing.",
    )
    parser.add_argument(
        "--include-doc-ids",
        default="",
        help="Comma-separated doc_ids that must be included in the sample (e.g. prior labeled 15).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing output JSONL rows; only generate missing sample_ids.",
    )
    parser.add_argument(
        "--doc-ids-file",
        default="",
        help="Optional JSON list/file of doc_ids to generate (overrides random sample when set).",
    )
    args = parser.parse_args()
    apply_hardware_profile(args)
    args.forget_domain_list = [d.strip() for d in args.forget_domains.split(",") if d.strip()]
    return args


def load_correctness(path: str) -> Dict[str, bool]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_flipped_doc_ids(baseline: Dict[str, bool], recovered: Dict[str, bool]) -> List[str]:
    shared = sorted(set(baseline) & set(recovered), key=lambda k: (len(k), k))
    flipped = [k for k in shared if (baseline[k] is False) and (recovered[k] is True)]
    if not flipped:
        raise RuntimeError(
            "No doc_ids flipped wrong-at-baseline -> right-under-junk-ablation. "
            "Check correctness paths / that junk ablation actually recovered accuracy."
        )
    return flipped


def load_task_docs(task_name: str) -> List[dict]:
    """Load WMDP-style multiple-choice docs.

    Prefers lm_eval task docs when available; falls back to HuggingFace
    `cais/wmdp` for `wmdp_bio` / `wmdp_cyber` so generation works even when
    lm_eval is broken/uninstalled.
    """
    if TaskManager is not None and get_task_dict is not None:
        try:
            task_manager = TaskManager()
            task_dict = get_task_dict([task_name], task_manager)
            task = task_dict[task_name]
            if hasattr(task, "test_docs") and task.has_test_docs():
                return list(task.test_docs())
            if hasattr(task, "validation_docs") and task.has_validation_docs():
                return list(task.validation_docs())
        except Exception as exc:
            print(f"WARNING: lm_eval task load failed ({exc}); trying HF datasets fallback")

    name = task_name.replace("-", "_")
    if name in {"wmdp_bio", "wmdp_cyber"}:
        from datasets import load_dataset

        config = name.replace("_", "-")
        return list(load_dataset("cais/wmdp", config, split="test"))
    raise RuntimeError(f"Cannot load docs for task '{task_name}' without lm_eval.")


def format_question(doc: dict) -> Optional[dict]:
    question = doc.get("question")
    choices = doc.get("choices")
    answer = doc.get("answer")
    if question is None or choices is None or answer is None:
        return None
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    gold_letter = letters[answer] if isinstance(answer, int) and answer < len(letters) else str(answer)
    lines = [f"Question: {question}"]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Explain your reasoning, then give your final answer as a single letter.")
    lines.append("Answer:")
    return {"prompt": "\n".join(lines), "gold_letter": gold_letter, "question": question, "choices": choices}


def load_junk_directions(directions_root: Path, model_label: str, variant: str, domains: List[str], allow_bio_only: bool):
    dirs = []
    used = []
    missing = []
    for domain in domains:
        path = directions_root / model_label / domain / f"{variant}.pt"
        if not path.exists():
            missing.append(domain)
            continue
        payload = torch.load(path, map_location="cpu")
        dirs.append(unit_vector(payload["direction"] if isinstance(payload, dict) else payload))
        used.append(domain)
    if not dirs:
        raise FileNotFoundError(f"No junk directions for {model_label}/{variant}")
    if missing:
        msg = f"Missing junk directions for {missing}."
        if allow_bio_only and used == ["forget_bio"] and set(missing) <= {"forget_cyber"}:
            print("WARNING: " + msg + " Continuing with --allow-bio-only.")
        else:
            raise FileNotFoundError(msg + " Pass --allow-bio-only to override.")
    return dirs, used


@torch.inference_mode()
def generate_with_ablation(
    model, tokenizer, prompt: str, directions, max_new_tokens: int, mode: str = "full"
) -> str:
    input_device = model.get_input_embeddings().weight.device
    encoded = tokenizer(prompt, return_tensors="pt").to(input_device)
    input_len = encoded["input_ids"].shape[1]
    # Temporarily enable cache for generation.
    prev = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    try:
        with all_layer_all_token_ablation(model, directions, mode=mode):
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        model.config.use_cache = prev
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)


def _parse_include_ids(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_existing_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["doc_id"])] = row
    return rows


def choose_sample_ids(
    flipped: List[str],
    num_samples: int,
    seed: int,
    include_ids: List[str],
    doc_ids_file: str,
) -> List[str]:
    if doc_ids_file:
        payload = json.loads(Path(doc_ids_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "sample_ids" in payload:
            ids = [str(x) for x in payload["sample_ids"]]
        elif isinstance(payload, list):
            ids = [str(x) for x in payload]
        else:
            raise ValueError(f"Unrecognized doc-ids file format: {doc_ids_file}")
        missing = [d for d in ids if d not in set(flipped)]
        if missing:
            raise ValueError(f"doc_ids not in flipped set: {missing[:10]}")
        return ids

    flipped_set = set(flipped)
    chosen: List[str] = []
    for did in include_ids:
        if did not in flipped_set:
            raise ValueError(f"--include-doc-ids contains non-flipped doc_id={did}")
        if did not in chosen:
            chosen.append(did)
    need = num_samples - len(chosen)
    if need < 0:
        raise ValueError(f"include-doc-ids ({len(chosen)}) exceeds --num-samples ({num_samples})")
    remaining = [d for d in flipped if d not in set(chosen)]
    rng = random.Random(seed)
    if need:
        if need > len(remaining):
            raise ValueError(f"Need {need} more flips but only {len(remaining)} remain")
        chosen.extend(rng.sample(remaining, need))
    elif not chosen:
        chosen = flipped if len(flipped) <= num_samples else rng.sample(flipped, num_samples)
    return [str(x) for x in chosen]


def main() -> None:
    args = parse_args()
    require_torch()
    set_seed(args.seed)

    baseline = load_correctness(args.baseline_correctness)
    junk = load_correctness(args.junk_correctness)
    flipped = find_flipped_doc_ids(baseline, junk)
    include_ids = _parse_include_ids(args.include_doc_ids)
    sample_ids = choose_sample_ids(
        flipped, args.num_samples, args.seed, include_ids, args.doc_ids_file
    )
    print(f"Flipped docs available={len(flipped)}; sampling {len(sample_ids)}")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_rows(out_path) if args.resume else {}
    todo = [d for d in sample_ids if d not in existing]
    print(f"Already have={len(existing)}; todo={len(todo)}")

    docs = load_task_docs(args.task)
    directions, used_domains = load_junk_directions(
        Path(args.directions_root),
        args.model_label,
        args.variant,
        args.forget_domain_list,
        allow_bio_only=args.allow_bio_only,
    )
    print(f"Ablating domains={used_domains} variant={args.variant}")

    label_sheet = {}
    # Preserve prior rows in sample order when resuming.
    ordered_rows: List[dict] = []
    for doc_id in sample_ids:
        if doc_id in existing and doc_id not in todo:
            ordered_rows.append(existing[doc_id])

    if todo:
        model_id = MODELS[args.model_label]
        model, tokenizer = load_model_and_tokenizer(
            model_id,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            gpu_memory=args.gpu_memory,
            cpu_memory=args.cpu_memory,
            tokenizer_id=BASE_MODEL,
            padding_side="left",  # generation
        )
        for doc_id in todo:
            idx = int(doc_id) if str(doc_id).isdigit() else None
            if idx is None or idx >= len(docs):
                raise RuntimeError(
                    f"Cannot map doc_id={doc_id} into task docs (len={len(docs)}). "
                    "Silent skip would shrink n and make rates inconclusive."
                )
            formatted = format_question(docs[idx])
            if formatted is None:
                raise RuntimeError(f"format_question failed for doc_id={doc_id}")
            completion = generate_with_ablation(
                model,
                tokenizer,
                formatted["prompt"],
                directions,
                args.max_new_tokens,
                mode="full",
            )
            row = {
                "doc_id": str(doc_id),
                "question": formatted["question"],
                "choices": formatted["choices"],
                "gold_letter": formatted["gold_letter"],
                "generated_completion": completion,
                "model_label": args.model_label,
                "variant": args.variant,
                "junk_domains": used_domains,
                "label": existing.get(doc_id, {}).get("label"),  # preserve if any
            }
            ordered_rows.append(row)
            existing[doc_id] = row
            print(f"  wrote doc_id={doc_id}")
        unload_model(model)
    else:
        ordered_rows = [existing[d] for d in sample_ids if d in existing]

    # Rewrite full sample in stable order (resume-safe).
    with out_path.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            label_sheet[str(row["doc_id"])] = {
                "label": row.get("label"),
                "notes": "",
                "allowed_labels": ["genuine", "format-artifact", "contradictory"],
            }

    if args.label_template_path:
        write_json(Path(args.label_template_path), label_sheet)
    write_json(
        out_path.with_suffix(".meta.json"),
        {
            "n_flipped_available": len(flipped),
            "n_sampled": len(sample_ids),
            "sample_ids": sample_ids,
            "include_doc_ids": include_ids,
            "junk_domains": used_domains,
            "variant": args.variant,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "labeling_scheme": ["genuine", "format-artifact", "contradictory"],
        },
    )
    print(f"Wrote {out_path} ({len(ordered_rows)} rows)")


if __name__ == "__main__":
    main()
