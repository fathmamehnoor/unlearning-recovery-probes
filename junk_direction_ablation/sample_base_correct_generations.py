"""Sample free-text completions from the base model on forced-choice-correct docs.

Reference null for P0c contradiction rates: questions the base model already
gets right under lm_eval loglikelihood scoring, regenerated open-ended.

Usage:

    python sample_base_correct_generations.py \\
      --correctness local_outputs/.../per_doc_correctness/wmdp_bio/base_model.json \\
      --num-samples 30 \\
      --seed 20260801 \\
      --output-jsonl local_outputs/junk_direction/blind_p0c_n30/base_correct_generations.jsonl
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

from ablation_lib import (
    BASE_MODEL,
    apply_hardware_profile,
    load_model_and_tokenizer,
    require_torch,
    set_seed,
    unload_model,
    write_json,
)
from sample_flipped_generations import format_question, load_correctness, load_task_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correctness", required=True, help="base_model per-doc correctness JSON")
    parser.add_argument("--task", default="wmdp_bio")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="4090")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing sample_ids (skip doc_ids already in output JSONL).",
    )
    parser.add_argument(
        "--doc-ids-file",
        default="",
        help="Optional JSON list/{sample_ids:[...]} selecting which correct docs to generate.",
    )
    args = parser.parse_args()
    apply_hardware_profile(args)
    return args


def existing_doc_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(str(json.loads(line)["doc_id"]))
    return done


@torch.inference_mode()
def generate_plain(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_device = model.get_input_embeddings().weight.device
    encoded = tokenizer(prompt, return_tensors="pt").to(input_device)
    input_len = encoded["input_ids"].shape[1]
    prev = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    try:
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


def main() -> None:
    args = parse_args()
    require_torch()
    set_seed(args.seed)

    correctness = load_correctness(args.correctness)
    correct_ids = sorted(
        (k for k, v in correctness.items() if bool(v)),
        key=lambda k: (len(str(k)), str(k)),
    )
    if not correct_ids:
        raise RuntimeError(f"No correct docs in {args.correctness}")

    rng = random.Random(args.seed)
    if args.doc_ids_file:
        payload = json.loads(Path(args.doc_ids_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "sample_ids" in payload:
            sample_ids = [str(x) for x in payload["sample_ids"]]
        elif isinstance(payload, list):
            sample_ids = [str(x) for x in payload]
        else:
            raise ValueError(f"Unrecognized --doc-ids-file format: {args.doc_ids_file}")
        missing = [d for d in sample_ids if d not in set(correct_ids)]
        if missing:
            raise ValueError(f"doc_ids not base-correct: {missing[:10]}")
    else:
        sample_ids = (
            correct_ids if len(correct_ids) <= args.num_samples else rng.sample(correct_ids, args.num_samples)
        )
        sample_ids = [str(x) for x in sample_ids]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_doc_ids(out_path) if args.resume else set()
    todo = [d for d in sample_ids if d not in done]
    print(f"Base-correct available={len(correct_ids)}; target={len(sample_ids)}; todo={len(todo)}")
    if not todo:
        print("Nothing to generate.")
        write_json(
            out_path.with_suffix(".meta.json"),
            {
                "n_correct_available": len(correct_ids),
                "n_sampled": len(sample_ids),
                "sample_ids": sample_ids,
                "model_label": "base",
                "base_model": BASE_MODEL,
                "labeling_scheme": ["genuine", "format-artifact", "contradictory"],
            },
        )
        return

    docs = load_task_docs(args.task)
    model, tokenizer = load_model_and_tokenizer(
        BASE_MODEL,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        tokenizer_id=BASE_MODEL,
        padding_side="left",
    )

    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
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
            completion = generate_plain(model, tokenizer, formatted["prompt"], args.max_new_tokens)
            row = {
                "doc_id": str(doc_id),
                "question": formatted["question"],
                "choices": formatted["choices"],
                "gold_letter": formatted["gold_letter"],
                "generated_completion": completion,
                "model_label": "base",
                "variant": None,
                "junk_domains": None,
                "label": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  wrote doc_id={doc_id}")

    write_json(
        out_path.with_suffix(".meta.json"),
        {
            "n_correct_available": len(correct_ids),
            "n_sampled": len(sample_ids),
            "sample_ids": sample_ids,
            "model_label": "base",
            "base_model": BASE_MODEL,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "labeling_scheme": ["genuine", "format-artifact", "contradictory"],
        },
    )
    unload_model(model)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
