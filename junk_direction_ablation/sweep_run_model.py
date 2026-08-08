"""Stage 1 sweep, step 2: one unlearned model, ONE load, all of its work.

For --model-label X this does, inside a single model load:

  1. Extraction: pooled activations (last_token + mean_over_positions, all 8
     grid layers) for forget_bio / forget_cyber / retain, one forward pass
     per domain. Combined with the cached base activations
     (sweep_extract_base_activations.py) to write all 16 x 3 = 48 direction
     files under <output-root>/directions/<model>/<domain>/layer{L}_{pos}.pt
     -- same payload format extract_junk_directions.py uses, so Stage 2 can
     reuse eval_junk_ablation_lm_eval.py unmodified.
  2. Screening: baseline (unablated) + 16 variants x {junk_direction_ablation,
     matched_control} = 33 wmdp_bio runs on the SAME fixed n=200 doc-id
     subset (shared across every model in the sweep), scored via lm_eval's
     low-level Instance API so the subset is exactly the 200 docs, not the
     first 200. Per-doc correctness is persisted for every run.

Padding side is flipped from left (extraction) to right (lm_eval scoring)
between the two phases on the SAME loaded model -- the weights don't care,
only tokenization does.

Usage:

    python sweep_run_model.py --model-label graddiff \\
      --output-root ../outputs/junk_direction_loss_based_sweep \\
      --hardware-profile manual --dtype bfloat16 \\
      --extract-batch-size 2 --eval-batch-size 4 --gpu-memory 22GiB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ablation_lib import (
    BASE_MODEL,
    MODELS,
    all_layer_all_token_ablation,
    apply_hardware_profile,
    load_model_and_tokenizer,
    release_memory,
    require_lm_eval,
    require_torch,
    set_seed,
    unload_model,
    write_json,
)
from eval_junk_ablation_lm_eval import resolve_junk_directions, resolve_retain_direction
from extract_junk_directions import ensure_probe_sets
from sweep_lib import (
    CHANCE_ACC,
    CHANCE_BAND,
    DEGENERATE_MODELS,
    FORGET_DOMAINS,
    HFLM,
    RETAIN_DOMAIN,
    SWEEP_DOMAINS,
    SWEEP_LAYERS,
    SWEEP_POSITIONS,
    TASK_NAME,
    build_doc_instances_all,
    build_task,
    collect_pooled_activations,
    filter_by_doc_ids,
    load_or_create_screen_doc_ids,
    load_pooled_cache,
    require_lm_eval_task_api,
    save_direction,
    save_pooled_cache,
    score_doc_subset,
    tokenize_chunks_left,
    variant_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-label", required=True, choices=list(MODELS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-chunks", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-text-chars", type=int, default=51)
    parser.add_argument("--extract-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", default=None)
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="manual")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Probe-set seed (must match the shared 150-chunk sets).")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--apply-chat-template",
        dest="apply_chat_template",
        action="store_true",
        default=False,
        help="Off by default -- ScaleAI RMU template is broken; matches the existing loss-based arm.",
    )
    parser.add_argument("--ablation-mode", choices=["full", "resid_pre_only"], default="full")
    parser.add_argument(
        "--allow-bio-only",
        action="store_true",
        help="Permit junk ablation with only u_bio if forget_cyber directions are missing (under-recovery confound).",
    )
    args = parser.parse_args()
    apply_hardware_profile(args)
    # extraction_phase() needs junk (forget_bio, forget_cyber) AND matched (retain)
    # directions, so the sweep always loads all three -- no CLI knob, unlike
    # sweep_extract_base_activations.py's more generic --domains.
    args.domain_labels = list(SWEEP_DOMAINS)
    if args.extract_batch_size is None:
        args.extract_batch_size = {"4090": 2, "a100": 4, "t4x2": 1, "manual": 2}[args.hardware_profile]
    if args.eval_batch_size is None:
        args.eval_batch_size = {"4090": "auto:2", "a100": "auto", "t4x2": "auto:1", "manual": "auto:2"}[
            args.hardware_profile
        ]
    return args


def extraction_phase(args, output_root: Path, model, tokenizer) -> None:
    probe_sets = ensure_probe_sets(args, output_root)
    if "forget_bio" not in probe_sets:
        raise RuntimeError("forget_bio probe set failed to load -- cannot extract junk directions.")
    if "forget_cyber" not in probe_sets and not args.allow_bio_only:
        raise RuntimeError(
            "forget_cyber probe set missing. Bio-only junk ablation under-recovers on multi-domain WMDP "
            "checkpoints. Pass --allow-bio-only to override deliberately."
        )
    probe_fingerprints = {
        domain: hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest() for domain, texts in probe_sets.items()
    }

    tokenizer.padding_side = "left"
    unl_pooled: Dict[str, Dict[int, Dict[str, "torch.Tensor"]]] = {}
    for domain, texts in probe_sets.items():
        fp = probe_fingerprints[domain]
        cached = load_pooled_cache(output_root, args.model_label, domain, SWEEP_LAYERS, SWEEP_POSITIONS, fp)
        if cached is not None:
            print(f"  reusing cached {args.model_label}/{domain} pooled activations")
            unl_pooled[domain] = cached
            continue
        print(
            f"[{time.strftime('%H:%M:%S')}] collecting {args.model_label}/{domain}: n={len(texts)} "
            f"layers={SWEEP_LAYERS} positions={SWEEP_POSITIONS}"
        )
        input_ids, attention_mask = tokenize_chunks_left(tokenizer, texts, args.max_length)
        acts = collect_pooled_activations(
            model, input_ids, attention_mask, SWEEP_LAYERS, SWEEP_POSITIONS, args.extract_batch_size
        )
        save_pooled_cache(output_root, args.model_label, domain, SWEEP_LAYERS, SWEEP_POSITIONS, fp, len(texts), acts)
        unl_pooled[domain] = acts
        del input_ids, attention_mask, acts
        release_memory()

    base_pooled: Dict[str, Dict[int, Dict[str, "torch.Tensor"]]] = {}
    for domain in probe_sets:
        fp = probe_fingerprints[domain]
        cached = load_pooled_cache(output_root, "base", domain, SWEEP_LAYERS, SWEEP_POSITIONS, fp)
        if cached is None:
            raise FileNotFoundError(
                f"Missing cached base pooled activations for domain={domain}. "
                "Run sweep_extract_base_activations.py first (it is shared across all four models)."
            )
        base_pooled[domain] = cached

    model_id = MODELS[args.model_label]
    for domain, texts in probe_sets.items():
        fp = probe_fingerprints[domain]
        for layer in SWEEP_LAYERS:
            for position in SWEEP_POSITIONS:
                out_path = (
                    output_root / "directions" / args.model_label / domain / f"{variant_name(layer, position)}.pt"
                )
                if out_path.exists():
                    continue
                save_direction(
                    output_root,
                    args.model_label,
                    model_id,
                    domain,
                    layer,
                    position,
                    unl_pooled[domain][layer][position],
                    base_pooled[domain][layer][position],
                    len(texts),
                    fp,
                )
    print(f"[{time.strftime('%H:%M:%S')}] directions written under {output_root / 'directions' / args.model_label}")


def load_direction_sets(
    args, output_root: Path, variant: str
) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
    directions_root = output_root / "directions"
    junk_dirs, _used = resolve_junk_directions(
        directions_root, args.model_label, variant, list(FORGET_DOMAINS), allow_bio_only=args.allow_bio_only
    )
    matched_dir = resolve_retain_direction(directions_root, args.model_label, variant, RETAIN_DOMAIN)
    return junk_dirs, [matched_dir]


def screening_phase(args, output_root: Path, model, tokenizer) -> List[dict]:
    require_lm_eval()
    require_lm_eval_task_api()
    tokenizer.padding_side = "right"

    task = build_task(TASK_NAME)
    by_doc_all = build_doc_instances_all(task, apply_chat_template=args.apply_chat_template)
    n_total = len(by_doc_all)
    doc_ids_path = output_root / "screen_doc_ids.json"
    doc_ids = load_or_create_screen_doc_ids(doc_ids_path, n_total)
    by_doc = filter_by_doc_ids(by_doc_all, doc_ids)
    print(f"[{time.strftime('%H:%M:%S')}] screen subset: n={len(by_doc)} of {n_total} total {TASK_NAME} docs")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.eval_batch_size)

    screen_root = output_root / "screen" / args.model_label
    correctness_root = screen_root / "per_doc_correctness"
    summary_path = screen_root / "summary.json"
    summary_rows: List[dict] = []
    if summary_path.exists():
        try:
            summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            summary_rows = []
    done = {
        (row["variant"], row["condition"])
        for row in summary_rows
        if row.get("doc_ids_path") == str(doc_ids_path) and Path(row.get("per_doc_correctness_path", "")).exists()
    }

    def run_condition(variant: str, condition: str, layer, position, directions) -> None:
        if (variant, condition) in done:
            print(f"  [{args.model_label}] {variant}/{condition} (resume skip: already on disk)")
            return
        with all_layer_all_token_ablation(model, directions, mode=args.ablation_mode):
            correctness, accuracy = score_doc_subset(lm, by_doc)
        out_path = correctness_root / variant / f"{condition}.json"
        write_json(out_path, correctness)
        row = {
            "model": MODELS[args.model_label],
            "model_label": args.model_label,
            "layer": layer,
            "position": position,
            "variant": variant,
            "condition": condition,
            "task": TASK_NAME,
            "accuracy": accuracy,
            "n": len(by_doc),
            "doc_ids_path": str(doc_ids_path),
            "per_doc_correctness_path": str(out_path),
            "degenerate": args.model_label in DEGENERATE_MODELS,
        }
        summary_rows[:] = [
            r for r in summary_rows if not (r["variant"] == variant and r["condition"] == condition)
        ]
        summary_rows.append(row)
        write_json(summary_path, summary_rows)
        print(f"  [{args.model_label}] {variant:<32} {condition:<24} acc={accuracy:.4f}")

    all_variants = [variant_name(layer, position) for layer in SWEEP_LAYERS for position in SWEEP_POSITIONS]
    run_baseline_once(args, correctness_root, all_variants, lm, model, by_doc, summary_rows, summary_path, doc_ids_path)

    for layer in SWEEP_LAYERS:
        for position in SWEEP_POSITIONS:
            variant = variant_name(layer, position)
            junk_dirs, matched_dirs = load_direction_sets(args, output_root, variant)
            run_condition(variant, "junk_direction_ablation", layer, position, junk_dirs)
            run_condition(variant, "matched_control", layer, position, matched_dirs)

    return summary_rows


def run_baseline_once(args, correctness_root: Path, all_variants: List[str], lm, model, by_doc, summary_rows, summary_path, doc_ids_path) -> None:
    """Baseline doesn't depend on (layer, pooling), so it is scored exactly
    ONCE, then a copy of the same per-doc correctness is written into every
    variant's folder -- matching the existing full-run convention (each
    variant folder is self-contained with its own baseline.json) so
    sweep_sanity_check.py and Stage 2 tooling don't need special-casing.
    Only one summary row is kept (the report script only needs one).
    """
    canonical_path = correctness_root / "baseline" / "baseline.json"
    all_paths_exist = canonical_path.exists() and all(
        (correctness_root / v / "baseline.json").exists() for v in all_variants
    )
    already_summarized = any(
        row.get("variant") == "baseline" and row.get("condition") == "baseline"
        and row.get("doc_ids_path") == str(doc_ids_path)
        for row in summary_rows
    )
    if all_paths_exist and already_summarized:
        print(f"  [{args.model_label}] baseline/baseline (resume skip: already on disk)")
        return
    with all_layer_all_token_ablation(model, None, mode=args.ablation_mode):
        correctness, accuracy = score_doc_subset(lm, by_doc)
    write_json(canonical_path, correctness)
    for v in all_variants:
        write_json(correctness_root / v / "baseline.json", correctness)
    row = {
        "model": MODELS[args.model_label],
        "model_label": args.model_label,
        "layer": None,
        "position": None,
        "variant": "baseline",
        "condition": "baseline",
        "task": TASK_NAME,
        "accuracy": accuracy,
        "n": len(by_doc),
        "doc_ids_path": str(doc_ids_path),
        "per_doc_correctness_path": str(canonical_path),
        "degenerate": args.model_label in DEGENERATE_MODELS,
    }
    summary_rows[:] = [
        r for r in summary_rows if not (r["variant"] == "baseline" and r["condition"] == "baseline")
    ]
    summary_rows.append(row)
    write_json(summary_path, summary_rows)
    print(f"  [{args.model_label}] {'baseline':<32} {'baseline':<24} acc={accuracy:.4f}")


def main() -> None:
    args = parse_args()
    require_torch()
    set_seed(args.seed)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / f"run_model_config_{args.model_label}.json", vars(args))

    if args.model_label in DEGENERATE_MODELS:
        print(f"NOTE: {args.model_label} is flagged degenerate -- include, report, do not interpret.")

    model_id = MODELS[args.model_label]
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model_label}: {model_id} (single load for extract+screen)")
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        tokenizer_id=BASE_MODEL,
        padding_side="left",
    )

    extraction_phase(args, output_root, model, tokenizer)
    summary_rows = screening_phase(args, output_root, model, tokenizer)

    model = unload_model(model)
    release_memory()

    n_flat_at_or_below_chance = sum(
        1 for r in summary_rows if r["condition"] == "baseline" and r["accuracy"] <= CHANCE_ACC + CHANCE_BAND
    )
    if n_flat_at_or_below_chance:
        print(f"NOTE: {args.model_label} baseline accuracy is at/below chance -- see FINDINGS.md flag.")

    print(f"[{time.strftime('%H:%M:%S')}] wrote {len(summary_rows)} screen rows for {args.model_label}")


if __name__ == "__main__":
    main()
