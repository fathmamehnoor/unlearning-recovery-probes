"""lm_eval ablation arms for junk-direction recovery (Step 3c–3e).

Five arms per unlearned model:
  1. baseline
  2. junk_direction_ablation      -- û from forget corpus (bio, or bio+cyber)
  3. matched_control_ablation     -- û extracted the same way on retain/wikitext
  4. random_direction_ablation_i  -- 8 isotropic unit directions
  5. base_model                   -- optional reference on the same axis

Always runs wmdp_bio + mmlu with log_samples=True and persists per-doc
correctness for McNemar / paired bootstrap.

Usage (full eval on the winning variant):

    python eval_junk_ablation_lm_eval.py \\
      --model-label rmu \\
      --directions-root outputs/junk_direction/directions \\
      --variant layer7_mean_over_positions \\
      --output-root outputs/junk_direction/eval/rmu_layer7_mean_over_positions \\
      --hardware-profile 4090

Smoke-screen over all six variants (limit=64) to pick a winner:

    python eval_junk_ablation_lm_eval.py \\
      --model-label rmu \\
      --directions-root outputs/junk_direction/directions \\
      --sweep-variants \\
      --limit 64 \\
      --skip-base-model --num-random-controls 2 \\
      --output-root outputs/junk_direction/smoke/rmu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ablation_lib import (
    ACC_KEY,
    BASE_MODEL,
    DEFAULT_LAYERS,
    MODELS,
    POSITION_CONVENTIONS,
    all_layer_all_token_ablation,
    apply_hardware_profile,
    load_model_and_tokenizer,
    per_doc_correctness,
    random_orthogonal_basis,
    release_memory,
    require_lm_eval,
    require_torch,
    run_lm_eval,
    set_seed,
    unload_model,
    unit_vector,
    variant_name,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-label", required=True, choices=list(MODELS))
    parser.add_argument("--directions-root", required=True, help="Root containing <model>/<domain>/<variant>.pt")
    parser.add_argument("--variant", default="layer7_last_token", help="Extraction variant to evaluate.")
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants to evaluate into <output-root>/<variant>/ (no smoke pick).",
    )
    parser.add_argument(
        "--sweep-variants",
        action="store_true",
        help="Evaluate every layer×position variant (usually with --limit for a smoke pick).",
    )
    parser.add_argument("--forget-domains", default="forget_bio,forget_cyber",
                        help="Domains whose û are ablated together for the junk arm.")
    parser.add_argument("--retain-domain", default="retain")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", default="wmdp_bio,mmlu")
    parser.add_argument("--limit", type=int, default=None, help="lm_eval sample limit (smoke tests).")
    parser.add_argument(
        "--mmlu-limit",
        type=int,
        default=None,
        help="Per-subject lm_eval limit applied to 'mmlu' only (utility gate), independent of "
             "--limit / the wmdp_bio arm which stays full-size. Runs mmlu as a separate "
             "simple_evaluate call from the other tasks so the two limits don't collide "
             "(lm_eval's `limit` is otherwise global across the whole task list).",
    )
    parser.add_argument("--num-random-controls", type=int, default=8)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-base-model", action="store_true", help="Skip the optional base_model reference arm.")
    parser.add_argument("--skip-matched-control", action="store_true")
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="4090")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    # ScaleAI RMU chat template is broken; default OFF for all Step-3 Llama-3 WMDP ckpts.
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=False)
    parser.add_argument("--chat-template", dest="apply_chat_template", action="store_true")
    parser.add_argument("--no-chat-template", dest="apply_chat_template", action="store_false")
    parser.add_argument(
        "--ablation-mode",
        choices=["full", "resid_pre_only"],
        default="full",
        help="full = Arditi-style block+attn+MLP hooks; resid_pre_only if MMLU tanks under full.",
    )
    parser.add_argument(
        "--allow-bio-only",
        action="store_true",
        help="Permit junk ablation with only û_bio when forget_cyber directions are missing. "
             "Without this flag, missing cyber is a hard error (under-recovery confound).",
    )
    args = parser.parse_args()
    apply_hardware_profile(args)
    args.forget_domain_list = [d.strip() for d in args.forget_domains.split(",") if d.strip()]
    args.task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    return args


def load_direction_file(path: Path) -> "torch.Tensor":
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "direction" not in payload:
            raise ValueError(f"{path} missing 'direction' key; keys={list(payload.keys())}")
        return unit_vector(payload["direction"])
    return unit_vector(payload)


def resolve_junk_directions(
    directions_root: Path,
    model_label: str,
    variant: str,
    forget_domains: Sequence[str],
    allow_bio_only: bool,
) -> Tuple[List["torch.Tensor"], List[str]]:
    dirs: List["torch.Tensor"] = []
    used: List[str] = []
    missing = []
    for domain in forget_domains:
        path = directions_root / model_label / domain / f"{variant}.pt"
        if not path.exists():
            missing.append(domain)
            continue
        dirs.append(load_direction_file(path))
        used.append(domain)
    if not dirs:
        raise FileNotFoundError(
            f"No junk directions found under {directions_root / model_label} for variant={variant} "
            f"domains={list(forget_domains)}"
        )
    if missing:
        msg = (
            f"Missing junk directions for domains {missing} (variant={variant}). "
            "Ablating a subset under-recovers on multi-domain WMDP checkpoints and makes "
            "a weak result inconclusive."
        )
        if allow_bio_only and used == ["forget_bio"] and set(missing) <= {"forget_cyber"}:
            print("WARNING: " + msg + " Continuing because --allow-bio-only was set.")
        else:
            raise FileNotFoundError(msg + " Pass --allow-bio-only to override for bio-only runs.")
    return dirs, used


def resolve_retain_direction(
    directions_root: Path,
    model_label: str,
    variant: str,
    retain_domain: str,
) -> "torch.Tensor":
    path = directions_root / model_label / retain_domain / f"{variant}.pt"
    return load_direction_file(path)


def list_variants(directions_root: Path, model_label: str, forget_domains: Sequence[str]) -> List[str]:
    # Prefer variants present for forget_bio; fall back to any forget domain.
    for domain in forget_domains:
        folder = directions_root / model_label / domain
        if folder.exists():
            names = sorted(p.stem for p in folder.glob("*.pt"))
            if names:
                return names
    # Construct the canonical six if nothing on disk yet.
    return [variant_name(layer, pos) for layer in DEFAULT_LAYERS for pos in POSITION_CONVENTIONS]


def run_tasks_split(model, tokenizer, args: argparse.Namespace) -> Tuple[dict, dict]:
    """Run args.task_list, honoring --mmlu-limit as an independent cap on 'mmlu' only.

    lm_eval's `limit` applies to every task in a single simple_evaluate() call, so
    a naive shared --limit would also truncate wmdp_bio (the arm actually being
    tested). When --mmlu-limit is set and 'mmlu' is in the task list, mmlu is run
    as its own simple_evaluate() call with that limit; everything else runs
    together under --limit (None = full size).
    """
    mmlu_limit = getattr(args, "mmlu_limit", None)
    if mmlu_limit is None or "mmlu" not in args.task_list:
        return run_lm_eval(
            model, tokenizer, tasks=args.task_list, batch_size=args.batch_size,
            apply_chat_template=args.apply_chat_template, seed=args.seed, limit=args.limit,
        )
    other_tasks = [t for t in args.task_list if t != "mmlu"]
    all_results: dict = {}
    all_samples: dict = {}
    if other_tasks:
        r, s = run_lm_eval(
            model, tokenizer, tasks=other_tasks, batch_size=args.batch_size,
            apply_chat_template=args.apply_chat_template, seed=args.seed, limit=args.limit,
        )
        all_results.update(r)
        all_samples.update(s)
    r, s = run_lm_eval(
        model, tokenizer, tasks=["mmlu"], batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template, seed=args.seed, limit=mmlu_limit,
    )
    all_results.update(r)
    all_samples.update(s)
    return all_results, all_samples


def evaluate_conditions(
    model,
    tokenizer,
    conditions: List[Tuple[str, Optional[Sequence["torch.Tensor"]]]],
    args: argparse.Namespace,
    output_root: Path,
    model_id: str,
    extra_meta: Optional[dict] = None,
) -> List[dict]:
    summary_rows: List[dict] = []
    all_results: dict = {}
    summary_path = output_root / "lm_eval_summary.json"
    results_path = output_root / "lm_eval_results.json"
    if summary_path.exists():
        try:
            summary_rows = json.loads(summary_path.read_text())
            if not isinstance(summary_rows, list):
                summary_rows = []
        except Exception:
            summary_rows = []
    if results_path.exists():
        try:
            loaded = json.loads(results_path.read_text())
            if isinstance(loaded, dict):
                all_results = loaded
        except Exception:
            all_results = {}

    done_conditions = {
        row.get("condition")
        for row in summary_rows
        if row.get("task") == "wmdp_bio" and (output_root / "per_doc_correctness" / "wmdp_bio" / f"{row.get('condition')}.json").exists()
    }

    for condition, directions in conditions:
        if condition in done_conditions:
            print(f"[{time.strftime('%H:%M:%S')}] condition={condition} (resume skip: already on disk)")
            continue
        print(f"[{time.strftime('%H:%M:%S')}] condition={condition}")
        with all_layer_all_token_ablation(model, directions, mode=args.ablation_mode):
            task_results, task_samples = run_tasks_split(model, tokenizer, args)
        all_results[condition] = task_results
        for task_name, samples in task_samples.items():
            correctness = per_doc_correctness(samples)
            write_json(output_root / "per_doc_correctness" / task_name / f"{condition}.json", correctness)
        for task, metrics in task_results.items():
            row = {
                "model": model_id,
                "model_label": args.model_label,
                "condition": condition,
                "task": task,
                "variant": getattr(args, "active_variant", args.variant),
            }
            if extra_meta:
                row.update(extra_meta)
            row.update({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})
            summary_rows.append(row)
            acc = row.get(ACC_KEY)
            print(f"  {condition:<32} {task:<16} acc={acc}")
        write_json(output_root / "lm_eval_results.json", all_results)
        write_json(output_root / "lm_eval_summary.json", summary_rows)
        release_memory()
    return summary_rows


def summarize_random_controls(summary_rows: List[dict], junk_condition: str = "junk_direction_ablation") -> List[dict]:
    by_task: Dict[str, dict] = {}
    for row in summary_rows:
        bucket = by_task.setdefault(row["task"], {"random": [], "junk": None})
        if str(row["condition"]).startswith("random_direction_ablation_"):
            bucket["random"].append(row)
        elif row["condition"] == junk_condition:
            bucket["junk"] = row
    out = []
    for task, bucket in by_task.items():
        accs = [float(r[ACC_KEY]) for r in bucket["random"] if ACC_KEY in r]
        if not accs:
            continue
        arr = np.array(accs, dtype=float)
        row = {
            "condition": "random_direction_ablation_summary",
            "task": task,
            "n_random_controls": int(arr.size),
            "random_acc_mean": float(arr.mean()),
            "random_acc_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "random_acc_values": accs,
        }
        if bucket["junk"] is not None and ACC_KEY in bucket["junk"]:
            junk_acc = float(bucket["junk"][ACC_KEY])
            row["junk_acc"] = junk_acc
            row["junk_minus_random_mean"] = junk_acc - row["random_acc_mean"]
            row["junk_control_z"] = (
                (junk_acc - row["random_acc_mean"]) / row["random_acc_std"] if row["random_acc_std"] > 0 else None
            )
        out.append(row)
    return out


def run_one_variant(args: argparse.Namespace, variant: str, output_root: Path) -> List[dict]:
    args.active_variant = variant
    directions_root = Path(args.directions_root)
    junk_dirs, junk_domains_used = resolve_junk_directions(
        directions_root,
        args.model_label,
        variant,
        args.forget_domain_list,
        allow_bio_only=args.allow_bio_only,
    )
    retain_dir = None
    if not args.skip_matched_control:
        retain_dir = resolve_retain_direction(
            directions_root, args.model_label, variant, args.retain_domain
        )

    model_id = MODELS[args.model_label]
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model_label}: {model_id}")
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        tokenizer_id=BASE_MODEL,
        padding_side="right",  # lm_eval loglikelihoods
    )
    hidden_size = int(model.config.hidden_size)
    junk_rank = len(junk_dirs)

    conditions: List[Tuple[str, Optional[Sequence["torch.Tensor"]]]] = []
    if not args.skip_baseline:
        conditions.append(("baseline", None))
    conditions.append(("junk_direction_ablation", junk_dirs))
    if retain_dir is not None:
        # Matched text-distribution control (1D retain û). Rank is not matched when
        # junk_rank>1; rank-matched randoms below cover the dimensionality confound.
        if junk_rank > 1:
            print(
                f"WARNING: junk arm rank={junk_rank} but matched_control is 1D retain. "
                "A junk>matched gap can be partly dimensional; require junk to also beat "
                "rank-matched randoms before promoting a hit."
            )
        conditions.append(("matched_control_ablation", [retain_dir]))
    for idx in range(args.num_random_controls):
        # Rank-matched random subspace: same number of directions as junk arm.
        conditions.append(
            (
                f"random_direction_ablation_{idx}",
                random_orthogonal_basis(hidden_size, junk_rank, args.seed + 1000 + idx),
            )
        )

    meta = {
        "junk_domains_used": junk_domains_used,
        "n_junk_directions": junk_rank,
        "variant": variant,
        "ablation_mode": args.ablation_mode,
    }
    write_json(output_root / "arm_config.json", {**{k: v for k, v in vars(args).items() if k != "active_variant"}, **meta, "model_id": model_id})
    summary_rows = evaluate_conditions(model, tokenizer, conditions, args, output_root, model_id, extra_meta=meta)
    model = unload_model(model)
    tokenizer = None
    release_memory()

    if not args.skip_base_model:
        print(f"[{time.strftime('%H:%M:%S')}] loading base_model reference: {BASE_MODEL}")
        base_model, base_tok = load_model_and_tokenizer(
            BASE_MODEL,
            dtype=args.dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            gpu_memory=args.gpu_memory,
            cpu_memory=args.cpu_memory,
            padding_side="right",
        )
        base_rows = evaluate_conditions(
            base_model,
            base_tok,
            [("base_model", None)],
            args,
            output_root,
            BASE_MODEL,
            extra_meta={"variant": variant},
        )
        summary_rows.extend(base_rows)
        base_model = unload_model(base_model)
        base_tok = None
        release_memory()

    control_rows = summarize_random_controls(summary_rows)
    if control_rows:
        summary_rows = summary_rows + control_rows
    write_json(output_root / "lm_eval_summary.json", summary_rows)
    return summary_rows


def pick_winner_from_smoke(smoke_root: Path) -> Optional[str]:
    """Pick variant by recovery = junk_acc - baseline_acc on wmdp_bio.

    Requires junk to beat matched_control when that arm exists. Absolute junk_acc
    alone is the wrong objective (a damaged model can score high or low for the
    wrong reasons).
    """
    import json

    best_name = None
    best_recovery = float("-inf")
    records = []
    for summary_path in sorted(smoke_root.glob("*/lm_eval_summary.json")):
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        by_cond = {}
        for row in rows:
            if row.get("task") != "wmdp_bio":
                continue
            by_cond[row["condition"]] = row
        if "junk_direction_ablation" not in by_cond or "baseline" not in by_cond:
            continue
        junk_acc = float(by_cond["junk_direction_ablation"][ACC_KEY])
        base_acc = float(by_cond["baseline"][ACC_KEY])
        recovery = junk_acc - base_acc
        matched = by_cond.get("matched_control_ablation")
        beats_matched = True
        matched_acc = None
        if matched is not None and ACC_KEY in matched:
            matched_acc = float(matched[ACC_KEY])
            beats_matched = junk_acc > matched_acc
        variant = by_cond["junk_direction_ablation"].get("variant") or summary_path.parent.name
        records.append(
            {
                "variant": variant,
                "junk_acc": junk_acc,
                "baseline_acc": base_acc,
                "recovery": recovery,
                "matched_acc": matched_acc,
                "beats_matched": beats_matched,
            }
        )
        if not beats_matched:
            continue
        if recovery > best_recovery:
            best_recovery = recovery
            best_name = variant
    write_json(smoke_root / "smoke_selection_details.json", {"candidates": records, "winner": best_name})
    if best_name is None and records:
        # Fall back to max recovery even if matched control not beaten — but flag it.
        fallback = max(records, key=lambda r: r["recovery"])
        print(
            f"WARNING: no variant beat matched_control on smoke; "
            f"falling back to max-recovery variant={fallback['variant']} "
            f"(recovery={fallback['recovery']:.4f}). Treat as inconclusive until full eval."
        )
        return fallback["variant"]
    return best_name


def main() -> None:
    args = parse_args()
    require_torch()
    require_lm_eval()
    set_seed(args.seed)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", vars(args))

    if args.sweep_variants and args.variants:
        raise ValueError("Pass only one of --sweep-variants or --variants.")

    if args.sweep_variants:
        variants = list_variants(Path(args.directions_root), args.model_label, args.forget_domain_list)
        sweep_summary = []
        for variant in variants:
            variant_root = output_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            print(f"\n[{time.strftime('%H:%M:%S')}] === sweep variant {variant} ===")
            rows = run_one_variant(args, variant, variant_root)
            for row in rows:
                if row.get("condition") == "junk_direction_ablation" and "wmdp_bio" in str(row.get("task")):
                    sweep_summary.append(
                        {
                            "variant": variant,
                            "task": row["task"],
                            "acc": row.get(ACC_KEY),
                            "baseline_task_rows_note": "see per-variant lm_eval_summary.json",
                        }
                    )
        write_json(output_root / "sweep_summary.json", sweep_summary)
        winner = pick_winner_from_smoke(output_root)
        write_json(output_root / "selected_variant.json", {"winner": winner, "sweep_summary": sweep_summary})
        print(f"[{time.strftime('%H:%M:%S')}] smoke winner={winner}")
    elif args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        for variant in variants:
            variant_root = output_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            print(f"\n[{time.strftime('%H:%M:%S')}] === variant {variant} ===")
            run_one_variant(args, variant, variant_root)
    else:
        run_one_variant(args, args.variant, output_root)

    print(f"[{time.strftime('%H:%M:%S')}] wrote outputs to {output_root}")


if __name__ == "__main__":
    main()
