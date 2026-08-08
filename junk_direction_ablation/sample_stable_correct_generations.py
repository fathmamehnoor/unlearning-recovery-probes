"""Non-flipped-correct control: junk-ablated free-text on always-correct docs.

Selection rule (hard):
  baseline correctness == True  AND  junk_direction_ablation correctness == True

These are docs the unlearned model already got right under forced-choice
*before* ablation and still gets right *under* junk ablation — i.e. not part
of the wrong→right flip set. Free-text is generated from the **junk-ablated**
unlearned model with the same prompt + ablation recipe as
`sample_flipped_generations.py`.

Interpretation (same P0c labels: genuine / format-artifact / contradictory):
  - If contradiction ≈ flip-arm rate (~47%): ablation globally decouples
    scoring from generation.
  - If contradiction ≈ base null (~7%): contradiction is specific to newly
    flipped docs; ~half the accuracy rise may be scorer credit without
    free-text agreement.

Usage:

    python sample_stable_correct_generations.py \\
      --model-label rmu \\
      --directions-root outputs/junk_direction/directions \\
      --variant layer7_last_token \\
      --baseline-correctness .../per_doc_correctness/wmdp_bio/baseline.json \\
      --junk-correctness .../per_doc_correctness/wmdp_bio/junk_direction_ablation.json \\
      --num-samples 30 \\
      --seed 20260802 \\
      --output-jsonl .../stable_correct_generations.jsonl

    # Selection-only dry run (no GPU):
    python sample_stable_correct_generations.py ... --plan-only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ablation_lib import (
    BASE_MODEL,
    MODELS,
    apply_hardware_profile,
    load_model_and_tokenizer,
    require_torch,
    set_seed,
    unload_model,
    write_json,
)
from sample_flipped_generations import (
    find_flipped_doc_ids,
    format_question,
    generate_with_ablation,
    load_correctness,
    load_junk_directions,
    load_task_docs,
)

SELECTION_RULE = "baseline==True AND junk_direction_ablation==True"
CONTROL_NAME = "stable_correct_junk_ablated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-label", choices=list(MODELS))
    parser.add_argument("--directions-root", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--forget-domains", default="forget_bio,forget_cyber")
    parser.add_argument("--baseline-correctness", default="")
    parser.add_argument("--junk-correctness", default="")
    parser.add_argument("--task", default="wmdp_bio")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260802,
        help="RNG seed for sampling stable-correct docs (default 20260802; distinct from flip/base 20260801).",
    )
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="4090")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument(
        "--allow-bio-only",
        action="store_true",
        help="Allow ablation with only û_bio if forget_cyber is missing (usually a confound — avoid).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing output JSONL rows; only generate missing sample_ids.",
    )
    parser.add_argument(
        "--doc-ids-file",
        default="",
        help="Optional JSON list/{sample_ids:[...]} — every id MUST satisfy the stable-correct rule.",
    )
    parser.add_argument(
        "--exclude-doc-ids-file",
        default="",
        help="Optional JSON list/{sample_ids:[...]} to exclude (e.g. other free-text samples).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Select sample_ids, write .meta.json + sample_ids sidecar, do not load the model.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run selection-logic unit tests and exit (no I/O / GPU).",
    )
    args = parser.parse_args()
    if args.self_test:
        return args
    missing = [
        name
        for name, val in [
            ("--model-label", args.model_label),
            ("--directions-root", args.directions_root),
            ("--variant", args.variant),
            ("--baseline-correctness", args.baseline_correctness),
            ("--junk-correctness", args.junk_correctness),
            ("--output-jsonl", args.output_jsonl),
        ]
        if not val
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    apply_hardware_profile(args)
    args.forget_domain_list = [d.strip() for d in args.forget_domains.split(",") if d.strip()]
    return args


def _as_bool(value: object, *, doc_id: str, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise TypeError(
        f"{source}[{doc_id!r}] has non-boolean correctness value {value!r} "
        f"(type={type(value).__name__}). Refusing to coerce."
    )


def find_stable_correct_doc_ids(
    baseline: Dict[str, object],
    junk: Dict[str, object],
) -> Tuple[List[str], dict]:
    """Return sorted doc_ids with baseline True and junk True.

    Also returns a diagnostics dict. Raises if the stable set intersects the
    flip set (should be impossible by construction).
    """
    shared = sorted(set(baseline) & set(junk), key=lambda k: (len(str(k)), str(k)))
    if not shared:
        raise RuntimeError("baseline and junk correctness JSON share zero doc_ids.")

    stable: List[str] = []
    flipped: List[str] = []
    for doc_id in shared:
        b = _as_bool(baseline[doc_id], doc_id=str(doc_id), source="baseline")
        j = _as_bool(junk[doc_id], doc_id=str(doc_id), source="junk")
        if b and j:
            stable.append(str(doc_id))
        elif (not b) and j:
            flipped.append(str(doc_id))

    overlap = sorted(set(stable) & set(flipped))
    if overlap:
        raise RuntimeError(
            f"Internal error: {len(overlap)} doc_ids are both stable-correct and flipped "
            f"(e.g. {overlap[:5]}). Selection logic is broken."
        )

    # Cross-check against the flip helper used by the recovery arm.
    flip_helper = find_flipped_doc_ids(
        {k: _as_bool(v, doc_id=str(k), source="baseline") for k, v in baseline.items() if k in junk},
        {k: _as_bool(v, doc_id=str(k), source="junk") for k, v in junk.items() if k in baseline},
    )
    if set(flip_helper) != set(flipped):
        raise RuntimeError(
            "Flip-set mismatch vs find_flipped_doc_ids "
            f"(local={len(flipped)} helper={len(flip_helper)}). Refusing to continue."
        )
    leak = sorted(set(stable) & set(flip_helper))
    if leak:
        raise RuntimeError(f"Stable set leaks into flip set: {leak[:10]}")

    diag = {
        "n_shared": len(shared),
        "n_baseline_correct": sum(
            1 for k in shared if _as_bool(baseline[k], doc_id=str(k), source="baseline")
        ),
        "n_junk_correct": sum(1 for k in shared if _as_bool(junk[k], doc_id=str(k), source="junk")),
        "n_stable_correct": len(stable),
        "n_flipped_wrong_to_right": len(flipped),
        "selection_rule": SELECTION_RULE,
    }
    if not stable:
        raise RuntimeError(
            "No stable-correct docs (baseline True ∧ junk True). "
            f"Diagnostics: {diag}"
        )
    return stable, diag


def _load_id_list(path: str) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "sample_ids" in payload:
        ids = [str(x) for x in payload["sample_ids"]]
    elif isinstance(payload, list):
        ids = [str(x) for x in payload]
    else:
        raise ValueError(f"Unrecognized id-list format in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate doc_ids in {path}")
    return ids


def choose_stable_sample_ids(
    stable: Sequence[str],
    num_samples: int,
    seed: int,
    doc_ids_file: str = "",
    exclude_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    stable_set = set(stable)
    exclude = {str(x) for x in (exclude_ids or [])}
    pool = [d for d in stable if d not in exclude]
    if not pool:
        raise RuntimeError("Stable-correct pool is empty after exclusions.")

    if doc_ids_file:
        ids = _load_id_list(doc_ids_file)
        bad = [d for d in ids if d not in stable_set]
        if bad:
            raise ValueError(
                f"--doc-ids-file contains {len(bad)} ids that are NOT stable-correct "
                f"(baseline∧junk). Examples: {bad[:10]}. Refusing — this would make "
                "the ablated null inconclusive."
            )
        excluded_hit = [d for d in ids if d in exclude]
        if excluded_hit:
            raise ValueError(
                f"--doc-ids-file overlaps --exclude-doc-ids-file: {excluded_hit[:10]}"
            )
        if len(ids) != num_samples:
            raise ValueError(
                f"--doc-ids-file has {len(ids)} ids but --num-samples={num_samples}. "
                "Refuse silent n-mismatch (would make rate comparisons inconclusive)."
            )
        return ids

    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if num_samples > len(pool):
        raise ValueError(
            f"Requested num_samples={num_samples} but only {len(pool)} stable-correct "
            f"docs available (pool before exclude={len(stable)}, excluded={len(exclude)})."
        )
    rng = random.Random(seed)
    return rng.sample(pool, num_samples) if num_samples < len(pool) else list(pool)


def _validate_existing_row(row: dict, path: Path) -> None:
    """Resume rows must be unambiguously from this control + generated under ablation."""
    control = row.get("control")
    if control != CONTROL_NAME:
        raise RuntimeError(
            f"{path} row doc_id={row.get('doc_id')} has control={control!r}; "
            f"expected {CONTROL_NAME!r}. Refusing resume of untagged/foreign rows "
            "(would make the ablated null inconclusive)."
        )
    if row.get("selection_rule") != SELECTION_RULE:
        raise RuntimeError(
            f"{path} row doc_id={row.get('doc_id')} has selection_rule={row.get('selection_rule')!r}; "
            f"expected {SELECTION_RULE!r}."
        )
    if row.get("baseline_correct") is not True or row.get("junk_correct") is not True:
        raise RuntimeError(
            f"{path} row doc_id={row.get('doc_id')} missing baseline_correct/junk_correct True flags."
        )
    domains = row.get("junk_domains")
    if not domains:
        raise RuntimeError(
            f"{path} row doc_id={row.get('doc_id')} has empty junk_domains — "
            "looks like a non-ablated generation. Refusing."
        )
    if row.get("ablation_mode") not in (None, "full"):
        # None allowed only for older rows written before ablation_mode was recorded;
        # still require junk_domains above. New writes always set ablation_mode=full.
        raise RuntimeError(
            f"{path} row doc_id={row.get('doc_id')} has ablation_mode={row.get('ablation_mode')!r}; "
            "this control requires full Arditi ablation."
        )


def _load_existing_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        _validate_existing_row(row, path)
        rows[str(row["doc_id"])] = row
    return rows


def run_self_test() -> None:
    import tempfile

    baseline = {"0": False, "1": True, "2": True, "3": False, "4": True}
    junk = {"0": True, "1": True, "2": False, "3": False, "4": True}  # 0 flip, 1+4 stable
    stable, diag = find_stable_correct_doc_ids(baseline, junk)
    assert stable == ["1", "4"], stable
    assert diag["n_flipped_wrong_to_right"] == 1
    assert diag["n_stable_correct"] == 2

    sample = choose_stable_sample_ids(stable, num_samples=1, seed=0)
    assert sample == ["1"] or sample == ["4"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Flip id must be refused.
        bad_ids = tmp_path / "bad.json"
        bad_ids.write_text(json.dumps(["0"]), encoding="utf-8")
        try:
            choose_stable_sample_ids(stable, num_samples=1, seed=0, doc_ids_file=str(bad_ids))
            raise AssertionError("expected ValueError for flip id in doc-ids-file")
        except ValueError:
            pass

        # Silent n-mismatch must be refused.
        ok_ids = tmp_path / "ok.json"
        ok_ids.write_text(json.dumps(["1", "4"]), encoding="utf-8")
        try:
            choose_stable_sample_ids(stable, num_samples=1, seed=0, doc_ids_file=str(ok_ids))
            raise AssertionError("expected ValueError for n-mismatch")
        except ValueError as exc:
            assert "num-samples" in str(exc)

        matched = choose_stable_sample_ids(stable, num_samples=2, seed=0, doc_ids_file=str(ok_ids))
        assert matched == ["1", "4"]

        # Untagged resume rows must be refused.
        dirty = tmp_path / "dirty.jsonl"
        dirty.write_text(
            json.dumps({"doc_id": "1", "generated_completion": "x"}) + "\n",
            encoding="utf-8",
        )
        try:
            _load_existing_rows(dirty)
            raise AssertionError("expected RuntimeError for untagged resume row")
        except RuntimeError as exc:
            assert "control=" in str(exc)

        # Valid resume row accepted.
        clean = tmp_path / "clean.jsonl"
        clean.write_text(
            json.dumps(
                {
                    "doc_id": "1",
                    "control": CONTROL_NAME,
                    "selection_rule": SELECTION_RULE,
                    "baseline_correct": True,
                    "junk_correct": True,
                    "junk_domains": ["forget_bio", "forget_cyber"],
                    "ablation_mode": "full",
                    "generated_completion": "ok",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        rows = _load_existing_rows(clean)
        assert list(rows) == ["1"]

    # Non-bool must hard-fail
    try:
        find_stable_correct_doc_ids({"0": "yes"}, {"0": True})
        raise AssertionError("expected TypeError for non-bool correctness")
    except TypeError:
        pass

    print("self-test OK")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    baseline = load_correctness(args.baseline_correctness)
    junk = load_correctness(args.junk_correctness)
    stable, diag = find_stable_correct_doc_ids(baseline, junk)
    print(
        f"[{CONTROL_NAME}] rule={SELECTION_RULE} "
        f"stable={diag['n_stable_correct']} flipped={diag['n_flipped_wrong_to_right']} "
        f"baseline_correct={diag['n_baseline_correct']} junk_correct={diag['n_junk_correct']}"
    )

    exclude_ids: List[str] = []
    if args.exclude_doc_ids_file:
        exclude_ids = _load_id_list(args.exclude_doc_ids_file)
        print(f"Excluding {len(exclude_ids)} doc_ids from pool")

    sample_ids = choose_stable_sample_ids(
        stable,
        num_samples=args.num_samples,
        seed=args.seed,
        doc_ids_file=args.doc_ids_file,
        exclude_ids=exclude_ids,
    )
    # Final hard guards before any generation.
    stable_set = set(stable)
    flip_set = set(find_flipped_doc_ids(baseline, junk))
    if any(d not in stable_set for d in sample_ids):
        raise RuntimeError("Sample contains non-stable-correct ids after selection.")
    if any(d in flip_set for d in sample_ids):
        raise RuntimeError("Sample contains flipped (wrong→right) ids — control would be inconclusive.")
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Sample contains duplicate doc_ids.")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "control": CONTROL_NAME,
        "selection_rule": SELECTION_RULE,
        "model_label": args.model_label,
        "unlearned_model_id": MODELS[args.model_label],
        "variant": args.variant,
        "forget_domains": args.forget_domain_list,
        "ablation_mode": "full",
        "baseline_correctness_path": str(Path(args.baseline_correctness).resolve()),
        "junk_correctness_path": str(Path(args.junk_correctness).resolve()),
        "directions_root": str(Path(args.directions_root).resolve()),
        "diagnostics": diag,
        "n_sampled": len(sample_ids),
        "sample_ids": sample_ids,
        "excluded_ids": exclude_ids,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt": "format_question from sample_flipped_generations.py (Explain… then letter)",
        "ablation": "all_layer_all_token_ablation mode=full via generate_with_ablation",
        "labeling_scheme": ["genuine", "format-artifact", "contradictory"],
        "interpretation": {
            "high_contradiction_like_flips": "ablation globally decouples scoring from generation",
            "low_contradiction_like_base": "contradiction concentrated on newly flipped docs",
        },
    }
    write_json(out_path.with_suffix(".meta.json"), meta)
    write_json(out_path.with_name(out_path.stem + "_sample_ids.json"), {"sample_ids": sample_ids})
    print(f"Sampled {len(sample_ids)} stable-correct ids; wrote meta + sample_ids sidecar")

    if args.plan_only:
        print("--plan-only: skipping model load / generation")
        return

    require_torch()
    set_seed(args.seed)

    existing = _load_existing_rows(out_path) if args.resume else {}
    # If resuming, ensure previously written rows are still in the planned sample
    # and still stable-correct.
    for doc_id, row in existing.items():
        if doc_id not in stable_set:
            raise RuntimeError(
                f"Resume file has doc_id={doc_id} that is not stable-correct under current "
                "correctness JSONs. Refusing."
            )
        if doc_id not in set(sample_ids):
            raise RuntimeError(
                f"Resume file has doc_id={doc_id} not in current sample plan. "
                "Refusing (would mix plans / inflate n)."
            )

    todo = [d for d in sample_ids if d not in existing]
    have = [d for d in sample_ids if d in existing]
    print(f"Already have={len(have)}; todo={len(todo)}")

    docs = load_task_docs(args.task)
    directions, used_domains = load_junk_directions(
        Path(args.directions_root),
        args.model_label,
        args.variant,
        args.forget_domain_list,
        allow_bio_only=args.allow_bio_only,
    )
    if set(used_domains) != set(args.forget_domain_list) and not args.allow_bio_only:
        raise RuntimeError(
            f"Ablation domains {used_domains} != requested {args.forget_domain_list}. "
            "Pass --allow-bio-only only if you accept that confound."
        )
    print(f"Ablating domains={used_domains} variant={args.variant} (REQUIRED for this control)")

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
            padding_side="left",
        )
        for doc_id in todo:
            idx = int(doc_id) if str(doc_id).isdigit() else None
            if idx is None or idx >= len(docs):
                raise RuntimeError(f"Cannot map doc_id={doc_id} into task docs (len={len(docs)})")
            formatted = format_question(docs[idx])
            if formatted is None:
                raise RuntimeError(f"format_question failed for doc_id={doc_id}")
            # MUST use full ablation — generating without it makes this control meaningless.
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
                "ablation_mode": "full",
                "control": CONTROL_NAME,
                "selection_rule": SELECTION_RULE,
                "baseline_correct": True,
                "junk_correct": True,
                "label": None,
            }
            ordered_rows.append(row)
            existing[doc_id] = row
            print(f"  wrote doc_id={doc_id}")
        unload_model(model)
    else:
        ordered_rows = [existing[d] for d in sample_ids if d in existing]

    if len(ordered_rows) != len(sample_ids):
        raise RuntimeError(
            f"Output row count {len(ordered_rows)} != sample size {len(sample_ids)}"
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            # Re-assert invariants on every written row.
            if row.get("control") != CONTROL_NAME:
                raise RuntimeError(f"Refusing to write row without control={CONTROL_NAME}")
            if str(row["doc_id"]) not in stable_set:
                raise RuntimeError(f"Refusing to write non-stable doc_id={row['doc_id']}")
            if not row.get("junk_domains"):
                raise RuntimeError(f"Refusing to write doc_id={row['doc_id']} without junk_domains")
            if row.get("ablation_mode") != "full":
                raise RuntimeError(
                    f"Refusing to write doc_id={row['doc_id']} with ablation_mode={row.get('ablation_mode')!r}"
                )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta["junk_domains"] = used_domains
    meta["n_written"] = len(ordered_rows)
    write_json(out_path.with_suffix(".meta.json"), meta)
    print(f"Wrote {out_path} ({len(ordered_rows)} rows)")


if __name__ == "__main__":
    main()
