"""Sanity gate for the layer x pooling sweep (PREREGISTER_LOSS_BASED_SWEEP.md).

Before trusting any sweep cell, the sweep pipeline must reproduce the
existing full-run (n=1273) numbers at the already-tested cells -- GradDiff
L6/L30, NPO-ILU L2/L30, IDK-AP L2/L30, all last_token -- restricted to the
same fixed n=200 doc subset, to within subsample noise:

  * per-document agreement >= 0.97 on the 200 shared docs
  * |screen accuracy - existing accuracy restricted to the same 200 docs| <= 0.02

for each of baseline / junk_direction_ablation / matched_control(_ablation).

If either bound fails anywhere, this exits non-zero and prints the
discrepancy. Do not proceed to sweep_report.py or Stage 2 until this passes.

Usage:

    python sweep_sanity_check.py \\
      --sweep-root ../outputs/junk_direction_loss_based_sweep \\
      --existing-root ../local_outputs/junk_direction_loss_based
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from ablation_lib import write_json
from sweep_lib import (
    SANITY_ACC_TOL,
    SANITY_AGREEMENT_MIN,
    SANITY_CELLS,
    variant_name,
)

# Screen condition name -> existing full-run condition name (naming diverged
# slightly: the existing arm calls matched control "matched_control_ablation",
# the sweep calls it "matched_control").
CONDITION_MAP = {
    "baseline": "baseline",
    "junk_direction_ablation": "junk_direction_ablation",
    "matched_control": "matched_control_ablation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--task", default="wmdp_bio")
    return parser.parse_args()


def load_correctness(path: Path) -> Dict[str, bool]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(k): bool(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def compare_cell(
    sweep_root: Path, existing_root: Path, task: str, model: str, layer: int, doc_ids: list
) -> list:
    variant = variant_name(layer, "last_token")
    rows = []
    for screen_cond, existing_cond in CONDITION_MAP.items():
        screen_path = sweep_root / "screen" / model / "per_doc_correctness" / variant / f"{screen_cond}.json"
        existing_path = existing_root / "eval" / model / variant / "per_doc_correctness" / task / f"{existing_cond}.json"
        if not screen_path.exists():
            rows.append({"model": model, "variant": variant, "condition": screen_cond, "status": "MISSING_SCREEN", "path": str(screen_path)})
            continue
        if not existing_path.exists():
            rows.append({"model": model, "variant": variant, "condition": screen_cond, "status": "MISSING_EXISTING", "path": str(existing_path)})
            continue
        screen = load_correctness(screen_path)
        existing = load_correctness(existing_path)
        doc_id_strs = [str(d) for d in doc_ids]
        missing_in_screen = [d for d in doc_id_strs if d not in screen]
        missing_in_existing = [d for d in doc_id_strs if d not in existing]
        if missing_in_screen or missing_in_existing:
            rows.append({
                "model": model, "variant": variant, "condition": screen_cond, "status": "MISSING_DOCS",
                "missing_in_screen": missing_in_screen[:5], "missing_in_existing": missing_in_existing[:5],
            })
            continue
        agree = sum(1 for d in doc_id_strs if screen[d] == existing[d]) / len(doc_id_strs)
        acc_screen = sum(screen[d] for d in doc_id_strs) / len(doc_id_strs)
        acc_existing = sum(existing[d] for d in doc_id_strs) / len(doc_id_strs)
        acc_diff = abs(acc_screen - acc_existing)
        ok = agree >= SANITY_AGREEMENT_MIN and acc_diff <= SANITY_ACC_TOL
        rows.append({
            "model": model, "variant": variant, "condition": screen_cond,
            "n_docs": len(doc_id_strs), "agreement": agree, "acc_screen": acc_screen,
            "acc_existing_restricted": acc_existing, "acc_diff": acc_diff,
            "agreement_min": SANITY_AGREEMENT_MIN, "acc_tol": SANITY_ACC_TOL,
            "status": "OK" if ok else "FAIL",
        })
    return rows


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    existing_root = Path(args.existing_root).expanduser().resolve()

    doc_ids_path = sweep_root / "screen_doc_ids.json"
    if not doc_ids_path.exists():
        print(f"ERROR: {doc_ids_path} not found -- run sweep_run_model.py for at least one model first.")
        sys.exit(2)
    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))["doc_ids"]

    all_rows = []
    for model, layers in SANITY_CELLS.items():
        for layer in layers:
            all_rows.extend(compare_cell(sweep_root, existing_root, args.task, model, layer, doc_ids))

    write_json(sweep_root / "sanity_check_results.json", {"cells": all_rows, "n_screen_docs": len(doc_ids)})

    n_fail = 0
    for row in all_rows:
        status = row["status"]
        if status == "OK":
            print(
                f"OK    {row['model']:<10} {row['variant']:<24} {row['condition']:<24} "
                f"agree={row['agreement']:.3f} acc_diff={row['acc_diff']:.4f}"
            )
        else:
            n_fail += 1
            print(f"{status:<18} {row['model']:<10} {row.get('variant','?'):<24} {row.get('condition','?'):<24} {row}")

    if n_fail:
        print(f"\nSANITY CHECK FAILED: {n_fail} cell(s) did not reproduce the existing full-run numbers.")
        print("Stop. Report the discrepancy. Do not run sweep_report.py or Stage 2 until this is fixed.")
        sys.exit(1)

    print(f"\nSANITY CHECK PASSED: all {len(all_rows)} reproduction cells within tolerance.")


if __name__ == "__main__":
    main()
