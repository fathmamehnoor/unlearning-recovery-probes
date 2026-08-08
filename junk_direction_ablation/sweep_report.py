"""Deliverables 1-3 for the layer x pooling sweep (PREREGISTER_LOSS_BASED_SWEEP.md):

  sweep_results.json  -- one record per (model, layer, pooling, condition) run
  sweep_results.md    -- junk - matched per cell, 4 models x 8 layers x 2 poolings,
                          promoted cells marked
  FINDINGS.md          -- per model: max junk - matched over the whole grid, and
                          whether anything promoted

Promotion is mechanical: junk_acc - matched_acc >= PROMOTION_DELTA (0.05) on the
n=200 screen. Nothing else promotes -- not a trend, not a near miss, not a
significant p-value at a smaller delta. Every cell is reported, including
nulls and degenerate-model cells (flagged, not interpreted).

Run only after sweep_sanity_check.py has passed for every model with a
sanity cell.

Usage:

    python sweep_report.py --sweep-root ../outputs/junk_direction_loss_based_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from ablation_lib import write_json
from sweep_lib import (
    CHANCE_ACC,
    CHANCE_BAND,
    DEGENERATE_MODELS,
    PROMOTION_DELTA,
    SWEEP_LAYERS,
    SWEEP_MODELS,
    SWEEP_POSITIONS,
    variant_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--models", default=",".join(SWEEP_MODELS))
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Only for debugging the report format itself. The pre-registration requires the sanity "
             "check to pass before results are interpreted.",
    )
    return parser.parse_args()


def require_sanity_passed(sweep_root: Path) -> None:
    path = sweep_root / "sanity_check_results.json"
    if not path.exists():
        raise RuntimeError(f"{path} not found. Run sweep_sanity_check.py before sweep_report.py.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    failed = [row for row in payload.get("cells", []) if row.get("status") not in ("OK",)]
    if failed:
        raise RuntimeError(
            f"sanity_check_results.json has {len(failed)} non-OK cell(s). "
            "Fix the discrepancy before trusting sweep results (see PREREGISTER_LOSS_BASED_SWEEP.md)."
        )


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not args.skip_sanity_check:
        require_sanity_passed(sweep_root)

    doc_ids_path = sweep_root / "screen_doc_ids.json"
    doc_ids_payload = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    doc_ids_hash = doc_ids_payload["doc_ids_hash"]
    n_screen = doc_ids_payload["n"]

    all_rows: List[dict] = []
    per_model_rows: Dict[str, List[dict]] = {}
    for model in models:
        summary_path = sweep_root / "screen" / model / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"{summary_path} missing -- run sweep_run_model.py --model-label {model} first.")
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        per_model_rows[model] = rows
        for r in rows:
            all_rows.append({
                "model": model,
                "layer": r["layer"],
                "pooling": r["position"],
                "variant": r["variant"],
                "condition": r["condition"],
                "accuracy": r["accuracy"],
                "n": r["n"],
                "doc_ids_hash": doc_ids_hash,
                "per_doc_correctness_path": r["per_doc_correctness_path"],
                "degenerate": r.get("degenerate", model in DEGENERATE_MODELS),
            })

    write_json(sweep_root / "sweep_results.json", all_rows)

    # --- Cell table: junk - matched at every (model, layer, pooling) ---
    cells: List[dict] = []
    for model in models:
        rows = per_model_rows[model]
        baseline_row = next((r for r in rows if r["condition"] == "baseline"), None)
        baseline_acc = baseline_row["accuracy"] if baseline_row else None
        at_or_below_chance = baseline_acc is not None and baseline_acc <= CHANCE_ACC + CHANCE_BAND
        by_variant: Dict[str, Dict[str, float]] = {}
        for r in rows:
            if r["condition"] == "baseline":
                continue
            by_variant.setdefault(r["variant"], {})[r["condition"]] = r["accuracy"]
        for layer in SWEEP_LAYERS:
            for pos in SWEEP_POSITIONS:
                variant = variant_name(layer, pos)
                found = by_variant.get(variant, {})
                junk = found.get("junk_direction_ablation")
                matched = found.get("matched_control")
                delta: Optional[float] = (junk - matched) if (junk is not None and matched is not None) else None
                promoted = delta is not None and delta >= PROMOTION_DELTA
                cells.append({
                    "model": model,
                    "layer": layer,
                    "pooling": pos,
                    "variant": variant,
                    "baseline_acc": baseline_acc,
                    "junk_acc": junk,
                    "matched_acc": matched,
                    "delta_junk_minus_matched": delta,
                    "promoted": promoted,
                    "degenerate": model in DEGENERATE_MODELS,
                    "at_or_below_chance": at_or_below_chance,
                })
    write_json(sweep_root / "sweep_cells.json", {"cells": cells, "n_screen": n_screen, "doc_ids_hash": doc_ids_hash})

    # --- sweep_results.md ---
    lines = [
        "# Layer x pooling sweep -- junk-direction null on loss-based methods",
        "",
        f"Screen: n={n_screen} fixed wmdp_bio doc subset (doc_ids_hash={doc_ids_hash[:12]}...), "
        f"`apply_chat_template=false`, rank-2 junk (bio+cyber) vs rank-1 matched (retain).",
        "",
        f"Promotion rule: junk - matched >= {PROMOTION_DELTA:.0%}. Nothing else promotes.",
        "",
        "| model | layer | pooling | baseline | junk | matched | junk-matched | promoted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_promoted = False
    for cell in cells:
        def fmt(x: Optional[float]) -> str:
            return f"{x:.1%}" if x is not None else "n/a"

        flag = "**PROMOTED**" if cell["promoted"] else ""
        if cell["promoted"]:
            any_promoted = True
        deg = " \\*" if cell["degenerate"] else ""
        lines.append(
            f"| {cell['model']}{deg} | {cell['layer']} | {cell['pooling']} | {fmt(cell['baseline_acc'])} | "
            f"{fmt(cell['junk_acc'])} | {fmt(cell['matched_acc'])} | "
            f"{fmt(cell['delta_junk_minus_matched'])} | {flag} |"
        )
    lines += [
        "",
        "\\*Degenerate model (npo, npo-ilu) -- include, report, do not interpret.",
        "",
        "**Any cell promoted:** " + ("YES -- see FINDINGS.md and Stage 2 outputs." if any_promoted else "No."),
    ]
    (sweep_root / "sweep_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- FINDINGS.md ---
    findings_lines = [
        "# FINDINGS -- layer x pooling sweep, loss-based junk-direction null",
        "",
        f"n={n_screen} screen, {len(models)} models x {len(SWEEP_LAYERS)} layers x {len(SWEEP_POSITIONS)} "
        f"poolings = {len(models) * len(SWEEP_LAYERS) * len(SWEEP_POSITIONS)} cells.",
        "",
    ]
    n_promoted_total = 0
    for model in models:
        model_cells = [c for c in cells if c["model"] == model and c["delta_junk_minus_matched"] is not None]
        if not model_cells:
            findings_lines.append(f"## {model}\n\nNo cells with both junk and matched accuracy -- see sweep_results.md.\n")
            continue
        best = max(model_cells, key=lambda c: c["delta_junk_minus_matched"])
        promoted_cells = [c for c in model_cells if c["promoted"]]
        n_promoted_total += len(promoted_cells)
        deg_note = " **(degenerate -- do not interpret)**" if model in DEGENERATE_MODELS else ""
        chance_note = " **(baseline at/below chance)**" if model_cells[0]["at_or_below_chance"] else ""
        findings_lines.append(f"## {model}{deg_note}{chance_note}")
        findings_lines.append("")
        findings_lines.append(
            f"- Max junk - matched over the grid: **{best['delta_junk_minus_matched']:+.1%}** "
            f"at layer {best['layer']} / {best['pooling']} "
            f"(junk={best['junk_acc']:.1%}, matched={best['matched_acc']:.1%}, baseline={best['baseline_acc']:.1%})."
        )
        findings_lines.append(
            f"- Promoted cells: {len(promoted_cells)}"
            + (
                " (" + "; ".join(f"L{c['layer']}/{c['pooling']} ({c['delta_junk_minus_matched']:+.1%})" for c in promoted_cells) + ")"
                if promoted_cells else " -- none."
            )
        )
        findings_lines.append("")

    findings_lines.append("## Overall")
    findings_lines.append("")
    findings_lines.append(
        f"**{n_promoted_total} cell(s) promoted out of {len(cells)}.** "
        + ("See Stage 2 outputs for confirmation." if n_promoted_total else "The pre-registered null holds across the full grid.")
    )
    (sweep_root / "FINDINGS.md").write_text("\n".join(findings_lines) + "\n", encoding="utf-8")

    promoted = [c for c in cells if c["promoted"]]
    write_json(sweep_root / "promoted_cells.json", promoted)
    print(f"wrote sweep_results.json, sweep_results.md, sweep_cells.json, FINDINGS.md, promoted_cells.json under {sweep_root}")
    print(f"promoted cells: {n_promoted_total} / {len(cells)}")
    if n_promoted_total:
        print("Promoted cells written to promoted_cells.json -- run sweep_stage2_confirm.py next.")


if __name__ == "__main__":
    main()
