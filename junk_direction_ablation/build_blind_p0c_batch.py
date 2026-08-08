"""Pool junk-flip + base-correct free-text into a blinded P0c labeling batch.

Writes:
  blind_items.jsonl   — identity-stripped items for labeling
  blind_key.json      — SEALED map blind_id -> arm / model / doc_id / prior_label
  blind_label_sheet.jsonl — empty labels
  sample_plan.json    — which doc_ids were selected per arm

Do NOT open blind_key.json until labeling is finished.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED = ("genuine", "format-artifact", "contradictory")
BATCH_SEED = 20260801


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rmu-jsonl", required=True, help="RMU flipped generations (n>=30 target set)")
    p.add_argument("--ilu-jsonl", required=True, help="ILU-RMU flipped generations")
    p.add_argument("--base-jsonl", required=True, help="Base-correct free-text generations")
    p.add_argument("--rmu-prior-labels", default="", help="Optional prior p0c_manual_labels.json for RMU")
    p.add_argument("--ilu-prior-labels", default="", help="Optional prior p0c_manual_labels.json for ILU-RMU")
    p.add_argument("--rmu-include-ids", default="", help="Comma-separated doc_ids that must be in the RMU 30")
    p.add_argument("--ilu-include-ids", default="", help="Comma-separated doc_ids that must be in the ILU 30")
    p.add_argument("--n-per-arm", type=int, default=30)
    p.add_argument("--seed", type=int, default=BATCH_SEED)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_prior_labels(path: str) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for item in obj.get("per_doc", []):
        if item.get("label") in ALLOWED:
            out[str(item["doc_id"])] = item["label"]
    # also accept flipped_generations with labels
    return out


def prior_from_jsonl(rows: List[dict]) -> Dict[str, str]:
    out = {}
    for r in rows:
        lab = r.get("label")
        if lab in ALLOWED:
            out[str(r["doc_id"])] = lab
    return out


def select_rows(
    rows: List[dict],
    n: int,
    must_include: List[str],
    seed: int,
) -> List[dict]:
    by_id = {str(r["doc_id"]): r for r in rows}
    chosen_ids: List[str] = []
    for did in must_include:
        did = str(did)
        if did not in by_id:
            raise KeyError(f"Required doc_id={did} missing from generations JSONL")
        if did not in chosen_ids:
            chosen_ids.append(did)
    remaining = [d for d in by_id if d not in chosen_ids]
    rng = random.Random(seed)
    need = n - len(chosen_ids)
    if need < 0:
        raise ValueError(f"must_include has {len(chosen_ids)} > n={n}")
    if need > len(remaining):
        raise ValueError(f"Need {need} more docs but only {len(remaining)} remain")
    if need:
        chosen_ids.extend(rng.sample(remaining, need))
    return [by_id[d] for d in chosen_ids]


def arm_record(row: dict, arm: str, model_label: str, prior: Optional[str]) -> dict:
    return {
        "arm": arm,
        "model_label": model_label,
        "doc_id": str(row["doc_id"]),
        "question": row["question"],
        "choices": row["choices"],
        "gold_letter": row["gold_letter"],
        "generated_completion": row["generated_completion"],
        "prior_label": prior,
        "variant": row.get("variant"),
        "junk_domains": row.get("junk_domains"),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rmu_rows = load_jsonl(Path(args.rmu_jsonl))
    ilu_rows = load_jsonl(Path(args.ilu_jsonl))
    base_rows = load_jsonl(Path(args.base_jsonl))

    rmu_prior = load_prior_labels(args.rmu_prior_labels)
    rmu_prior.update(prior_from_jsonl(rmu_rows))
    ilu_prior = load_prior_labels(args.ilu_prior_labels)
    ilu_prior.update(prior_from_jsonl(ilu_rows))

    rmu_must = [x.strip() for x in args.rmu_include_ids.split(",") if x.strip()]
    ilu_must = [x.strip() for x in args.ilu_include_ids.split(",") if x.strip()]
    if not rmu_must:
        rmu_must = list(rmu_prior.keys())
    if not ilu_must:
        ilu_must = list(ilu_prior.keys())

    rmu_sel = select_rows(rmu_rows, args.n_per_arm, rmu_must, args.seed)
    ilu_sel = select_rows(ilu_rows, args.n_per_arm, ilu_must, args.seed + 1)
    # base: take all rows if exactly n, else sample with seed (meta should already be n)
    if len(base_rows) < args.n_per_arm:
        raise ValueError(f"base JSONL has {len(base_rows)} < {args.n_per_arm}")
    if len(base_rows) == args.n_per_arm:
        base_sel = base_rows
    else:
        base_sel = select_rows(base_rows, args.n_per_arm, [], args.seed + 2)

    pool: List[dict] = []
    for r in rmu_sel:
        pool.append(arm_record(r, "rmu_junk", "rmu", rmu_prior.get(str(r["doc_id"]))))
    for r in ilu_sel:
        pool.append(arm_record(r, "ilu_rmu_junk", "ilu-rmu", ilu_prior.get(str(r["doc_id"]))))
    for r in base_sel:
        pool.append(arm_record(r, "base_correct", "base", None))

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    key: Dict[str, Any] = {}
    items_path = out_dir / "blind_items.jsonl"
    sheet_path = out_dir / "blind_label_sheet.jsonl"
    with items_path.open("w", encoding="utf-8") as fi, sheet_path.open("w", encoding="utf-8") as fs:
        for i, rec in enumerate(pool):
            blind_id = f"B{i:03d}"
            key[blind_id] = {
                "arm": rec["arm"],
                "model_label": rec["model_label"],
                "doc_id": rec["doc_id"],
                "prior_label": rec["prior_label"],
                "variant": rec["variant"],
                "junk_domains": rec["junk_domains"],
            }
            item = {
                "blind_id": blind_id,
                "question": rec["question"],
                "choices": rec["choices"],
                "gold_letter": rec["gold_letter"],
                "generated_completion": rec["generated_completion"],
                "label": None,
            }
            sheet = {"blind_id": blind_id, "label": None, "rationale": ""}
            fi.write(json.dumps(item, ensure_ascii=False) + "\n")
            fs.write(json.dumps(sheet, ensure_ascii=False) + "\n")

    key_path = out_dir / "blind_key.json"
    key_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_per_arm": args.n_per_arm,
                "n_total": len(pool),
                "protocol": list(ALLOWED),
                "warning": "SEALED until labeling complete. Do not open while labeling.",
                "key": key,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    plan = {
        "seed": args.seed,
        "n_per_arm": args.n_per_arm,
        "rmu_doc_ids": [str(r["doc_id"]) for r in rmu_sel],
        "ilu_doc_ids": [str(r["doc_id"]) for r in ilu_sel],
        "base_doc_ids": [str(r["doc_id"]) for r in base_sel],
        "rmu_prior_in_batch": sum(1 for r in rmu_sel if str(r["doc_id"]) in rmu_prior),
        "ilu_prior_in_batch": sum(1 for r in ilu_sel if str(r["doc_id"]) in ilu_prior),
    }
    (out_dir / "sample_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README_BLINDING.md").write_text(
        "# Blind P0c batch\n\n"
        "1. Label using **only** `blind_items.jsonl` / `blind_label_sheet.jsonl`.\n"
        "2. Allowed labels: `genuine` | `format-artifact` | `contradictory`.\n"
        "3. Do **not** open `blind_key.json` until the sheet is complete.\n"
        "4. Then run `unblind_p0c_batch.py`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(pool)} blind items -> {out_dir}")
    print(f"SEALED key: {key_path}")


if __name__ == "__main__":
    main()
