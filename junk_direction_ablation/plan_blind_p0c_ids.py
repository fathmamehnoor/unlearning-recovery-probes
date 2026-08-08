#!/usr/bin/env python3
"""Plan n=30 doc_ids per arm (reuse prior 15) and seed blind_p0c_n30 dirs."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/junk_direction/blind_p0c_n30"
SEED = 20260801


def load_flips(eval_dir: Path):
    base = json.loads((eval_dir / "per_doc_correctness/wmdp_bio/baseline.json").read_text())
    junk = json.loads((eval_dir / "per_doc_correctness/wmdp_bio/junk_direction_ablation.json").read_text())
    return sorted(k for k in base if (not bool(base[k])) and bool(junk.get(k)))


def prior_ids(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [str(r["doc_id"]) for r in rows]


def choose(flipped, must, n, seed):
    chosen = list(must)
    rem = [d for d in flipped if d not in set(chosen)]
    need = n - len(chosen)
    chosen.extend(random.Random(seed).sample(rem, need))
    return chosen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rmu_eval = ROOT / "outputs/junk_direction/eval/rmu_layer7_last_token"
    ilu_eval = ROOT / "outputs/junk_direction/eval/ilu-rmu_layer7_last_token"
    rmu_flips = load_flips(rmu_eval)
    ilu_flips = load_flips(ilu_eval)
    rmu_must = prior_ids(rmu_eval / "flipped_generations.jsonl")
    ilu_must = prior_ids(ilu_eval / "flipped_generations.jsonl")

    rmu30 = choose(rmu_flips, rmu_must, 30, SEED)
    ilu30 = choose(ilu_flips, ilu_must, 30, SEED + 1)

    base_corr = json.loads((rmu_eval / "per_doc_correctness/wmdp_bio/base_model.json").read_text())
    base_ids = sorted(k for k, v in base_corr.items() if bool(v))
    base30 = random.Random(SEED).sample(base_ids, 30)

    plan = {
        "seed": SEED,
        "rmu_sample_ids": rmu30,
        "ilu_sample_ids": ilu30,
        "base_sample_ids": base30,
        "rmu_prior_15": rmu_must,
        "ilu_prior_15": ilu_must,
        "rmu_new_15": [d for d in rmu30 if d not in set(rmu_must)],
        "ilu_new_15": [d for d in ilu30 if d not in set(ilu_must)],
    }
    (OUT / "doc_id_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (OUT / "rmu_sample_ids.json").write_text(json.dumps({"sample_ids": rmu30}, indent=2) + "\n")
    (OUT / "ilu_sample_ids.json").write_text(json.dumps({"sample_ids": ilu30}, indent=2) + "\n")
    (OUT / "base_sample_ids.json").write_text(json.dumps({"sample_ids": base30}, indent=2) + "\n")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in plan.items()}, indent=2))
    print("new rmu", plan["rmu_new_15"])
    print("new ilu", plan["ilu_new_15"])


if __name__ == "__main__":
    main()
