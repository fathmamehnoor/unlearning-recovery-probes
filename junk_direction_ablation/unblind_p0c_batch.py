"""Unblind a completed P0c label sheet and tabulate arms.

Reports Fisher exact (RMU vs ILU) on genuine and contradictory, Wilson CIs,
base contradiction rate, and prior-15 re-label agreement.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ALLOWED = ("genuine", "format-artifact", "contradictory")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blind-dir", required=True)
    p.add_argument(
        "--labels",
        default="",
        help="Completed label sheet JSONL (default: blind_dir/blind_label_sheet.jsonl)",
    )
    p.add_argument("--output-json", default="", help="Default: blind_dir/blind_p0c_results.json")
    p.add_argument("--output-md", default="", help="Default: blind_dir/blind_p0c_results.md")
    return p.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """Return (odds_ratio, two-sided p). Prefer scipy; fall back to manual if missing."""
    try:
        from scipy.stats import fisher_exact

        oddsr, p = fisher_exact([[a, b], [c, d]])
        return float(oddsr), float(p)
    except Exception:
        # Conservative fallback: no p-value
        oddsr = (a * d) / (b * c) if b * c else float("inf")
        return float(oddsr), float("nan")


def cohen_kappa(y1: List[str], y2: List[str]) -> float:
    assert len(y1) == len(y2) and y1
    n = len(y1)
    labels = sorted(set(y1) | set(y2))
    agree = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p_e = 0.0
    for lab in labels:
        p_e += (sum(1 for x in y1 if x == lab) / n) * (sum(1 for x in y2 if x == lab) / n)
    if abs(1 - p_e) < 1e-12:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - p_e) / (1 - p_e)


def rate_block(counts: Counter, n: int) -> dict:
    out = {}
    for lab in ALLOWED:
        k = counts.get(lab, 0)
        lo, hi = wilson_ci(k, n)
        out[lab] = {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": [lo, hi]}
    return out


def main() -> None:
    args = parse_args()
    blind_dir = Path(args.blind_dir)
    labels_path = Path(args.labels) if args.labels else blind_dir / "blind_label_sheet.jsonl"
    key_obj = json.loads((blind_dir / "blind_key.json").read_text(encoding="utf-8"))
    key = key_obj["key"]
    items = {r["blind_id"]: r for r in load_jsonl(blind_dir / "blind_items.jsonl")}
    labels = load_jsonl(labels_path)

    merged = []
    missing = []
    for row in labels:
        bid = row["blind_id"]
        lab = row.get("label")
        if lab not in ALLOWED:
            missing.append(bid)
            continue
        meta = key[bid]
        item = items[bid]
        merged.append(
            {
                "blind_id": bid,
                "label": lab,
                "rationale": row.get("rationale") or "",
                "arm": meta["arm"],
                "model_label": meta["model_label"],
                "doc_id": meta["doc_id"],
                "prior_label": meta.get("prior_label"),
                "gold_letter": item["gold_letter"],
                "question": item["question"],
                "generated_completion": item["generated_completion"],
            }
        )
    if missing:
        raise SystemExit(f"Incomplete/invalid labels for {len(missing)} items (e.g. {missing[:5]})")

    by_arm: Dict[str, List[dict]] = {}
    for r in merged:
        by_arm.setdefault(r["arm"], []).append(r)

    arm_summary = {}
    for arm, rows in sorted(by_arm.items()):
        c = Counter(r["label"] for r in rows)
        arm_summary[arm] = {"n": len(rows), "counts": dict(c), "rates": rate_block(c, len(rows))}

    def count_lab(arm: str, lab: str) -> int:
        return sum(1 for r in by_arm.get(arm, []) if r["label"] == lab)

    n_rmu = len(by_arm.get("rmu_junk", []))
    n_ilu = len(by_arm.get("ilu_rmu_junk", []))
    rmu_g, ilu_g = count_lab("rmu_junk", "genuine"), count_lab("ilu_rmu_junk", "genuine")
    rmu_c, ilu_c = count_lab("rmu_junk", "contradictory"), count_lab("ilu_rmu_junk", "contradictory")

    fisher_genuine = fisher_exact_2x2(rmu_g, n_rmu - rmu_g, ilu_g, n_ilu - ilu_g)
    fisher_contra = fisher_exact_2x2(rmu_c, n_rmu - rmu_c, ilu_c, n_ilu - ilu_c)

    # prior agreement on reused docs
    prior_pairs = [
        (r["prior_label"], r["label"])
        for r in merged
        if r.get("prior_label") in ALLOWED and r["arm"] in {"rmu_junk", "ilu_rmu_junk"}
    ]
    if prior_pairs:
        y1 = [a for a, _ in prior_pairs]
        y2 = [b for _, b in prior_pairs]
        prior_agree = {
            "n": len(prior_pairs),
            "exact_match": sum(1 for a, b in prior_pairs if a == b),
            "exact_match_rate": sum(1 for a, b in prior_pairs if a == b) / len(prior_pairs),
            "cohen_kappa": cohen_kappa(y1, y2),
            "by_arm": {},
        }
        for arm in ("rmu_junk", "ilu_rmu_junk"):
            pairs = [(r["prior_label"], r["label"]) for r in by_arm.get(arm, []) if r.get("prior_label") in ALLOWED]
            if pairs:
                prior_agree["by_arm"][arm] = {
                    "n": len(pairs),
                    "exact_match": sum(1 for a, b in pairs if a == b),
                    "exact_match_rate": sum(1 for a, b in pairs if a == b) / len(pairs),
                    "cohen_kappa": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
                }
    else:
        prior_agree = {"n": 0}

    base_n = len(by_arm.get("base_correct", []))
    base_contra = count_lab("base_correct", "contradictory")
    base_lo, base_hi = wilson_ci(base_contra, base_n)

    # Same-model stable-correct controls (pooled flip+stable blind batches).
    flip_vs_stable = {}
    for model, flip_arm, stable_arm in (
        ("rmu", "rmu_junk", "rmu_stable"),
        ("ilu-rmu", "ilu_rmu_junk", "ilu_rmu_stable"),
    ):
        if flip_arm not in by_arm or stable_arm not in by_arm:
            continue
        n_f, n_s = len(by_arm[flip_arm]), len(by_arm[stable_arm])
        f_c, s_c = count_lab(flip_arm, "contradictory"), count_lab(stable_arm, "contradictory")
        f_g, s_g = count_lab(flip_arm, "genuine"), count_lab(stable_arm, "genuine")
        fisher_c = fisher_exact_2x2(f_c, n_f - f_c, s_c, n_s - s_c)
        fisher_g = fisher_exact_2x2(f_g, n_f - f_g, s_g, n_s - s_g)
        flip_vs_stable[model] = {
            "flip_arm": flip_arm,
            "stable_arm": stable_arm,
            "contradictory": {
                "flip": {"k": f_c, "n": n_f, "rate": f_c / n_f if n_f else None, "wilson95": list(wilson_ci(f_c, n_f))},
                "stable": {"k": s_c, "n": n_s, "rate": s_c / n_s if n_s else None, "wilson95": list(wilson_ci(s_c, n_s))},
                "fisher_odds_ratio": fisher_c[0],
                "fisher_p_value": fisher_c[1],
            },
            "genuine": {
                "flip": {"k": f_g, "n": n_f, "rate": f_g / n_f if n_f else None, "wilson95": list(wilson_ci(f_g, n_f))},
                "stable": {"k": s_g, "n": n_s, "rate": s_g / n_s if n_s else None, "wilson95": list(wilson_ci(s_g, n_s))},
                "fisher_odds_ratio": fisher_g[0],
                "fisher_p_value": fisher_g[1],
            },
        }

    alpha = 0.05
    results = {
        "n_total": len(merged),
        "arms": arm_summary,
        "fisher_rmu_vs_ilu": {
            "genuine": {
                "table": [[rmu_g, n_rmu - rmu_g], [ilu_g, n_ilu - ilu_g]],
                "odds_ratio": fisher_genuine[0],
                "p_value": fisher_genuine[1],
                "significant_at_0.05": (fisher_genuine[1] < alpha) if fisher_genuine[1] == fisher_genuine[1] else None,
            },
            "contradictory": {
                "table": [[rmu_c, n_rmu - rmu_c], [ilu_c, n_ilu - ilu_c]],
                "odds_ratio": fisher_contra[0],
                "p_value": fisher_contra[1],
                "significant_at_0.05": (fisher_contra[1] < alpha) if fisher_contra[1] == fisher_contra[1] else None,
            },
        },
        "base_contradiction": {
            "k": base_contra,
            "n": base_n,
            "rate": (base_contra / base_n) if base_n else None,
            "wilson95": [base_lo, base_hi],
        },
        "flip_vs_stable": flip_vs_stable,
        "prior_label_agreement": prior_agree,
        "per_item": merged,
        "interpretation_rule": (
            "Do not claim RMU vs ILU qualitative regimes unless Fisher p < 0.05 "
            "on the pre-registered contrasts (genuine and/or contradictory). "
            "For flip-vs-stable, both arms must be labeled under the same blinded "
            "manual protocol — do not compare manual flip rates to letter-proxy stable rates."
        ),
    }

    out_json = Path(args.output_json) if args.output_json else blind_dir / "blind_p0c_results.json"
    out_md = Path(args.output_md) if args.output_md else blind_dir / "blind_p0c_results.md"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt_rate(arm: str, lab: str) -> str:
        r = arm_summary[arm]["rates"][lab]
        return f"{r['k']}/{r['n']} ({100 * r['rate']:.0f}%)"

    arm_order = (
        "rmu_junk",
        "ilu_rmu_junk",
        "rmu_stable",
        "ilu_rmu_stable",
        "base_correct",
    )
    title_n = "×".join(str(arm_summary[a]["n"]) for a in arm_order if a in arm_summary)
    lines = [
        f"# Blind P0c results (n={title_n})",
        "",
        "| arm | genuine | format-artifact | contradictory |",
        "|---|---|---|---|",
    ]
    for arm in arm_order:
        if arm not in arm_summary:
            continue
        lines.append(
            f"| {arm} | {fmt_rate(arm, 'genuine')} | {fmt_rate(arm, 'format-artifact')} | {fmt_rate(arm, 'contradictory')} |"
        )
    lines += [
        "",
        f"Fisher exact RMU vs ILU junk **genuine**: p={fisher_genuine[1]:.4f} (OR={fisher_genuine[0]:.3f})",
        f"Fisher exact RMU vs ILU junk **contradictory**: p={fisher_contra[1]:.4f} (OR={fisher_contra[0]:.3f})",
    ]
    if base_n:
        lines.append(
            f"Base contradiction rate: {base_contra}/{base_n} "
            f"({100 * base_contra / base_n:.1f}%, Wilson95 [{100 * base_lo:.1f}%, {100 * base_hi:.1f}%])"
        )
    if flip_vs_stable:
        lines.append("")
        lines.append("## Flip vs same-model stable-correct (same blinded protocol)")
        for model, block in flip_vs_stable.items():
            c = block["contradictory"]
            g = block["genuine"]
            lines.append(
                f"- **{model}** contradictory: flip {c['flip']['k']}/{c['flip']['n']} "
                f"({100 * c['flip']['rate']:.0f}%) vs stable {c['stable']['k']}/{c['stable']['n']} "
                f"({100 * c['stable']['rate']:.0f}%); Fisher p={c['fisher_p_value']:.4f}"
            )
            lines.append(
                f"  genuine: flip {g['flip']['k']}/{g['flip']['n']} "
                f"({100 * g['flip']['rate']:.0f}%) vs stable {g['stable']['k']}/{g['stable']['n']} "
                f"({100 * g['stable']['rate']:.0f}%); Fisher p={g['fisher_p_value']:.4f}"
            )
    lines.append("")
    if prior_agree.get("n"):
        lines.append(
            f"Prior-label agreement (reused docs): {prior_agree['exact_match']}/{prior_agree['n']} "
            f"exact ({100 * prior_agree['exact_match_rate']:.0f}%), κ={prior_agree['cohen_kappa']:.3f}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md.read_text())
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
