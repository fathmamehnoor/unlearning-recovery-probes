"""Stage 1 sweep, step 1: base-model pooled activations, ALL 8 grid layers x
both poolings, ONE forward pass per domain corpus. Shared across all four
loss-based models -- run this once before sweep_run_model.py.

Reuses the exact 150-chunks-per-domain / seed-42 probe set already used by
the original last_token-only loss-based null
(local_outputs/junk_direction_loss_based/probe_sets/), per
PREREGISTER_LOSS_BASED_SWEEP.md. That directory must be copied into
--output-root/probe_sets before running (the orchestration script does this;
see run_layer_pooling_sweep.sh).

Usage:

    python sweep_extract_base_activations.py \\
      --output-root ../outputs/junk_direction_loss_based_sweep \\
      --hardware-profile manual --dtype bfloat16 --batch-size 2 --gpu-memory 22GiB
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from ablation_lib import (
    BASE_MODEL,
    apply_hardware_profile,
    load_model_and_tokenizer,
    release_memory,
    require_torch,
    set_seed,
    unload_model,
    write_json,
)
from extract_junk_directions import ensure_probe_sets
from sweep_lib import (
    SWEEP_DOMAINS,
    SWEEP_LAYERS,
    SWEEP_POSITIONS,
    collect_pooled_activations,
    load_pooled_cache,
    save_pooled_cache,
    tokenize_chunks_left,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--domains", default=",".join(SWEEP_DOMAINS))
    parser.add_argument("--n-chunks", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-text-chars", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="manual")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Probe-set seed; must match the existing 150-chunk sets.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-cyber-if-missing", action="store_true")
    args = parser.parse_args()
    apply_hardware_profile(args)
    if args.batch_size is None:
        args.batch_size = {"4090": 2, "a100": 4, "t4x2": 1, "manual": 2}[args.hardware_profile]
    args.domain_labels = [d.strip() for d in args.domains.split(",") if d.strip()]
    return args


def main() -> None:
    args = parse_args()
    require_torch()
    set_seed(args.seed)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "extract_base_run_config.json", vars(args))

    probe_root = output_root / "probe_sets"
    if not probe_root.exists():
        raise FileNotFoundError(
            f"{probe_root} does not exist. Copy the existing 150-chunk/seed-42 probe set from "
            "local_outputs/junk_direction_loss_based/probe_sets/ first (see run_layer_pooling_sweep.sh), "
            "or run extract_junk_directions.py's probe-sampling path directly against this output-root."
        )
    probe_sets = ensure_probe_sets(args, output_root)
    if not probe_sets:
        raise RuntimeError("No probe sets loaded.")
    probe_fingerprints = {
        domain: hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        for domain, texts in probe_sets.items()
    }

    print(f"[{time.strftime('%H:%M:%S')}] loading base model: {BASE_MODEL}")
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

    for domain, texts in probe_sets.items():
        fp = probe_fingerprints[domain]
        cached = load_pooled_cache(output_root, "base", domain, SWEEP_LAYERS, SWEEP_POSITIONS, fp)
        if cached is not None:
            print(f"  reusing cached base/{domain} pooled activations")
            continue
        print(f"[{time.strftime('%H:%M:%S')}] collecting base/{domain}: n={len(texts)} "
              f"layers={SWEEP_LAYERS} positions={SWEEP_POSITIONS}")
        input_ids, attention_mask = tokenize_chunks_left(tokenizer, texts, args.max_length)
        acts = collect_pooled_activations(
            model, input_ids, attention_mask, SWEEP_LAYERS, SWEEP_POSITIONS, args.batch_size
        )
        save_pooled_cache(output_root, "base", domain, SWEEP_LAYERS, SWEEP_POSITIONS, fp, len(texts), acts)
        del input_ids, attention_mask, acts
        release_memory()

    model = unload_model(model)
    release_memory()
    write_json(
        output_root / "base_activations_index.json",
        {
            "domains": list(probe_sets.keys()),
            "layers": list(SWEEP_LAYERS),
            "positions": list(SWEEP_POSITIONS),
            "n_chunks": args.n_chunks,
            "probe_fingerprints": probe_fingerprints,
        },
    )
    print(f"[{time.strftime('%H:%M:%S')}] base pooled activations cached under {output_root / 'pooled_activation_cache'}")


if __name__ == "__main__":
    main()
