"""Extract junk directions û = normalize(mean(a_unl) - mean(a_base)).

For each selected model and each domain corpus {forget_bio, forget_cyber,
retain}, extracts residual-stream directions at chosen layers × position
poolings. Default for RMU/ILU-RMU is layers 6/7/8 × last/mean (six variants).
Loss-based null runs use --use-diagnostic-layers (divergence + norm-spike
nominees per model) and typically --positions last_token only.

Corpora:
  forget_bio   -- cais/wmdp-corpora bio-forget-corpus (or local jsonl)
  forget_cyber -- cais/wmdp-corpora cyber-forget-corpus
  retain       -- wikitext-2-raw-v1 (matched-control text distribution)

No chat template: ScaleAI RMU ships a broken chat template; raw text chunks
are the correct RMU training distribution anyway.

Usage:

    python extract_junk_directions.py \\
      --output-root outputs/junk_direction \\
      --models rmu,ilu-rmu \\
      --domains forget_bio,forget_cyber,retain \\
      --n-chunks 150 \\
      --max-length 512 \\
      --hardware-profile 4090
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None

from ablation_lib import (
    BASE_MODEL,
    DEFAULT_LAYERS,
    DIAGNOSTIC_LAYERS,
    MODELS,
    POSITION_CONVENTIONS,
    apply_hardware_profile,
    load_model_and_tokenizer,
    release_memory,
    require_torch,
    set_seed,
    unload_model,
    unit_vector,
    variant_name,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DOMAIN_SPECS = {
    "forget_bio": {
        "kind": "hf",
        # bio-forget is NOT inside cais/wmdp-corpora — it is a separate gated dataset.
        "candidates": [
            ("cais/wmdp-bio-forget-corpus", None),
        ],
        "split": "train",
        "text_column": "text",
        "local_jsonl": REPO_ROOT / "data" / "wmdp" / "wmdp-corpora" / "bio-forget-corpus.jsonl",
        "access_hint": (
            "WMDP bio-forget is gated. Request access at cais/wmdp-bio-forget-corpus, "
            "or place bio-forget-corpus.jsonl under data/wmdp/wmdp-corpora/ "
            "(from the wmdp-corpora.zip release)."
        ),
    },
    "forget_cyber": {
        "kind": "hf",
        "candidates": [
            ("cais/wmdp-corpora", "cyber-forget-corpus"),
            ("cais/wmdp-cyber-forget-corpus", None),
        ],
        "split": "train",
        "text_column": "text",
        "local_jsonl": REPO_ROOT / "data" / "wmdp" / "wmdp-corpora" / "cyber-forget-corpus.jsonl",
        "access_hint": "Install/download cyber-forget-corpus from cais/wmdp-corpora or the wmdp-corpora.zip.",
    },
    "retain": {
        "kind": "hf",
        "candidates": [
            # Newer huggingface_hub requires namespace/name; bare 'wikitext' fails.
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("wikitext", "wikitext-2-raw-v1"),
        ],
        "split": "test",
        "text_column": "text",
        "local_jsonl": None,
        "access_hint": "Use Salesforce/wikitext (wikitext-2-raw-v1) or a local jsonl.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--models",
        default="rmu,ilu-rmu",
        help=f"Comma-separated subset of {','.join(MODELS)}.",
    )
    parser.add_argument(
        "--domains",
        default="forget_bio,forget_cyber,retain",
        help="Comma-separated subset of forget_bio,forget_cyber,retain.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layers (shared by all models). Default: 6,7,8. "
             "Ignored when --use-diagnostic-layers is set.",
    )
    parser.add_argument(
        "--use-diagnostic-layers",
        action="store_true",
        help="Extract at each model's DIAGNOSTIC_LAYERS (divergence + norm-spike nominees). "
             "Base activations use the union across selected models.",
    )
    parser.add_argument(
        "--positions",
        default=",".join(POSITION_CONVENTIONS),
        help="Comma-separated pooling conventions (default: last_token,mean_over_positions).",
    )
    parser.add_argument("--n-chunks", type=int, default=150, help="Chunks per domain (100-200).")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-text-chars", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=None, help="Activation collection batch size (int).")
    parser.add_argument("--hardware-profile", choices=["4090", "a100", "t4x2", "manual"], default="4090")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--skip-cyber-if-missing",
        action="store_true",
        help="If forget_cyber corpus cannot be loaded, skip it instead of failing.",
    )
    args = parser.parse_args()
    apply_hardware_profile(args)
    if args.batch_size is None:
        # Activation collection wants an int; map hardware profiles to safe defaults.
        args.batch_size = {"4090": 2, "a100": 4, "t4x2": 1, "manual": 2}[args.hardware_profile]
    args.model_labels = [m.strip() for m in args.models.split(",") if m.strip()]
    args.domain_labels = [d.strip() for d in args.domains.split(",") if d.strip()]
    args.position_list = [p.strip() for p in args.positions.split(",") if p.strip()]
    unknown_pos = [p for p in args.position_list if p not in POSITION_CONVENTIONS]
    if unknown_pos:
        raise ValueError(f"Unknown positions {unknown_pos}; choose from {list(POSITION_CONVENTIONS)}.")
    unknown_models = [m for m in args.model_labels if m not in MODELS]
    if unknown_models:
        raise ValueError(f"Unknown models {unknown_models}; choose from {list(MODELS)}.")
    unknown_domains = [d for d in args.domain_labels if d not in DOMAIN_SPECS]
    if unknown_domains:
        raise ValueError(f"Unknown domains {unknown_domains}; choose from {list(DOMAIN_SPECS)}.")
    if args.use_diagnostic_layers:
        missing_diag = [m for m in args.model_labels if m not in DIAGNOSTIC_LAYERS]
        if missing_diag:
            raise ValueError(f"No DIAGNOSTIC_LAYERS entry for {missing_diag}.")
        args.layers_by_model = {m: list(DIAGNOSTIC_LAYERS[m]) for m in args.model_labels}
        # Union for base activation collection.
        args.layer_list = sorted({int(x) for layers in args.layers_by_model.values() for x in layers})
    else:
        layer_src = args.layers if args.layers is not None else ",".join(str(x) for x in DEFAULT_LAYERS)
        args.layer_list = [int(x) for x in layer_src.split(",") if x.strip()]
        args.layers_by_model = {m: list(args.layer_list) for m in args.model_labels}
    # last_token-only runs store (n, hidden) instead of (n, seq, hidden). A full-seq
    # cache for layers {2,6,15,30,31} × 3 domains is ~19GB and OOMs mid-extract,
    # which is worse than a failed run: it leaves partial directions.
    args.activation_storage = "last_token" if args.position_list == ["last_token"] else "full_seq"
    if not (100 <= args.n_chunks <= 200):
        print(f"WARNING: --n-chunks={args.n_chunks} is outside the spec range 100-200.")
    return args


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def load_texts_from_jsonl(path: Path, text_column: str, min_chars: int) -> List[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = []
    for row in rows:
        text = normalize_text(row.get(text_column) or row.get("content") or row.get("prompt"))
        if len(text) >= min_chars:
            texts.append(text)
    return texts


def load_texts_from_hf(candidates, split: str, text_column: str, min_chars: int) -> List[str]:
    if load_dataset is None:
        raise ModuleNotFoundError("datasets is required to download corpora.")
    last_error = None
    for repo, config in candidates:
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
            texts = []
            for row in ds:
                text = normalize_text(row.get(text_column))
                if len(text) >= min_chars:
                    texts.append(text)
            if texts:
                print(f"  loaded {len(texts)} texts from {repo}" + (f"/{config}" if config else ""))
                return texts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  failed {repo}" + (f"/{config}" if config else "") + f": {exc}")
    raise RuntimeError(f"Could not load corpus from candidates {candidates}: {last_error}")


def load_domain_texts(domain: str, min_chars: int, skip_cyber_if_missing: bool) -> Optional[List[str]]:
    spec = DOMAIN_SPECS[domain]
    local = spec.get("local_jsonl")
    if local is not None and Path(local).exists():
        texts = load_texts_from_jsonl(Path(local), spec["text_column"], min_chars)
        print(f"  loaded {len(texts)} texts from local {local}")
        return texts
    try:
        return load_texts_from_hf(spec["candidates"], spec["split"], spec["text_column"], min_chars)
    except Exception as exc:
        hint = spec.get("access_hint", "")
        if domain == "forget_cyber" and skip_cyber_if_missing:
            print(f"WARNING: skipping forget_cyber ({exc}). {hint}")
            print(
                "WARNING: ablating only û_bio on a two-domain WMDP checkpoint under-recovers "
                "and makes a weak result inconclusive (method vs missing direction)."
            )
            return None
        raise RuntimeError(f"Failed to load domain={domain}: {exc}. {hint}") from exc


def sample_chunks(texts: Sequence[str], n: int, seed: int, domain: str, output_root: Path) -> List[str]:
    if len(texts) < n:
        raise RuntimeError(f"Domain {domain}: only {len(texts)} usable texts, wanted {n}.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=n, replace=False)
    sampled = [texts[int(i)] for i in indices]
    path = output_root / "probe_sets" / f"{domain}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(sampled):
            f.write(json.dumps({"id": f"{domain}:{i}", "domain": domain, "text": text}, ensure_ascii=False) + "\n")
    fingerprint = hashlib.sha256("\n".join(sampled).encode("utf-8")).hexdigest()
    write_json(
        output_root / "probe_sets" / f"{domain}_meta.json",
        {"domain": domain, "n": len(sampled), "seed": seed, "fingerprint": fingerprint},
    )
    return sampled


def tokenize_chunks(tokenizer, texts: Sequence[str], max_length: int):
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def layers_cache_tag(layers: Sequence[int]) -> str:
    return "L" + "-".join(str(int(x)) for x in layers)


def activation_cache_path(output_root: Path, label: str, domain: str, layers: Sequence[int], storage: str) -> Path:
    # Include layers + storage in the filename so a prior 6/7/8 full-seq cache
    # cannot be silently reused for a diagnostic last_token run (or vice versa).
    return (
        output_root
        / "activation_cache"
        / f"{label}_{domain}_{layers_cache_tag(layers)}_{storage}.pt"
    )


@torch.inference_mode()
def collect_layer_activations(
    model,
    input_ids,
    attention_mask,
    layers: Sequence[int],
    batch_size: int,
    storage: str = "full_seq",
) -> Dict[int, "torch.Tensor"]:
    """Collect residual stream AFTER decoder block `layer` (same index as Step 2).

    storage='full_seq': {layer: (n, seq, hidden)} — required for mean_over_positions.
    storage='last_token': {layer: (n, hidden)} — much smaller; only for last_token pooling.
    """
    require_torch()
    if storage not in {"full_seq", "last_token"}:
        raise ValueError(f"Unknown activation storage: {storage}")
    n_layers_model = int(model.config.num_hidden_layers)
    for layer in layers:
        if layer < 0 or layer >= n_layers_model:
            raise ValueError(f"layer={layer} out of range for model with {n_layers_model} layers.")

    input_device = model.get_input_embeddings().weight.device
    n = input_ids.shape[0]
    buckets: Dict[int, List["torch.Tensor"]] = {layer: [] for layer in layers}

    for start in range(0, n, batch_size):
        batch_ids = input_ids[start:start + batch_size].to(input_device)
        batch_mask = attention_mask[start:start + batch_size].to(input_device)
        outputs = model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        for layer in layers:
            h = hidden_states[layer + 1].detach().float().cpu()
            if storage == "last_token":
                # Left-padding invariant: real last token is at index -1.
                h = h[:, -1, :]
            buckets[layer].append(h)
        del outputs, hidden_states, batch_ids, batch_mask
        release_memory()

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in buckets.items()}


def pool_activations(
    acts: "torch.Tensor",
    attention_mask: "torch.Tensor",
    position: str,
) -> "torch.Tensor":
    """Pool activations to (n, hidden).

    Accepts either full-seq caches (n, seq, hidden) or last_token caches (n, hidden).
    """
    if position == "last_token":
        if acts.ndim == 2:
            return acts
        if not bool((attention_mask[:, -1] == 1).all()):
            raise RuntimeError("last_token pooling requires left padding with real tokens at index -1.")
        return acts[:, -1, :]
    if position == "mean_over_positions":
        if acts.ndim == 2:
            raise RuntimeError(
                "Activation cache is last_token-only; re-run extraction with "
                "--positions including mean_over_positions (full_seq storage)."
            )
        mask = attention_mask.unsqueeze(-1).float()
        summed = (acts * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom
    raise ValueError(f"Unknown position convention: {position}")


def mean_activation(pooled: "torch.Tensor") -> "torch.Tensor":
    return pooled.mean(dim=0)


def extract_direction(a_unl_mean: "torch.Tensor", a_base_mean: "torch.Tensor") -> "torch.Tensor":
    return unit_vector(a_unl_mean - a_base_mean)


def load_mask(output_root: Path, label: str, domain: str, n: int, seq: int) -> "torch.Tensor":
    path = output_root / "activation_cache" / f"{label}_{domain}_mask.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    payload = torch.load(output_root / "activation_cache" / f"{label}_{domain}.pt", map_location="cpu")
    if "attention_mask" in payload:
        return payload["attention_mask"]
    raise FileNotFoundError(f"Missing attention mask for {label}/{domain}")


def ensure_probe_sets(args: argparse.Namespace, output_root: Path) -> Dict[str, List[str]]:
    probe_sets: Dict[str, List[str]] = {}
    for domain in args.domain_labels:
        cached = output_root / "probe_sets" / f"{domain}.jsonl"
        meta_path = output_root / "probe_sets" / f"{domain}_meta.json"
        if cached.exists():
            texts = [json.loads(line)["text"] for line in cached.read_text(encoding="utf-8").splitlines() if line.strip()]
            fingerprint = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
            if len(texts) != args.n_chunks:
                raise ValueError(
                    f"Cached probe set {cached} has {len(texts)} chunks but --n-chunks={args.n_chunks}. "
                    "Delete probe_sets/ (and activation_cache/) or match --n-chunks/--seed."
                )
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if int(meta.get("seed", -1)) != args.seed:
                    raise ValueError(
                        f"Cached probe set {domain} seed={meta.get('seed')} != --seed={args.seed}. "
                        "Delete probe_sets/ and activation_cache/ to resample."
                    )
                if meta.get("fingerprint") not in (None, fingerprint):
                    raise ValueError(
                        f"Cached probe set {domain} fingerprint mismatch vs jsonl contents. "
                        "Delete probe_sets/ and activation_cache/."
                    )
            else:
                write_json(
                    meta_path,
                    {"domain": domain, "n": len(texts), "seed": args.seed, "fingerprint": fingerprint,
                     "note": "meta backfilled from existing jsonl"},
                )
            print(f"Reusing cached probe set {domain}: {len(texts)} chunks")
            probe_sets[domain] = texts
            continue
        print(f"Loading corpus for domain={domain}")
        texts = load_domain_texts(domain, args.min_text_chars, args.skip_cyber_if_missing)
        if texts is None:
            continue
        probe_sets[domain] = sample_chunks(texts, args.n_chunks, args.seed, domain, output_root)
        print(f"  sampled {len(probe_sets[domain])} chunks -> {cached}")
    if "forget_bio" in args.domain_labels and "forget_bio" not in probe_sets:
        raise RuntimeError("forget_bio is required and failed to load — cannot run Step 3 without it.")
    if "retain" in args.domain_labels and "retain" not in probe_sets:
        raise RuntimeError("retain is required for the matched control and failed to load.")
    return probe_sets


def collect_for_model(
    model_id: str,
    label: str,
    probe_sets: Dict[str, List[str]],
    args: argparse.Namespace,
    output_root: Path,
    tokenizer,
    probe_fingerprints: Dict[str, str],
    layers: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[int, "torch.Tensor"]]:
    """Returns nested dict domain -> layer -> pooled-ready full acts (n, seq, hidden)."""
    layer_list = list(layers) if layers is not None else list(args.layer_list)
    storage = str(args.activation_storage)
    print(
        f"[{time.strftime('%H:%M:%S')}] loading {label}: {model_id} "
        f"(layers={layer_list}, storage={storage})"
    )
    model, _ = load_model_and_tokenizer(
        model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        tokenizer_id=BASE_MODEL if label != "base" else model_id,
        padding_side="left",  # extraction only; lm_eval uses right padding
    )
    domain_acts: Dict[str, Dict[int, "torch.Tensor"]] = {}
    for domain, texts in probe_sets.items():
        cache_path = activation_cache_path(output_root, label, domain, layer_list, storage)
        if cache_path.exists():
            print(f"  reusing cached activations {cache_path}")
            payload = torch.load(cache_path, map_location="cpu")
            expected_fp = probe_fingerprints[domain]
            if payload.get("probe_fingerprint") not in (None, expected_fp):
                raise ValueError(
                    f"Activation cache {cache_path} was built on a different probe set. "
                    "Delete activation_cache/ and re-run extraction."
                )
            if payload.get("storage") not in (None, storage):
                raise ValueError(
                    f"Activation cache {cache_path} storage={payload.get('storage')} != {storage}."
                )
            cached_layers = [int(x) for x in payload.get("layers", [])]
            if cached_layers != layer_list:
                if not set(layer_list).issubset(set(cached_layers)):
                    raise ValueError(
                        f"Activation cache {cache_path} layers={cached_layers} does not cover "
                        f"requested {layer_list}. Delete activation_cache/ or match --layers."
                    )
            if int(payload.get("n_chunks", -1)) not in (-1, len(texts)) and payload.get("n_chunks") is not None:
                if int(payload["n_chunks"]) != len(texts):
                    raise ValueError(
                        f"Activation cache {cache_path} n_chunks={payload.get('n_chunks')} != {len(texts)}."
                    )
            acts_map = {int(k): v for k, v in payload["acts"].items()}
            any_act = next(iter(acts_map.values()))
            if any_act.shape[0] != len(texts):
                raise ValueError(
                    f"Activation cache {cache_path} has n={any_act.shape[0]} but probe set has {len(texts)}."
                )
            domain_acts[domain] = {layer: acts_map[layer] for layer in layer_list}
            continue
        input_ids, attention_mask = tokenize_chunks(tokenizer, texts, args.max_length)
        if not bool((attention_mask[:, -1] == 1).all()):
            raise RuntimeError("Left-padding invariant broken during extraction tokenization.")
        print(
            f"  collecting {domain}: n={input_ids.shape[0]} seq={input_ids.shape[1]} "
            f"layers={layer_list} storage={storage}"
        )
        acts = collect_layer_activations(
            model, input_ids, attention_mask, layer_list, args.batch_size, storage=storage
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "label": label,
                "domain": domain,
                "layers": list(layer_list),
                "storage": storage,
                "n_chunks": len(texts),
                "max_length": args.max_length,
                "probe_fingerprint": probe_fingerprints[domain],
                "dtype": str(args.dtype),
                "acts": {str(k): v for k, v in acts.items()},
                "attention_mask": attention_mask.cpu(),
            },
            cache_path,
        )
        torch.save(attention_mask.cpu(), output_root / "activation_cache" / f"{label}_{domain}_mask.pt")
        domain_acts[domain] = acts
        del input_ids, attention_mask, acts
        release_memory()
    unload_model(model)
    model = None
    release_memory()
    return domain_acts


def main() -> None:
    args = parse_args()
    require_torch()
    set_seed(args.seed)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "extract_run_config.json", vars(args))

    probe_sets = ensure_probe_sets(args, output_root)
    if not probe_sets:
        raise RuntimeError("No probe sets loaded.")
    probe_fingerprints = {
        domain: hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        for domain, texts in probe_sets.items()
    }
    if "forget_cyber" not in probe_sets and "forget_cyber" in args.domain_labels:
        print(
            "WARNING: running without forget_cyber. Full eval will only ablate û_bio unless cyber "
            "directions are added later."
        )

    print(f"[{time.strftime('%H:%M:%S')}] loading base tokenizer: {BASE_MODEL}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[{time.strftime('%H:%M:%S')}] collecting BASE activations (union layers={args.layer_list})")
    base_acts = collect_for_model(
        BASE_MODEL,
        "base",
        probe_sets,
        args,
        output_root,
        tokenizer,
        probe_fingerprints,
        layers=args.layer_list,
    )

    for model_label in args.model_labels:
        model_id = MODELS[model_label]
        model_layers = list(args.layers_by_model[model_label])
        print(
            f"\n[{time.strftime('%H:%M:%S')}] === extracting directions for {model_label} "
            f"(layers={model_layers}, positions={args.position_list}) ==="
        )
        unl_acts = collect_for_model(
            model_id,
            model_label,
            probe_sets,
            args,
            output_root,
            tokenizer,
            probe_fingerprints,
            layers=model_layers,
        )

        for domain, texts in probe_sets.items():
            mask = load_mask(output_root, model_label, domain, len(texts), args.max_length)
            base_mask = load_mask(output_root, "base", domain, len(texts), args.max_length)
            if not torch.equal(mask, base_mask):
                raise RuntimeError(f"Attention mask mismatch base vs {model_label} on domain={domain}.")

            for layer in model_layers:
                for position in args.position_list:
                    a_base = pool_activations(base_acts[domain][layer], mask, position)
                    a_unl = pool_activations(unl_acts[domain][layer], mask, position)
                    direction = extract_direction(mean_activation(a_unl), mean_activation(a_base))
                    raw_norm = float((mean_activation(a_unl) - mean_activation(a_base)).norm())
                    if raw_norm < 1e-4:
                        print(
                            f"WARNING: near-zero raw direction for {model_label}/{domain}/"
                            f"{variant_name(layer, position)} (norm={raw_norm:.2e}) — "
                            "direction may be noise."
                        )
                    variant = variant_name(layer, position)
                    out_path = output_root / "directions" / model_label / domain / f"{variant}.pt"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "direction": direction,
                        "u_raw_norm": raw_norm,
                        "model_label": model_label,
                        "model_id": model_id,
                        "base_model": BASE_MODEL,
                        "domain": domain,
                        "layer": layer,
                        "position": position,
                        "variant": variant,
                        "n_chunks": len(texts),
                        "max_length": args.max_length,
                        "seed": args.seed,
                        "probe_fingerprint": probe_fingerprints[domain],
                        "formula": "u = normalize(mean(a_unl) - mean(a_base))",
                        "use_diagnostic_layers": bool(args.use_diagnostic_layers),
                    }
                    torch.save(payload, out_path)
                    write_json(
                        out_path.with_suffix(".json"),
                        {k: (float(v) if isinstance(v, float) else v) for k, v in payload.items() if k != "direction"},
                    )
                    print(
                        f"  saved {model_label}/{domain}/{variant} "
                        f"raw_norm={raw_norm:.4f} -> {out_path}"
                    )

        del unl_acts
        release_memory()

    cyber_present = "forget_cyber" in probe_sets
    if "forget_cyber" in args.domain_labels and not cyber_present:
        msg = (
            "forget_cyber was requested but not loaded. Bio-only junk ablation under-recovers "
            "on multi-domain WMDP checkpoints and makes a null inconclusive."
        )
        if args.skip_cyber_if_missing:
            print("WARNING: " + msg + " Continuing because --skip-cyber-if-missing was set.")
        else:
            raise RuntimeError(msg + " Fix corpus access or pass --skip-cyber-if-missing deliberately.")
    write_json(
        output_root / "directions_index.json",
        {
            "models": args.model_labels,
            "domains_present": sorted(probe_sets.keys()),
            "layers_union": args.layer_list,
            "layers_by_model": args.layers_by_model,
            "positions": list(args.position_list),
            "activation_storage": args.activation_storage,
            "use_diagnostic_layers": bool(args.use_diagnostic_layers),
            "n_chunks": args.n_chunks,
            "cyber_present": cyber_present,
        },
    )
    print(f"[{time.strftime('%H:%M:%S')}] wrote directions under {output_root / 'directions'}")


if __name__ == "__main__":
    main()
