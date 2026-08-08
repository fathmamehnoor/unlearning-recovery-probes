"""Shared helpers for Step 3 junk-direction extraction and ablation.

Matches the Arditi-style residual-stream ablation used in
`unlearning_direction_evaluation/scripts/wmdp_bio_lm_eval_ablation.py`
(all-layer / all-token pre-hooks on block input + attn output + MLP output),
extended to ablate one or more unit directions (û_bio ± û_cyber).
"""

from __future__ import annotations

import gc
import json
import random
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
except Exception:
    simple_evaluate = None
    HFLM = None


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODELS = {
    "rmu": "ScaleAI/mhj-llama3-8b-rmu",
    "ilu-rmu": "OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct",
    "graddiff": "OPTML-Group/GradDiff-WMDP-llama3-8b-instruct",
    "npo": "OPTML-Group/NPO-WMDP-llama3-8b-instruct",
    "npo-ilu": "OPTML-Group/NPO-ILU-WMDP-llama3-8b-instruct",
    "idk-ap": "OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct",
}

# Optional local-path overrides for Vast curl prefetch (avoids hub-cache OOM).
# JUNK_BASE_MODEL_PATH=/path/to/base
# JUNK_MODEL_PATHS_JSON='{"graddiff":"/path/to/graddiff", ...}'
import os as _os

if _os.environ.get("JUNK_BASE_MODEL_PATH"):
    BASE_MODEL = _os.environ["JUNK_BASE_MODEL_PATH"]
_paths_json = _os.environ.get("JUNK_MODEL_PATHS_JSON")
if _paths_json:
    import json as _json

    _overrides = _json.loads(_paths_json)
    if not isinstance(_overrides, dict):
        raise ValueError("JUNK_MODEL_PATHS_JSON must be a JSON object")
    MODELS = {**MODELS, **{str(k): str(v) for k, v in _overrides.items()}}

# Layer indexing matches llama3_wmdp_layer_divergence.py:
# hidden_states[0] = embeddings; hidden_states[layer+1] = residual after decoder block `layer`.
# Step-2 bump for RMU / ILU-RMU landed at layer 7.
DEFAULT_LAYERS = (6, 7, 8)
POSITION_CONVENTIONS = ("last_token", "mean_over_positions")

# From black_box/layer_divergence_probe: for RMU/ILU-RMU, divergence and norm-spike
# coincide at 7. For loss-based methods they diverge — extract at both nominees.
DIAGNOSTIC_LAYERS = {
    "rmu": (7,),
    "ilu-rmu": (7,),
    "graddiff": (6, 30),   # divergence, norm_spike
    "npo": (31, 15),
    "npo-ilu": (2, 30),
    "idk-ap": (2, 30),
}

# Degenerate / uninterpretable on every prior arm — include, flag, do not interpret.
DEGENERATE_MODELS = frozenset({"npo", "npo-ilu"})

LOSS_BASED_MODELS = ("graddiff", "npo", "npo-ilu", "idk-ap")

HARDWARE_PROFILES = {
    "4090": {"batch_size": "auto:2", "dtype": "bfloat16", "gpu_memory": "22GiB", "cpu_memory": "64GiB"},
    "a100": {"batch_size": "auto", "dtype": "bfloat16", "gpu_memory": "72GiB", "cpu_memory": "64GiB"},
    "t4x2": {"batch_size": "auto:1", "dtype": "float16", "gpu_memory": "14GiB", "cpu_memory": "30GiB"},
    "manual": {"batch_size": "auto:2", "dtype": "bfloat16", "gpu_memory": "22GiB", "cpu_memory": "64GiB"},
}

ACC_KEY = "acc,none"


def require_torch() -> None:
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ModuleNotFoundError("torch and transformers are required.")


def require_lm_eval() -> None:
    if simple_evaluate is None or HFLM is None:
        raise ModuleNotFoundError("lm_eval is required (pip install lm_eval).")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    require_torch()
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def release_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "model"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def max_memory(gpu_memory: Optional[str], cpu_memory: Optional[str]) -> Optional[Dict[object, str]]:
    require_torch()
    if not torch.cuda.is_available() or not gpu_memory:
        return None
    memory = {idx: gpu_memory for idx in range(torch.cuda.device_count())}
    if cpu_memory:
        memory["cpu"] = cpu_memory
    return memory


def load_model_and_tokenizer(
    model_id: str,
    dtype: str,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    gpu_memory: Optional[str] = None,
    cpu_memory: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    padding_side: str = "right",
):
    """Load a causal LM.

    padding_side defaults to 'right' — required for lm_eval loglikelihood scoring.
    Use padding_side='left' only for activation extraction that pools the final
    non-pad token. Mixing these up silently corrupts either extraction or eval.
    """
    require_torch()
    if padding_side not in {"left", "right"}:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id or model_id, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    kwargs = {
        "torch_dtype": dtype_from_name(dtype),
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    memory = max_memory(gpu_memory, cpu_memory)
    if memory:
        kwargs["max_memory"] = memory
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def unload_model(model) -> None:
    """Best-effort GPU free. Callers MUST drop their own reference:

        model = unload_model(model)   # or: unload_model(model); model = None
    """
    try:
        model.to("cpu")
    except Exception:  # noqa: BLE001
        pass
    del model
    release_memory()
    return None


def random_orthogonal_basis(hidden_size: int, rank: int, seed: int) -> List["torch.Tensor"]:
    """Rank-matched random control: `rank` orthonormal random directions."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mat = torch.randn(hidden_size, rank, generator=generator)
    # QR gives orthonormal columns
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return [unit_vector(q[:, i]) for i in range(rank)]


def get_decoder_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    raise AttributeError("Could not locate decoder layers on this causal LM.")


def attention_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def mlp_module(layer):
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def unit_vector(vector: "torch.Tensor") -> "torch.Tensor":
    vector = vector.detach().float().cpu().flatten()
    norm = vector.norm()
    if float(norm) <= 1e-8:
        raise ValueError("Direction has near-zero norm.")
    return (vector / norm).contiguous()


def orthogonalize_directions(directions: Sequence["torch.Tensor"]) -> List["torch.Tensor"]:
    """Gram-Schmidt orthonormalize a list of directions (CPU float)."""
    basis: List["torch.Tensor"] = []
    for raw in directions:
        v = unit_vector(raw).clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        if float(v.norm()) <= 1e-6:
            continue
        basis.append(unit_vector(v))
    if not basis:
        raise ValueError("No linearly independent directions after orthogonalization.")
    return basis


def random_unit_direction(hidden_size: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return unit_vector(torch.randn(hidden_size, generator=generator))


@contextmanager
def all_layer_all_token_ablation(
    model,
    directions: Optional[Sequence["torch.Tensor"]],
    mode: str = "full",
):
    """Project one or more unit directions out of residual-stream writes.

    mode='full' (default): block pre-hook + attention output + MLP output
    (Arditi Eq. 4 / Appendix E; matches wmdp_bio_lm_eval_ablation.py).

    mode='resid_pre_only': block pre-hook only. Use if full ablation tanks
    MMLU (Step 3 decision rule: damage vs junk-clearing).
    """
    if not directions:
        yield
        return
    if mode not in {"full", "resid_pre_only"}:
        raise ValueError(f"Unknown ablation mode: {mode}")
    basis = orthogonalize_directions(list(directions))
    handles = []

    def project_away(hidden: "torch.Tensor") -> "torch.Tensor":
        out = hidden
        for direction in basis:
            local = direction.to(device=out.device, dtype=out.dtype)
            projection = out @ local
            out = out - projection.unsqueeze(-1) * local
        return out

    def pre_hook(_module, inputs):
        return (project_away(inputs[0]),) + tuple(inputs[1:])

    def output_hook(_module, _inputs, output):
        if torch.is_tensor(output):
            return project_away(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (project_away(output[0]),) + tuple(output[1:])
        return output

    try:
        for layer_idx, layer in enumerate(get_decoder_layers(model)):
            handles.append(layer.register_forward_pre_hook(pre_hook))
            if mode == "full":
                attn = attention_module(layer)
                if attn is None:
                    raise AttributeError(f"Layer {layer_idx} has no attention module.")
                handles.append(attn.register_forward_hook(output_hook))
                mlp = mlp_module(layer)
                if mlp is None:
                    raise AttributeError(f"Layer {layer_idx} has no MLP module.")
                handles.append(mlp.register_forward_hook(output_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def run_lm_eval(
    model,
    tokenizer,
    tasks: Sequence[str],
    batch_size,
    apply_chat_template: bool,
    seed: int,
    limit: Optional[int] = None,
) -> Tuple[dict, dict]:
    require_lm_eval()
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        apply_chat_template=apply_chat_template,
        limit=limit,
        log_samples=True,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    return results["results"], results.get("samples", {})


def gold_index(sample: dict) -> Optional[int]:
    target = sample.get("target")
    if isinstance(target, bool):
        target = int(target)
    if isinstance(target, int):
        return target
    if isinstance(target, str):
        stripped = target.strip()
        if stripped.isdigit():
            return int(stripped)
        if len(stripped) == 1 and stripped.upper() in "ABCDEFGH":
            return ord(stripped.upper()) - ord("A")
    doc = sample.get("doc") or {}
    for key in ("answer", "label", "gold"):
        val = doc.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and len(val) == 1 and val.upper() in "ABCDEFGH":
            return ord(val.upper()) - ord("A")
    return None


def choice_loglikelihoods(sample: dict) -> Optional[List[float]]:
    resps = sample.get("filtered_resps")
    if resps is None:
        resps = sample.get("resps")
    if not isinstance(resps, list) or not resps:
        return None
    lls: List[float] = []
    for entry in resps:
        cur = entry
        while isinstance(cur, (list, tuple)) and cur and isinstance(cur[0], (list, tuple)):
            cur = cur[0]
        if isinstance(cur, (list, tuple)):
            cur = cur[0]
        try:
            lls.append(float(cur))
        except (TypeError, ValueError):
            return None
    return lls


def per_doc_correctness(samples: List[dict]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for i, sample in enumerate(samples):
        doc_id = sample.get("doc_id")
        key = str(doc_id) if doc_id is not None else str(i)
        if doc_id is None:
            print(
                f"WARNING: sample {i} has no doc_id; pairing by position only works if "
                "every condition iterates docs in the same order."
            )
        lls = choice_loglikelihoods(sample)
        gold = gold_index(sample)
        if lls is None or gold is None or gold >= len(lls):
            continue
        out[key] = max(range(len(lls)), key=lambda idx: lls[idx]) == gold
    return out


def apply_hardware_profile(args) -> None:
    profile = HARDWARE_PROFILES[args.hardware_profile]
    for key, value in profile.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def variant_name(layer: int, position: str) -> str:
    return f"layer{layer}_{position}"


def parse_variant_name(name: str) -> Tuple[int, str]:
    match = re.fullmatch(r"layer(\d+)_(last_token|mean_over_positions)", name)
    if not match:
        raise ValueError(f"Unrecognized variant name: {name}")
    return int(match.group(1)), match.group(2)
