"""Confirm the directional-ablation hook in wmdp_bio_lm_eval_ablation.py is
actually active when lm_eval scores WMDP-Bio, rather than a silent no-op
(hook registered on the wrong module, removed before the forward pass runs,
a projection that cancels itself, etc).

Builds the exact (context, continuation) loglikelihood requests lm_eval's
wmdp_bio task produces for its first --num-docs docs -- the same code path
`wmdp_bio_lm_eval_ablation.run_lm_eval`'s `simple_evaluate` call uses
internally -- then scores them on the SAME loaded model and weights twice:
once with no hook, once with `all_layer_all_token_ablation` registered.
That context manager is imported directly from wmdp_bio_lm_eval_ablation.py
(not reimplemented here), so this exercises the real hook code, not a
stand-in that could drift from it. Prints the four raw per-choice (A/B/C/D)
loglikelihoods for both conditions and asserts they differ.

Usage:

    python scripts/hook_activation_check.py \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --direction-path outputs/wmdp_bio_cosmic_eval/wmdp_selected_controller.pt \
      --num-docs 3

If --direction-path is omitted, a random unit direction is used -- fine here,
since this check only needs the hook to visibly perturb loglikelihoods, not
to validate that a specific direction encodes anything meaningful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wmdp_bio_lm_eval_ablation import (  # noqa: E402
    all_layer_all_token_ablation,
    apply_hardware_profile,
    load_direction,
    load_model_and_tokenizer,
    random_unit_direction,
    release_memory,
    require_dependencies,
)

try:
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
except ModuleNotFoundError:
    HFLM = None
    TaskManager = None
    get_task_dict = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--tokenizer", default="", help="Defaults to --model.")
    parser.add_argument("--direction-path", default="", help="Optional .pt artifact with the ablation direction. Falls back to a random unit direction if omitted.")
    parser.add_argument("--direction-key", default="")
    parser.add_argument("--task", default="wmdp_bio")
    parser.add_argument("--num-docs", type=int, default=3)
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="apply_chat_template", action="store_false")
    parser.add_argument("--hardware-profile", choices=["a100", "t4x2", "manual"], default="a100")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-6,
                         help="Minimum per-choice loglikelihood change to count as 'differs' -- guards against "
                              "treating floating-point noise as evidence the hook fired.")
    args, _ = parser.parse_known_args()
    if not args.tokenizer:
        args.tokenizer = args.model
    apply_hardware_profile(args)
    return args


def require_lm_eval_task_api() -> None:
    if HFLM is None or TaskManager is None or get_task_dict is None:
        raise ModuleNotFoundError("lm_eval is required for this check (pip install lm_eval).")


def build_doc_instances(task, lm: "HFLM", args: argparse.Namespace) -> Dict[int, list]:
    """Populates task.instances via lm_eval's own request-building code (the
    same path simple_evaluate uses) and groups the resulting Instance objects
    by doc_id, sorted A/B/C/D by idx."""
    task.build_all_requests(
        limit=args.num_docs,
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=False,
        chat_template=lm.apply_chat_template if args.apply_chat_template else None,
        tokenizer_name=getattr(lm, "tokenizer_name", "") if args.apply_chat_template else "",
    )
    by_doc: Dict[int, list] = {}
    for inst in task.instances:
        by_doc.setdefault(inst.doc_id, []).append(inst)
    for insts in by_doc.values():
        insts.sort(key=lambda i: i.idx)
    return dict(sorted(by_doc.items()))


def score(lm: "HFLM", instances: List) -> List[float]:
    return [float(ll) for ll, _is_greedy in lm.loglikelihood(instances)]


def main() -> None:
    args = parse_args()
    require_dependencies()
    require_lm_eval_task_api()

    print(f"loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args)
    hidden_size = int(model.config.hidden_size)

    if args.direction_path:
        direction, metadata = load_direction(Path(args.direction_path), hidden_size, args.direction_key)
        print(f"direction: {args.direction_path} metadata={metadata}")
    else:
        direction = random_unit_direction(hidden_size, args.seed)
        print("no --direction-path given -- using a random unit direction (only checking that the hook fires).")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    task_manager = TaskManager()
    task_dict = get_task_dict([args.task], task_manager)
    task = task_dict[args.task]

    by_doc = build_doc_instances(task, lm, args)
    if len(by_doc) < args.num_docs:
        raise RuntimeError(f"Only found {len(by_doc)} docs for task={args.task}, wanted {args.num_docs}.")

    print(f"[no hook]  scoring {len(by_doc)} docs")
    with all_layer_all_token_ablation(model, None):
        baseline_scores = {doc_id: score(lm, insts) for doc_id, insts in by_doc.items()}

    print(f"[ablation] scoring {len(by_doc)} docs with the hook registered")
    with all_layer_all_token_ablation(model, direction):
        ablated_scores = {doc_id: score(lm, insts) for doc_id, insts in by_doc.items()}

    letters = "ABCD"
    any_failed = False
    for doc_id in by_doc:
        base = baseline_scores[doc_id]
        abl = ablated_scores[doc_id]
        print(f"\ndoc_id={doc_id}")
        for letter, b, a in zip(letters, base, abl):
            print(f"  {letter}: no_hook={b:.6f}  ablated={a:.6f}  delta={a - b:+.6f}")
        if np.allclose(base, abl, atol=args.atol):
            any_failed = True
            print(f"  FAIL: all four loglikelihoods are within {args.atol} of the no-hook run -- "
                  "the ablation hook does not appear to be affecting this forward pass.")

    del model, tokenizer, lm
    release_memory()

    if any_failed:
        raise AssertionError(
            "Ablation hook produced no detectable change in wmdp_bio loglikelihoods for at least one doc. "
            "The hook is either not registered on the modules lm_eval's forward pass actually calls, is being "
            "removed before scoring runs, or the direction/projection is a no-op for this model."
        )
    print(f"\nOK: ablation hook changed all four per-choice loglikelihoods on every one of {len(by_doc)} docs.")


if __name__ == "__main__":
    main()
