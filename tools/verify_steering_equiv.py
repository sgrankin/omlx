# SPDX-License-Identifier: Apache-2.0
"""Verify the in-tree _steer (mx.compile-fused) matches a legacy direct loop
in actual decode output. Loads a model, applies steering, decodes the same
prompt with the new path and with a monkey-patched legacy _steer, and
diffs the token streams.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import omlx.patches.steering as steering_mod
from omlx.cli import _load_steering_model
from omlx.patches.steering import (
    apply_steering_patch,
    model_hidden_size,
    remove_steering_patch,
)
from omlx.steering import SteeringSpec, SteeringVector


def _legacy_steer(layer: Any, h: mx.array) -> mx.array:
    if layer._steer_add is not None:
        add = layer._steer_add
        if add.dtype != h.dtype:
            add = add.astype(h.dtype)
        h = h + add
    for unit, strength in layer._steer_proj:
        u = unit if unit.dtype == h.dtype else unit.astype(h.dtype)
        coeff = (h * u).sum(axis=-1, keepdims=True)
        h = h - strength * coeff * u
    return h


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", None) or model


def _build_vector(n_layers: int, n_embd: int, seed: int) -> SteeringVector:
    rng = mx.random.key(seed)
    directions: dict[int, mx.array] = {}
    for il in range(n_layers):
        rng, sk = mx.random.split(rng)
        v = mx.random.normal((n_embd,), key=sk, dtype=mx.float32)
        v = v / mx.linalg.norm(v)
        directions[il] = v
    return SteeringVector(
        directions=directions, n_embd=n_embd, method="manual", scaling="unit",
        model="synthetic",
    )


def _decode(model: Any, tokenizer: Any, prompt_ids: list[int], n: int) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    tm = _text_model(model)
    cache = make_prompt_cache(tm)

    def logits(out: Any) -> mx.array:
        return out.logits if hasattr(out, "logits") else out

    out = logits(tm(mx.array(prompt_ids)[None], cache=cache))[:, -1, :]
    tokens: list[int] = []
    for _ in range(n):
        nxt = int(mx.argmax(out[0]).item())
        tokens.append(nxt)
        out = logits(tm(mx.array([[nxt]]), cache=cache))[:, -1, :]
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--prompt", default="Explain kernel fusion in two sentences.")
    args = parser.parse_args()

    path = (
        args.model if Path(args.model).is_absolute()
        else str(Path.home() / ".omlx" / "models" / args.model)
    )
    print(f"Loading {path}")
    model, tokenizer = _load_steering_model(path)
    n_layers = len(_text_model(model).model.layers)
    n_embd = model_hidden_size(model)
    print(f"  n_layers={n_layers}  n_embd={n_embd}")

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = list(tokenizer.encode(templated))
        except Exception:
            prompt_ids = list(tokenizer.encode(args.prompt))
    else:
        prompt_ids = list(tokenizer.encode(args.prompt))

    v1 = _build_vector(n_layers, n_embd, seed=2)
    v2 = _build_vector(n_layers, n_embd, seed=3)

    configs = [
        ("add P=0", [SteeringSpec(vector=v1, strength=args.strength, mode="add")]),
        ("proj P=1", [SteeringSpec(vector=v1, strength=args.strength, mode="project")]),
        ("proj P=2", [
            SteeringSpec(vector=v1, strength=args.strength, mode="project"),
            SteeringSpec(vector=v2, strength=args.strength, mode="project"),
        ]),
    ]
    # ``_steered_call`` resolves ``_steer`` from module globals at call time,
    # so rebinding the module attribute (not a class attribute — there is no
    # wrapper class anymore) redirects every steered layer.
    saved_steer = steering_mod._steer
    all_ok = True
    for label, specs in configs:
        apply_steering_patch(model, specs)

        # In-tree (compile-fused) path.
        steering_mod._steer = saved_steer
        in_tree = _decode(model, tokenizer, prompt_ids, args.tokens)

        # Legacy direct loop.
        steering_mod._steer = _legacy_steer
        legacy = _decode(model, tokenizer, prompt_ids, args.tokens)

        matches = in_tree == legacy
        diff_pos = next(
            (i for i in range(min(len(in_tree), len(legacy))) if in_tree[i] != legacy[i]),
            None,
        )
        print(f"\n{label}: {'MATCH' if matches else f'DIVERGES at token {diff_pos}'}")
        print(f"  in_tree: {tokenizer.decode(in_tree)!r}")
        print(f"  legacy:  {tokenizer.decode(legacy)!r}")
        if not matches:
            all_ok = False

    remove_steering_patch(model)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
