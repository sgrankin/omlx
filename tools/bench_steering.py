# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: steering forward-path overhead, legacy loop vs pre-stacked.

For each (model, P) where P is the number of stacked projections per layer,
we time a short decode loop with steering applied. The model is loaded once
and steering is reapplied with synthetic random unit-direction vectors. We
compare the new in-tree `_steer` (pre-stacked, indexed) against a monkey-
patched legacy `_steer` that iterates the Python tuple list (the pre-fusion
implementation). Same patch, same weights, same prompt — only the inner
forward-path code differs.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omlx.cli import _load_steering_model
from omlx.patches.steering import (
    apply_steering_patch,
    model_hidden_size,
    remove_steering_patch,
)
from omlx.steering import SteeringSpec, SteeringVector


def _legacy_steer(layer: Any, h: mx.array) -> mx.array:
    """Pre-fusion implementation: iterate _steer_proj list of (unit, strength)."""
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


def _new_steer(layer: Any, h: mx.array) -> mx.array:
    """Current in-tree implementation, pinned here for explicit A/B.

    Known pre-existing staleness: ``_steer_units`` / ``_steer_strengths`` no
    longer exist (that stacked-tensor implementation was replaced by the
    compiled body) — leaving as-is, not fixed by this change.
    """
    if layer._steer_add is not None:
        add = layer._steer_add
        if add.dtype != h.dtype:
            add = add.astype(h.dtype)
        h = h + add
    units = layer._steer_units
    if units is not None:
        strengths = layer._steer_strengths
        if units.dtype != h.dtype:
            units = units.astype(h.dtype)
            strengths = strengths.astype(h.dtype)
        for i in range(units.shape[0]):
            u = units[i]
            coeff = (h * u).sum(axis=-1, keepdims=True)
            h = h - strengths[i] * coeff * u
    return h


# Per-instance compiled steer functions, keyed by id(layer). Compiled once on
# first call against a given P and dtype; mx.compile internally caches by
# input shapes/dtypes too.
_COMPILED_CACHE: dict[int, Any] = {}


def _compiled_steer(layer: Any, h: mx.array) -> mx.array:
    """mx.compile-fused variant: the Python loop over projections traces
    once into a single graph, eliminating per-op dispatch overhead."""
    key = id(layer)
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        # Capture Python ints/floats and array references at trace time.
        proj = [(u, float(s)) for u, s in layer._steer_proj]
        add = layer._steer_add

        if add is not None and proj:
            def body(h):
                h = h + add
                for unit, strength in proj:
                    u = unit if unit.dtype == h.dtype else unit.astype(h.dtype)
                    coeff = (h * u).sum(axis=-1, keepdims=True)
                    h = h - strength * coeff * u
                return h
        elif add is not None:
            def body(h):
                return h + add
        elif proj:
            def body(h):
                for unit, strength in proj:
                    u = unit if unit.dtype == h.dtype else unit.astype(h.dtype)
                    coeff = (h * u).sum(axis=-1, keepdims=True)
                    h = h - strength * coeff * u
                return h
        else:
            def body(h):
                return h

        fn = mx.compile(body)
        _COMPILED_CACHE[key] = fn
    return fn(h)


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", None) or model


def _build_synthetic_vector(n_layers: int, n_embd: int, seed: int) -> SteeringVector:
    """Random unit directions on every layer (idx 0..n_layers-1)."""
    rng = mx.random.key(seed)
    directions: dict[int, mx.array] = {}
    for il in range(n_layers):
        rng, sk = mx.random.split(rng)
        v = mx.random.normal((n_embd,), key=sk, dtype=mx.float32)
        v = v / mx.linalg.norm(v)
        directions[il] = v
    return SteeringVector(
        directions=directions,
        n_embd=n_embd,
        method="manual",
        scaling="unit",
        model="synthetic-bench",
    )


def _make_specs(vectors: list[SteeringVector], mode: str, strength: float) -> list[SteeringSpec]:
    return [SteeringSpec(vector=v, mode=mode, strength=strength) for v in vectors]


def _count_layers(model: Any) -> int:
    tm = _text_model(model)
    root = getattr(tm, "model", tm)
    return len(root.layers)


def _decode_steps(model: Any, tokenizer: Any, prompt_ids: list[int], n_steps: int) -> float:
    """Time a single-token decode loop of n_steps; return wall seconds."""
    from mlx_lm.models.cache import make_prompt_cache

    tm = _text_model(model)
    cache = make_prompt_cache(tm)

    def _logits(out: Any) -> mx.array:
        return out.logits if hasattr(out, "logits") else out

    # Prefill — not timed.
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    logits = _logits(out)[:, -1, :]
    nxt = int(mx.argmax(logits[0]).item())  # forces eval
    mx.eval(logits)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())
    mx.eval(logits)
    return time.perf_counter() - t0


def _run_one(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    n_warmup: int,
    n_steps: int,
    label: str,
    steer_fn: Callable | None,
) -> dict:
    """Run a single configuration: warmup + measurement; return stats dict."""
    if steer_fn is not None:
        # ``_steered_call`` resolves ``_steer`` from module globals at call
        # time, so rebinding the module attribute (not a class attribute —
        # there is no wrapper class anymore) redirects every steered layer.
        import omlx.patches.steering as steering_mod

        steering_mod._steer = steer_fn

    _decode_steps(model, tokenizer, prompt_ids, n_warmup)
    s1 = _decode_steps(model, tokenizer, prompt_ids, n_steps)
    s2 = _decode_steps(model, tokenizer, prompt_ids, n_steps)
    s3 = _decode_steps(model, tokenizer, prompt_ids, n_steps)
    best = min(s1, s2, s3)
    tps = n_steps / best
    print(
        f"  {label:>22s}  best={best:.3f}s  ({s1:.3f}/{s2:.3f}/{s3:.3f})  "
        f"tok/s={tps:.2f}"
    )
    return {"label": label, "best_s": best, "tps": tps, "runs": [s1, s2, s3]}


def bench_model(model_path: str, n_steps: int, n_warmup: int) -> dict:
    print(f"\n=== {model_path} ===")
    t_load = time.perf_counter()
    model, tokenizer = _load_steering_model(model_path)
    t_load = time.perf_counter() - t_load
    print(f"  loaded in {t_load:.1f}s")

    n_layers = _count_layers(model)
    n_embd = model_hidden_size(model)
    print(f"  n_layers={n_layers}  n_embd={n_embd}")

    prompt = "Write one sentence about kernel fusion on GPUs."
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = list(tokenizer.encode(templated))
        except Exception:
            prompt_ids = list(tokenizer.encode(prompt))
    else:
        prompt_ids = list(tokenizer.encode(prompt))

    results: dict[str, Any] = {"model": model_path, "n_layers": n_layers, "configs": []}

    # No-steering baseline (set once; doesn't depend on _steer impl).
    remove_steering_patch(model)
    print("\n  -- no steering --")
    r = _run_one(model, tokenizer, prompt_ids, n_warmup, n_steps, "baseline", None)
    results["configs"].append({**r, "P": 0, "mode": None})

    def _bench_variants(label_prefix: str, specs: list[SteeringSpec], P: int, mode: str):
        apply_steering_patch(model, specs)
        # Compiled variant must reset its cache between patches (per-instance keys).
        _COMPILED_CACHE.clear()
        for variant, fn in (
            ("legacy", _legacy_steer),
            ("new", _new_steer),
            ("compiled", _compiled_steer),
        ):
            r = _run_one(model, tokenizer, prompt_ids, n_warmup, n_steps, variant, fn)
            results["configs"].append({**r, "P": P, "mode": mode, "variant": variant})

    print("\n  -- P=1, mode=add --")
    v = _build_synthetic_vector(n_layers, n_embd, seed=1)
    _bench_variants("add", _make_specs([v], mode="add", strength=0.5), 0, "add")

    print("\n  -- P=1, mode=project --")
    v1 = _build_synthetic_vector(n_layers, n_embd, seed=2)
    _bench_variants("p1", _make_specs([v1], mode="project", strength=0.5), 1, "project")

    print("\n  -- P=2, mode=project --")
    v2 = _build_synthetic_vector(n_layers, n_embd, seed=3)
    _bench_variants("p2", _make_specs([v1, v2], mode="project", strength=0.5), 2, "project")

    print("\n  -- P=4, mode=project --")
    v3 = _build_synthetic_vector(n_layers, n_embd, seed=4)
    v4 = _build_synthetic_vector(n_layers, n_embd, seed=5)
    _bench_variants(
        "p4", _make_specs([v1, v2, v3, v4], mode="project", strength=0.5), 4, "project"
    )

    remove_steering_patch(model)
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Model paths (under ~/.omlx/models/)")
    parser.add_argument("--steps", type=int, default=32, help="Decode steps per timed run")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup decode steps")
    args = parser.parse_args()

    home_models = Path.home() / ".omlx" / "models"

    all_results = []
    for m in args.models:
        path = m if Path(m).is_absolute() else str(home_models / m)
        try:
            all_results.append(bench_model(path, args.steps, args.warmup))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {m}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n=== Summary ===")
    for r in all_results:
        print(f"\n{r['model']}  (n_layers={r['n_layers']})")
        baseline = next(
            (c for c in r["configs"] if c["label"] == "baseline"), None
        )
        if baseline:
            print(f"  baseline           tok/s={baseline['tps']:.2f}")
        for c in r["configs"]:
            if c["label"] == "baseline":
                continue
            slowdown_pct = (
                100.0 * (baseline["best_s"] - c["best_s"]) / baseline["best_s"]
                if baseline
                else 0.0
            )
            print(
                f"  P={c.get('P', '?')}  {c.get('mode', '?'):>7s}  "
                f"{c.get('variant', '?'):>6s}  tok/s={c['tps']:6.2f}  "
                f"(vs baseline: {slowdown_pct:+.1f}%)"
            )


if __name__ == "__main__":
    main()
