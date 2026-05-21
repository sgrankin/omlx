# SPDX-License-Identifier: Apache-2.0
"""A/B bench for a block-compile patch on either gemma4 or qwen3_5.

Runs a decode loop unpatched (baseline), then patched, reports tok/s for
each, and runs a token-equivalence check. The two architectures use
different mlx-vlm packages so we pick the patch by --arch.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _logits, _text_model


def reload_gemma4():
    """Re-import gemma4 language module to discard any prior patch."""
    mods = [m for m in list(sys.modules) if "gemma4" in m]
    for m in mods:
        sys.modules.pop(m, None)


def decode_overlap(model, prompt_ids, n_steps):
    from mlx_lm.models.cache import make_prompt_cache
    tm = _text_model(model)
    cache = make_prompt_cache(tm)
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    logits = _logits(out)[:, -1, :]
    y = mx.argmax(logits[0], keepdims=True)
    mx.async_eval(y)
    tokens = []
    for n in range(n_steps):
        if n + 1 != n_steps:
            out = tm(y[None, :], cache=cache)
            logits = _logits(out)[:, -1, :]
            next_y = mx.argmax(logits[0], keepdims=True)
            mx.async_eval(next_y)
        tokens.append(int(y.item()))
        if n + 1 != n_steps:
            y = next_y
    return tokens


def time_runs(model, prompt_ids, n_steps, label, runs=3):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        tokens = decode_overlap(model, prompt_ids, n_steps)
        times.append(time.perf_counter() - t)
    best = min(times)
    print(f"  {label:<14s}  best={best:.3f}s  ({'/'.join(f'{t:.3f}' for t in times)})  "
          f"tok/s={n_steps / best:.2f}")
    return best, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="gemma-4-26B-A4B-it-oQ6-fp16")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--arch", choices=("gemma4", "qwen3_5"), default="gemma4")
    parser.add_argument("--with-attn", action="store_true",
                        help="Also apply attention pre/post compile patch (gemma4 only)")
    parser.add_argument("--with-rope", action="store_true",
                        help="Also apply RoPE compile patch (gemma4 only)")
    parser.add_argument("--with-gateup", action="store_true",
                        help="Also apply gate+up matmul fusion (any model with SwitchGLU)")
    args = parser.parse_args()

    path = (
        args.model if Path(args.model).is_absolute()
        else str(Path.home() / ".omlx" / "models" / args.model)
    )

    # Load once, then test both with and without the patch.
    print(f"Loading {path}")
    t = time.perf_counter()
    model, _ = _load_model(path)
    print(f"  loaded in {time.perf_counter() - t:.1f}s")

    n_layers = len(_text_model(model).model.layers)
    print(f"  n_layers={n_layers}")

    # Build a simple prompt
    prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # tiny token sequence

    # Warmup
    decode_overlap(model, prompt_ids, args.warmup)

    print("\n=== A) BASELINE (unpatched) ===")
    base_time, base_tokens = time_runs(model, prompt_ids, args.steps, "baseline")

    print("\n=== B) PATCHED (mx.compile block body) ===")
    if args.arch == "gemma4":
        from tools.patch_gemma4_compile import apply_gemma4_block_compile_patch as apply_patch
    elif args.arch == "qwen3_5":
        from tools.patch_qwen3_5_compile import apply_qwen3_5_block_compile_patch as apply_patch
    else:
        raise ValueError(f"unknown arch: {args.arch}")
    ok = apply_patch()
    print(f"  patch applied: {ok}")
    if args.with_attn and args.arch == "gemma4":
        from tools.patch_gemma4_attn_compile import apply_gemma4_attn_compile_patch
        ok2 = apply_gemma4_attn_compile_patch()
        print(f"  attn patch applied: {ok2}")
    if args.with_rope and args.arch == "gemma4":
        from tools.patch_gemma4_rope_compile import apply_gemma4_rope_compile_patch
        ok3 = apply_gemma4_rope_compile_patch()
        print(f"  rope patch applied: {ok3}")
    if args.with_gateup:
        from tools.patch_gemma4_gateup_fuse import apply_gemma4_gateup_fuse_patch
        ok4 = apply_gemma4_gateup_fuse_patch(model)
        print(f"  gate+up fusion applied: {ok4}")

    # Compile invalidates trace cache; do a warmup with the new path
    decode_overlap(model, prompt_ids, args.warmup)
    patched_time, patched_tokens = time_runs(model, prompt_ids, args.steps, "patched")

    print("\n=== C) Equivalence check ===")
    if base_tokens == patched_tokens:
        print("  ✓ token-identical")
    else:
        first_diff = next(
            (i for i in range(min(len(base_tokens), len(patched_tokens)))
             if base_tokens[i] != patched_tokens[i]),
            None,
        )
        print(f"  ✗ DIVERGE at index {first_diff}")
        print(f"    baseline[:10] = {base_tokens[:10]}")
        print(f"    patched [:10] = {patched_tokens[:10]}")

    delta_pct = 100.0 * (base_time - patched_time) / base_time
    print(f"\n=== Summary ===")
    print(f"  baseline: {args.steps / base_time:.2f} tok/s")
    print(f"  patched:  {args.steps / patched_time:.2f} tok/s")
    print(f"  delta:    {delta_pct:+.1f}%  (patched is {'faster' if delta_pct > 0 else 'slower'})")


if __name__ == "__main__":
    main()
