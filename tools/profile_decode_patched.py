# SPDX-License-Identifier: Apache-2.0
"""Profile decode AFTER applying the block-compile patch. Find what Python
overhead remains so we can target the next fusion candidate.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model, _logits


def decode_overlap(model, prompt_ids, n_steps):
    from mlx_lm.models.cache import make_prompt_cache
    tm = _text_model(model)
    cache = make_prompt_cache(tm)
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    logits = _logits(out)[:, -1, :]
    y = mx.argmax(logits[0], keepdims=True)
    mx.async_eval(y)
    for n in range(n_steps):
        if n + 1 != n_steps:
            out = tm(y[None, :], cache=cache)
            logits = _logits(out)[:, -1, :]
            next_y = mx.argmax(logits[0], keepdims=True)
            mx.async_eval(next_y)
        _ = int(y.item())
        if n + 1 != n_steps:
            y = next_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="gemma-4-26B-A4B-it-oQ6-fp16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--arch", choices=("gemma4", "qwen3_5"), default="gemma4")
    args = parser.parse_args()

    path = (
        args.model if Path(args.model).is_absolute()
        else str(Path.home() / ".omlx" / "models" / args.model)
    )
    print(f"Loading {path}")
    model, _ = _load_model(path)
    print(f"  n_layers={len(_text_model(model).model.layers)}")

    # Apply patch
    if args.arch == "gemma4":
        from tools.patch_gemma4_compile import apply_gemma4_block_compile_patch
        ok = apply_gemma4_block_compile_patch()
    else:
        from tools.patch_qwen3_5_compile import apply_qwen3_5_block_compile_patch
        ok = apply_qwen3_5_block_compile_patch()
    print(f"  patch applied: {ok}")

    prompt_ids = list(range(20))

    # Warmup (compile traces)
    decode_overlap(model, prompt_ids, args.warmup)

    # Time it
    t0 = time.perf_counter()
    decode_overlap(model, prompt_ids, args.steps)
    wall = time.perf_counter() - t0
    print(f"\nUntimed pass: {args.steps / wall:.2f} tok/s")

    # cProfile
    pr = cProfile.Profile()
    pr.enable()
    decode_overlap(model, prompt_ids, args.steps)
    pr.disable()

    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(args.top)
    print("\n=== cProfile (sorted by tottime) — PATCHED ===")
    print(s.getvalue())


if __name__ == "__main__":
    main()
