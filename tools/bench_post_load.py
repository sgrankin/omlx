# SPDX-License-Identifier: Apache-2.0
"""Bench apply_post_load_transforms in PRODUCTION ORDER.

The transforms (and the always-on gated_delta_advance / qwen3_5_attention
patches that apply_post_load_transforms also runs) must be applied right
after load, before any decode — that is what the engine does. Applying
them mid-stream, after the model has already decoded, leaves stale
compiled state and mis-measures.

So this runs as two separate invocations:
    bench_post_load.py <model> baseline
    bench_post_load.py <model> transformed
Each loads fresh; `transformed` calls apply_post_load_transforms before
the first decode. Compare the two tok/s.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model, _logits


def decode_overlap(model, prompt_ids, n_steps):
    from mlx_lm.models.cache import make_prompt_cache
    tm = _text_model(model)
    cache = make_prompt_cache(tm)
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    y = mx.argmax(_logits(out)[:, -1, :][0], keepdims=True)
    mx.async_eval(y)
    toks = []
    for n in range(n_steps):
        if n + 1 != n_steps:
            out = tm(y[None, :], cache=cache)
            next_y = mx.argmax(_logits(out)[:, -1, :][0], keepdims=True)
            mx.async_eval(next_y)
        toks.append(int(y.item()))
        if n + 1 != n_steps:
            y = next_y
    return toks


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.6-35B-A3B-oQ6-fp16"
    mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    assert mode in ("baseline", "transformed")

    path = str(Path.home() / ".omlx" / "models" / model_name)
    print(f"[{mode}] loading {path}")
    model, _ = _load_model(path)

    if mode == "transformed":
        # Production order: transforms applied at load, before any decode.
        from omlx.utils.model_loading import apply_post_load_transforms
        from omlx.patches.gateup_fuse import _FUSED_ATTR
        from omlx.patches.block_compile import _COMPILED_ATTR

        apply_post_load_transforms(model, model_settings=None)
        root = getattr(_text_model(model), "model", _text_model(model))
        mods = list(root.modules())
        n_gu = sum(1 for m in mods if getattr(m, _FUSED_ATTR, None) is not None)
        n_bc = sum(1 for m in mods if getattr(m, _COMPILED_ATTR, None) is not None)
        print(f"[{mode}] gate+up fused={n_gu}  block-compiled={n_bc}")

    prompt_ids = list(range(24))
    # Heavy warmup: mx.compile tracing is one-shot, but the Metal driver
    # JIT-compiles each new pipeline-state lazily over the first many
    # dispatches. Warm hard so the timed runs reflect steady state.
    for _ in range(6):
        decode_overlap(model, prompt_ids, steps)
    runs = []
    toks = None
    for _ in range(6):
        t = time.perf_counter()
        toks = decode_overlap(model, prompt_ids, steps)
        runs.append(time.perf_counter() - t)
    best = min(runs)
    print(f"[{mode}] best={best:.3f}s  tok/s={steps/best:.2f}  "
          f"({'/'.join(f'{r:.3f}' for r in runs)})")
    print(f"[{mode}] first 12 tokens: {toks[:12]}")


if __name__ == "__main__":
    main()
