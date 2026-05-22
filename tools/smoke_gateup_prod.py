# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the production wiring of the decode-perf post-load transforms.

Loads a real MoE model and runs it through omlx's apply_post_load_transforms
(the actual engine load-path hook). Gate+up fusion now runs separately, in
the engine (omlx.patches.qwen35_moe_gate_up), so this only exercises block
compile here; it confirms decoded tokens are unchanged vs a pre-transform
baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model, _logits


def decode(model, prompt_ids, n):
    from mlx_lm.models.cache import make_prompt_cache
    tm = _text_model(model)
    cache = make_prompt_cache(tm)
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    nxt = int(mx.argmax(_logits(out)[0, -1]).item())
    toks = []
    for _ in range(n):
        toks.append(nxt)
        out = tm(mx.array([[nxt]]), cache=cache)
        nxt = int(mx.argmax(_logits(out)[0, -1]).item())
    return toks


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26B-A4B-it-oQ6-fp16"
    path = str(Path.home() / ".omlx" / "models" / model_name)
    print(f"Loading {path}")
    model, _ = _load_model(path)

    prompt_ids = list(range(20))
    print("Baseline decode (pre-transform)...")
    base = decode(model, prompt_ids, 32)

    # The real engine hook.
    from types import SimpleNamespace

    from omlx.utils.model_loading import apply_post_load_transforms
    from omlx.patches.block_compile import _COMPILED_ATTR

    print("Running apply_post_load_transforms ...")
    apply_post_load_transforms(
        model, model_settings=SimpleNamespace(block_compile_enabled=True)
    )

    # Count fused SwitchGLU + block-compiled feed-forward submodules.
    from mlx_lm.models.switch_layers import SwitchGLU
    tm = _text_model(model)
    root = getattr(tm, "model", tm)
    mods = list(root.modules())
    glu_total = sum(1 for m in mods if isinstance(m, SwitchGLU))
    glu_fused = sum(
        1 for m in mods if isinstance(m, SwitchGLU) and hasattr(m, "gate_up_proj")
    )
    blk = sum(1 for m in mods if getattr(m, _COMPILED_ATTR, None) is not None)
    print(f"  SwitchGLU gate+up fused: {glu_fused}/{glu_total}")
    print(f"  block-compiled submodules: {blk}")

    print("Patched decode (post-transform)...")
    patched = decode(model, prompt_ids, 32)

    if base == patched:
        print(f"\n✓ prod wiring OK: gate+up={glu_fused}, block-compile={blk}, "
              f"token-identical")
        return 0
    print("\n✗ token mismatch after apply_post_load_transforms")
    return 1


if __name__ == "__main__":
    sys.exit(main())
