# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the production wiring of gate+up fusion.

Loads a real MoE model and runs it through omlx's apply_post_load_transforms
(the actual engine load-path hook), confirming the fusion fires and the
decoded tokens are unchanged vs a pre-transform baseline.
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
    from omlx.utils.model_loading import apply_post_load_transforms
    from omlx.patches.gateup_fuse import _FUSED_ATTR

    print("Running apply_post_load_transforms ...")
    apply_post_load_transforms(model, model_settings=None)

    # Count fused SwitchGLU instances.
    from mlx_lm.models.switch_layers import SwitchGLU
    tm = _text_model(model)
    root = getattr(tm, "model", tm)
    fused = sum(
        1 for m in root.modules()
        if isinstance(m, SwitchGLU) and getattr(m, _FUSED_ATTR, None) is not None
    )
    total = sum(1 for m in root.modules() if isinstance(m, SwitchGLU))
    print(f"  SwitchGLU fused: {fused}/{total}")

    print("Patched decode (post-transform)...")
    patched = decode(model, prompt_ids, 32)

    if base == patched:
        print(f"\n✓ prod wiring OK: {fused} layers fused, token-identical")
        return 0
    print("\n✗ token mismatch after apply_post_load_transforms")
    return 1


if __name__ == "__main__":
    sys.exit(main())
