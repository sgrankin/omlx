# SPDX-License-Identifier: Apache-2.0
"""Isolate the QKV-fusion patch: decode the same prompt with ONLY the
qkv patch applied (no block-compile, no gate+up), diff against baseline.
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
    logits = _logits(out)[:, -1, :]
    nxt = int(mx.argmax(logits[0]).item())
    toks = []
    for _ in range(n):
        toks.append(nxt)
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())
    return toks


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26B-A4B-it-oQ6-fp16"
    path = str(Path.home() / ".omlx" / "models" / model_name)
    print(f"Loading {path}")
    model, _ = _load_model(path)

    prompt_ids = list(range(20))

    print("Baseline decode...")
    base = decode(model, prompt_ids, 24)

    print("Applying QKV patch...")
    from tools.patch_qkv_fuse import apply_qkv_fuse_patch
    ok = apply_qkv_fuse_patch()
    print(f"  applied: {ok}")

    print("Patched decode...")
    patched = decode(model, prompt_ids, 24)

    if base == patched:
        print("\n✓ QKV-only: token-identical")
        return 0
    diff = next((i for i in range(min(len(base), len(patched)))
                 if base[i] != patched[i]), None)
    print(f"\n✗ QKV-only DIVERGES at index {diff}")
    print(f"  baseline: {base}")
    print(f"  patched:  {patched}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
