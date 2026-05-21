# SPDX-License-Identifier: Apache-2.0
"""Compare the FULL gemma4 attention forward — original vs QKV-fused —
on the same inputs with a real KV cache. Pinpoints the post-matmul bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26B-A4B-it-oQ6-fp16"
    path = str(Path.home() / ".omlx" / "models" / model_name)
    print(f"Loading {path}")
    model, _ = _load_model(path)
    tm = _text_model(model)
    layers = tm.model.layers

    from mlx_lm.models.cache import make_prompt_cache
    from tools.patch_qkv_fuse import _build_qkv_fused_call

    # Save the original __call__ before patching anything
    import mlx_vlm.models.gemma4.language as g4
    original_call = g4.Attention.__call__

    from mlx_lm.models.base import create_attention_mask

    all_ok = True
    for L in (1, 20):
        print(f"\n--- L={L} ({'decode' if L == 1 else 'prefill'}) ---")
        for i in range(min(6, len(layers))):
            attn = layers[i].self_attn
            in_dim = attn.q_proj.weight.shape[1] * 32 // attn.q_proj.bits
            x = mx.random.normal((1, L, in_dim), dtype=mx.float16)

            cache_a = make_prompt_cache(tm)[i]
            cache_b = make_prompt_cache(tm)[i]

            # Build a causal mask for L>1, matching what the model does.
            mask = None
            if L > 1:
                mask = create_attention_mask(x, cache_a)

            out_a, kv_a, off_a = original_call(
                attn, x, mask=mask, cache=cache_a, shared_kv=None, offset=None
            )
            mx.eval(out_a)

            fused = _build_qkv_fused_call(attn)
            if fused is None:
                print(f"  layer {i}: not fusable, skip")
                continue
            out_b, kv_b, off_b = fused(
                x, mask=mask, cache=cache_b, shared_kv=None, offset=None
            )
            mx.eval(out_b)

            diff = float(mx.abs(out_a - out_b).max())
            ok = diff < 1e-2
            all_ok = all_ok and ok
            print(f"  layer {i}: out diff={diff:.5f}  [{'OK' if ok else 'MISMATCH'}]")

    print(f"\n{'ALL OK' if all_ok else 'BUG in post-matmul attention forward'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
