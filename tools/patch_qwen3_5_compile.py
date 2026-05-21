# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch mlx_vlm.qwen3_5 DecoderLayer to fuse the post-attn body.

Qwen3_5DecoderLayer is much leaner than gemma4 — just one pre-attn
layernorm, one attention call (linear or full), one residual add,
post-attn norm, mlp, another residual add. Both linear_attn and self_attn
mutate KV cache state, so they must stay outside the compiled region.

The fusable portion is the post-attention chain:
  h = x + r
  h = h + mlp(post_attention_layernorm(h))

It's only ~4 ops, but mlp() internally is (gate_proj, up_proj, silu, mul,
down_proj) = 5 dispatches per layer. Wrapping it captures all of those.
"""

from __future__ import annotations

import mlx.core as mx

_PATCHED = False


def _build_post_attn_fn(layer):
    """Capture (mlp, post_attention_layernorm) in a closure and compile."""
    post_attention_layernorm = layer.post_attention_layernorm
    mlp = layer.mlp

    def body(x, r):
        h = x + r
        return h + mlp(post_attention_layernorm(h))

    return mx.compile(body)


def _make_patched_call(_CACHE: dict):
    def patched_call(self, x, mask=None, cache=None, position_ids=None,
                     gdn_sink=None):
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x), mask, cache, gdn_sink=gdn_sink
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)

        key = id(self)
        slot = _CACHE.get(key)
        if slot is None:
            slot = {"fn": _build_post_attn_fn(self)}
            _CACHE[key] = slot
        return slot["fn"](x, r)
    return patched_call


def apply_qwen3_5_block_compile_patch() -> bool:
    """Patch both the dense Qwen3.6 and MoE (qwen3_5_moe) DecoderLayer classes.

    Both have identical __call__ body shape, so the same compile strategy
    works for both. Dense models (e.g. Qwen3.6-27B) use
    ``mlx_vlm.models.qwen3_5.language.Qwen3_5DecoderLayer``; MoE models
    (e.g. Qwen3.6-35B-A3B) use ``mlx_vlm.models.qwen3_5_moe.language.
    Qwen3_5MoeDecoderLayer``. Patching only one means the MoE model still
    runs the unfused Python path.
    """
    global _PATCHED
    if _PATCHED:
        return False

    patched = 0
    _CACHE_DENSE: dict[int, dict] = {}
    _CACHE_MOE: dict[int, dict] = {}

    try:
        from mlx_vlm.models.qwen3_5 import language as q3_5
        q3_5.Qwen3_5DecoderLayer.__call__ = _make_patched_call(_CACHE_DENSE)
        patched += 1
    except ImportError:
        pass

    try:
        from mlx_vlm.models.qwen3_5_moe import language as q3_5_moe
        q3_5_moe.Qwen3_5MoeDecoderLayer.__call__ = _make_patched_call(_CACHE_MOE)
        patched += 1
    except ImportError:
        pass

    if patched == 0:
        return False
    _PATCHED = True
    return True


if __name__ == "__main__":
    apply_qwen3_5_block_compile_patch()
    print("qwen3_5 Qwen3_5DecoderLayer post-attn body fused via mx.compile")
