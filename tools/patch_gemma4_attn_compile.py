# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch gemma4 Attention.__call__ to fuse pre- and post-cache compute.

self_attn body breaks into three sections:
  PRE   — q_proj/k_proj/v_proj + norms + transposes + k.rope (pure)
  CACHE — cache.update_and_fetch(K, V) (in-place mutation, uncompilable)
  POST  — q.transpose + q.rope + sdpa + transpose+reshape + o_proj (pure)

Compile PRE and POST separately, keep CACHE untouched. Only the
non-shared-kv path goes through PRE compile; the shared-kv path is small
(q_proj + q_norm) and not worth a compile.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

_PATCHED = False


def _build_attn_pre_fn(layer):
    """(x, offset) -> (q_pre, k_pre, v_pre, offset). Used when shared_kv is None."""
    q_proj = layer.q_proj
    q_norm = layer.q_norm
    n_heads = layer.n_heads
    n_kv_heads = layer.n_kv_heads
    head_dim = layer.head_dim
    use_k_eq_v = layer.use_k_eq_v
    k_proj = getattr(layer, "k_proj", None)
    v_proj = getattr(layer, "v_proj", None) if not use_k_eq_v else None
    k_norm = getattr(layer, "k_norm", None)
    v_norm = getattr(layer, "v_norm", None)
    rope = layer.rope

    def body(x, offset):
        B, L, _ = x.shape
        queries = q_proj(x).reshape(B, L, n_heads, head_dim)
        queries = q_norm(queries)

        keys = k_proj(x).reshape(B, L, n_kv_heads, head_dim)
        if use_k_eq_v:
            values = keys
        else:
            values = v_proj(x).reshape(B, L, n_kv_heads, head_dim)

        keys = k_norm(keys)
        keys = keys.transpose(0, 2, 1, 3)
        keys = rope(keys, offset=offset)

        values = v_norm(values)
        values = values.transpose(0, 2, 1, 3)

        return queries, keys, values

    return mx.compile(body)


def _build_q_rope_fn(layer):
    """queries_pre, offset -> queries with RoPE applied + transposed."""
    rope = layer.rope

    def body(queries, offset):
        queries = queries.transpose(0, 2, 1, 3)
        queries = rope(queries, offset=offset)
        return queries

    return mx.compile(body)


def _build_output_proj_fn(layer):
    """sdpa output -> o_proj(transpose+reshape(output))."""
    o_proj = layer.o_proj

    def body(output):
        B, _, L, _ = output.shape  # (B, n_heads, L, head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return o_proj(output)

    return mx.compile(body)


def apply_gemma4_attn_compile_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError:
        return False

    Attention = g4.Attention
    _CACHE: dict[int, dict] = {}

    from mlx_vlm.models.base import scaled_dot_product_attention as _sdpa

    def patched_call(self, x: mx.array, mask: Optional[mx.array] = None,
                     cache: Any = None, shared_kv: Optional[tuple] = None,
                     offset: Optional[Any] = None):
        B, L, _ = x.shape

        key = id(self)
        slot = _CACHE.get(key)
        if slot is None:
            slot = {
                "pre": _build_attn_pre_fn(self),
                "q_rope": _build_q_rope_fn(self),
                "out": _build_output_proj_fn(self),
            }
            _CACHE[key] = slot

        if shared_kv is not None:
            # Shared-kv path: PRE is just q_proj + q_norm; uncompiled.
            queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
            queries = self.q_norm(queries)
            keys, values = shared_kv
        else:
            offset = mx.array(cache.offset) if cache is not None else mx.array(0)
            queries, keys, values = slot["pre"](x, offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        # Q rope post-cache (compiled, small)
        queries = slot["q_rope"](queries, offset)
        # SDPA: uncompiled — preserves TurboQuant / quantized cache code paths
        output = _sdpa(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        # Output projection (compiled, small)
        out = slot["out"](output)
        return out, (keys, values), offset

    Attention.__call__ = patched_call
    _PATCHED = True
    return True


if __name__ == "__main__":
    apply_gemma4_attn_compile_patch()
    print("gemma4 Attention pre/post compiled")
