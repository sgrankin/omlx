# SPDX-License-Identifier: Apache-2.0
"""Fuse q_proj + k_proj + v_proj into one matmul for gemma4 attention.

All three projections take the same input x. Concatenate weights/scales/
biases along the output-dim axis, do ONE mx.quantized_matmul, split the
output into Q/K/V slices.

NUMERICAL STABILITY — decode-only gating
----------------------------------------
The fused mx.quantized_matmul is BIT-EXACT vs separate matmuls at L=1
(decode) but differs by ~1-2 ulp fp16 at L>1 (prefill) — widening the
output dim changes the gemm kernel's tiling, hence the fp accumulation
order (verified in tools/debug_qkv_fuse.py: L=1 diff 0.0, L=20 diff
~0.015). That prefill perturbation compounds to token flips.

Fix: only take the fused path when L==1. Prefill (L>1) falls through to
the original separate-projection path — bit-exact, and prefill is a
one-shot cost anyway. Decode (L==1) takes the fused path. Net result:
token-identical end to end (verified).

OUTCOME: correct but NOT worth shipping. With the L==1 gate the patch is
token-identical, but benching shows it adds ~0% on top of the gate+up
fusion. Attention's q/k/v projections go through mx.quantized_matmul,
which — unlike the MoE gather_qmm — is NOT dispatch-bound (its per-call
overhead is small relative to the matmul compute). Fusing them saves a
negligible amount. The earlier "+2.5%" figure was a measurement
artifact from a broken (garbage-output) config. Kept as a documented
artifact: the L==1-gating technique is reusable, and this confirms the
gather_qmm-vs-quantized_matmul distinction.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

_PATCHED = False


def _build_qkv_fused_call(attn):
    """Return a fused-projection forward closure for one gemma4 Attention,
    or None if the layer can't be fused."""
    q_proj = attn.q_proj
    n_heads = attn.n_heads
    head_dim = attn.head_dim
    q_norm = attn.q_norm
    o_proj = attn.o_proj
    rope = attn.rope
    scale = attn.scale

    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    k_norm = getattr(attn, "k_norm", None)
    v_norm = getattr(attn, "v_norm", None)
    use_k_eq_v = getattr(attn, "use_k_eq_v", False)
    n_kv_heads = attn.n_kv_heads

    if k_proj is None:
        return None  # kv_shared_only

    quant_inputs = [q_proj, k_proj]
    if not use_k_eq_v and v_proj is not None:
        quant_inputs.append(v_proj)
    for q in quant_inputs:
        if not isinstance(q, nn.QuantizedLinear):
            return None
    g0 = quant_inputs[0]
    for q in quant_inputs[1:]:
        if (q.group_size != g0.group_size or q.bits != g0.bits
                or q.mode != g0.mode):
            return None

    weights_to_fuse = [q_proj["weight"], k_proj["weight"]]
    scales_to_fuse = [q_proj["scales"], k_proj["scales"]]
    biases_to_fuse = [q_proj.get("biases"), k_proj.get("biases")]
    if not use_k_eq_v:
        weights_to_fuse.append(v_proj["weight"])
        scales_to_fuse.append(v_proj["scales"])
        biases_to_fuse.append(v_proj.get("biases"))

    fused_weight = mx.concatenate(weights_to_fuse, axis=0)
    fused_scales = mx.concatenate(scales_to_fuse, axis=0)
    if all(b is not None for b in biases_to_fuse):
        fused_biases = mx.concatenate(biases_to_fuse, axis=0)
    elif all(b is None for b in biases_to_fuse):
        fused_biases = None
    else:
        return None

    mx.eval(fused_weight, fused_scales)
    if fused_biases is not None:
        mx.eval(fused_biases)

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    group_size = g0.group_size
    bits = g0.bits
    mode = g0.mode

    from mlx_vlm.models.base import scaled_dot_product_attention as _sdpa

    def fused_forward(x, mask=None, cache=None, shared_kv=None, offset=None):
        B, L, _ = x.shape

        if shared_kv is not None:
            queries = q_proj(x).reshape(B, L, n_heads, head_dim)
            queries = q_norm(queries)
            keys, values = shared_kv
        else:
            y = mx.quantized_matmul(
                x, fused_weight, fused_scales, fused_biases,
                transpose=True, group_size=group_size, bits=bits, mode=mode,
            )
            # Reshape the CONTIGUOUS y into (B, L, total_heads, head_dim)
            # FIRST, then slice along the heads axis. Slicing the last axis
            # of y and then reshaping is wrong for L>1: the slice is
            # non-contiguous (stride = total_dim between L positions) and
            # the reshape silently misinterprets the layout. L=1 (decode)
            # hides the bug; L>1 (prefill) corrupts Q/K/V.
            n_kv_total = n_kv_heads if use_k_eq_v else 2 * n_kv_heads
            total_heads = n_heads + n_kv_total
            y = y.reshape(B, L, total_heads, head_dim)
            queries = y[..., :n_heads, :]
            queries = q_norm(queries)
            keys = y[..., n_heads:n_heads + n_kv_heads, :]
            if use_k_eq_v:
                values = keys
            else:
                values = y[..., n_heads + n_kv_heads:, :]

            offset = mx.array(cache.offset) if cache is not None else mx.array(0)

            keys = k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = rope(keys, offset=offset)

            values = v_norm(values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = rope(queries, offset=offset)

        output = _sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return o_proj(output), (keys, values), offset

    return fused_forward


def apply_qkv_fuse_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError:
        return False

    Attention = g4.Attention
    _CACHE: dict[int, tuple] = {}
    original = Attention.__call__

    def patched(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        # Decode-only gating: the fused matmul is bit-exact at L==1 but
        # ~1ulp off at L>1. Prefill keeps the original (bit-exact) path.
        if x.shape[1] != 1:
            return original(self, x, mask, cache, shared_kv, offset)
        key = id(self)
        slot = _CACHE.get(key)
        if slot is None:
            fn = _build_qkv_fused_call(self)
            if fn is None:
                _CACHE[key] = ("fallback", None)
                return original(self, x, mask, cache, shared_kv, offset)
            _CACHE[key] = ("fused", fn)
            slot = _CACHE[key]
        tag, fn = slot
        if tag == "fallback":
            return original(self, x, mask, cache, shared_kv, offset)
        return fn(x, mask=mask, cache=cache, shared_kv=shared_kv, offset=offset)

    Attention.__call__ = patched
    _PATCHED = True
    return True


if __name__ == "__main__":
    apply_qkv_fuse_patch()
    print("gemma4 Attention QKV matmuls fused")
