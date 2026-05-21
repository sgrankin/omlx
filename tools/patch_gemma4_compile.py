# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch mlx_vlm.gemma4 DecoderLayer to fuse the post-attention chain.

The forward pass after self_attn is pure (no in-place state) and has 8+ small
op dispatches per layer (RMSNorm × 4, residual adds × 2, MLP linears × 3,
optional per-layer-input gate). At ~30 layers × ~175 dispatches/layer this
is one of the larger Python-side contributors to per-token wall time.

mx.compile traces the post-attn body once per (h-shape, residual-shape) pair
and dispatches the fused graph. The self_attn call stays out of the compiled
region because it mutates the KV cache in place — that's a side effect
mx.compile cannot capture.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

# Apply once
_PATCHED = False


def apply_gemma4_block_compile_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_vlm.models.gemma4 import language as g4
    except ImportError:
        return False

    DecoderLayer = g4.DecoderLayer

    _CACHE: dict[int, Any] = {}

    def _build_post_attn_fn(layer):
        """Capture a layer's submodules in a closure and compile the body."""
        post_attention_layernorm = layer.post_attention_layernorm
        pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        post_feedforward_layernorm = layer.post_feedforward_layernorm
        mlp = layer.mlp
        layer_scalar = layer.layer_scalar
        enable_moe = layer.enable_moe

        if enable_moe:
            router = layer.router
            experts = layer.experts
            post_feedforward_layernorm_1 = layer.post_feedforward_layernorm_1
            pre_feedforward_layernorm_2 = layer.pre_feedforward_layernorm_2
            post_feedforward_layernorm_2 = layer.post_feedforward_layernorm_2

        per_layer_input_gate = layer.per_layer_input_gate
        per_layer_projection = layer.per_layer_projection
        post_per_layer_input_norm = layer.post_per_layer_input_norm

        if enable_moe:
            def body(h_attn, residual, per_layer_input):
                h = post_attention_layernorm(h_attn)
                h = residual + h
                residual2 = h

                h1 = pre_feedforward_layernorm(h)
                h1 = mlp(h1)
                h1 = post_feedforward_layernorm_1(h1)

                top_k_indices, top_k_weights = router(h)
                h2 = pre_feedforward_layernorm_2(h)
                h2 = experts(h2, top_k_indices, top_k_weights)
                h2 = post_feedforward_layernorm_2(h2)

                h = h1 + h2
                h = post_feedforward_layernorm(h)
                h = residual2 + h

                if (
                    per_layer_input_gate is not None
                    and per_layer_projection is not None
                    and post_per_layer_input_norm is not None
                    and per_layer_input is not None
                ):
                    residual3 = h
                    gate = per_layer_input_gate(h)
                    gate = nn.gelu_approx(gate)
                    gate = mx.multiply(gate, per_layer_input)
                    gate = per_layer_projection(gate)
                    gate = post_per_layer_input_norm(gate)
                    h = residual3 + gate

                if layer_scalar is not None:
                    h = h * layer_scalar
                return h
        else:
            def body(h_attn, residual, per_layer_input):
                h = post_attention_layernorm(h_attn)
                h = residual + h
                residual2 = h

                h = pre_feedforward_layernorm(h)
                h = mlp(h)
                h = post_feedforward_layernorm(h)
                h = residual2 + h

                if (
                    per_layer_input_gate is not None
                    and per_layer_projection is not None
                    and post_per_layer_input_norm is not None
                    and per_layer_input is not None
                ):
                    residual3 = h
                    gate = per_layer_input_gate(h)
                    gate = nn.gelu_approx(gate)
                    gate = mx.multiply(gate, per_layer_input)
                    gate = per_layer_projection(gate)
                    gate = post_per_layer_input_norm(gate)
                    h = residual3 + gate

                if layer_scalar is not None:
                    h = h * layer_scalar
                return h

        return mx.compile(body)

    def patched_call(self, x, mask=None, cache=None,
                     per_layer_input=None, shared_kv=None, offset=None):
        residual = x
        h = self.input_layernorm(x)
        h, shared_kv, offset = self.self_attn(
            h, mask, cache, shared_kv=shared_kv, offset=offset
        )

        key = id(self)
        fn = _CACHE.get(key)
        if fn is None:
            fn = _build_post_attn_fn(self)
            _CACHE[key] = fn
        h = fn(h, residual, per_layer_input)
        return h, shared_kv, offset

    # Drop in.
    DecoderLayer.__call__ = patched_call
    _PATCHED = True
    return True


if __name__ == "__main__":
    apply_gemma4_block_compile_patch()
    print("gemma4 DecoderLayer post-attn body fused via mx.compile")
