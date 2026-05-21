# SPDX-License-Identifier: Apache-2.0
"""Fuse gate_proj + up_proj matmuls in gemma4 / qwen3_5 MLP + SwitchGLU.

Both up_proj and gate_proj take the same input x. The HF checkpoint ships
them as one fused `gate_up_proj` tensor; mlx-vlm splits it on load. We
undo that split: concatenate the on-disk weights/scales/biases along the
output-dim axis, do ONE matmul, split the output into gate/up halves.

EAGER BUILD: the fused weights are concatenated + materialized
(`mx.eval`) when ``apply_gemma4_gateup_fuse_patch(model)`` is called,
walking the loaded model. This MUST NOT be done lazily on first
__call__: gemma4 MoE blocks call both ``self.mlp`` (dense) and
``self.experts`` inside the post-attn body, and when that body is itself
wrapped in ``mx.compile`` (the block-compile patch) the first call —
hence a lazy build — would run ``mx.eval`` mid-trace and corrupt the
compiled graph. Eager build sidesteps that entirely.

Patches:
  - mlx_lm.switch_layers.SwitchGLU       — gemma4 Experts, qwen3_5_moe.
    Always on. Token-identical, the big win (~+10% on MoE models).
  - mlx_vlm.models.gemma4.language.MLP / qwen3_5.Qwen3_5MLP — dense MLP.
    GATED OFF by default; enable with env OMLX_GATEUP_DENSE=1.

Why the dense path is gated off: gemma4 MoE blocks call self.mlp (dense)
*inside* the post-attn body, which the block-compile patch wraps in
mx.compile. The dense fusion's mx.quantized_matmul, traced inside that
compiled body, produces corrupt output (garbage tokens) — an unresolved
mx.compile interaction specific to quantized_matmul (gather_qmm in the
MoE path does not hit it). Dense fusion is token-correct STANDALONE
(verified on gemma 31B / Qwen3.6-27B without block-compile) but only
worth +0.4-1.5% there, so it is not worth the interaction risk. Left in,
env-gated, as a documented artifact.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

_PATCHED = False

# Attribute name under which the eagerly-built fused forward is stored on
# each MLP / SwitchGLU instance (via object.__setattr__ to bypass nn.Module).
_FUSED_ATTR = "_omlx_gateup_fused"


def _quant_ok(a, b) -> bool:
    return (
        a.group_size == b.group_size
        and a.bits == b.bits
        and a.mode == b.mode
    )


def _build_switchglu_fused(switch_glu) -> bool:
    """Eagerly build + attach a fused forward to a SwitchGLU instance.
    Returns True if fused, False if left on the original path."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    gate_proj = switch_glu.gate_proj
    up_proj = switch_glu.up_proj
    down_proj = switch_glu.down_proj
    activation = switch_glu.activation

    if not (isinstance(gate_proj, QuantizedSwitchLinear)
            and isinstance(up_proj, QuantizedSwitchLinear)):
        return False
    if not _quant_ok(gate_proj, up_proj):
        return False

    group_size = gate_proj.group_size
    bits = gate_proj.bits
    mode = gate_proj.mode

    # Concatenate along the per-expert output-dim axis (axis=1).
    fused_weight = mx.concatenate([gate_proj["weight"], up_proj["weight"]], axis=1)
    fused_scales = mx.concatenate([gate_proj["scales"], up_proj["scales"]], axis=1)
    g_b, u_b = gate_proj.get("biases"), up_proj.get("biases")
    if g_b is not None and u_b is not None:
        fused_biases = mx.concatenate([g_b, u_b], axis=1)
    elif g_b is None and u_b is None:
        fused_biases = None
    else:
        return False
    # Materialize NOW — outside any compile trace.
    mx.eval(fused_weight, fused_scales)
    if fused_biases is not None:
        mx.eval(fused_biases)

    gate_dim = gate_proj.output_dims

    def fused_forward(x, indices):
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        y = mx.gather_qmm(
            x, fused_weight, fused_scales, fused_biases,
            rhs_indices=idx, transpose=True,
            group_size=group_size, bits=bits, mode=mode,
            sorted_indices=do_sort,
        )
        x_gate = y[..., :gate_dim]
        x_up = y[..., gate_dim:]
        x = down_proj(activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    object.__setattr__(switch_glu, _FUSED_ATTR, fused_forward)
    return True


def _build_dense_mlp_fused(mlp, activation_fn) -> bool:
    """Eagerly build + attach a fused forward to a dense MLP instance.
    ``activation_fn(gate, up)`` is the model's gated activation."""
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj

    if not (isinstance(gate_proj, nn.QuantizedLinear)
            and isinstance(up_proj, nn.QuantizedLinear)):
        return False
    if not _quant_ok(gate_proj, up_proj):
        return False

    group_size = gate_proj.group_size
    bits = gate_proj.bits
    mode = gate_proj.mode

    fused_weight = mx.concatenate([gate_proj["weight"], up_proj["weight"]], axis=0)
    fused_scales = mx.concatenate([gate_proj["scales"], up_proj["scales"]], axis=0)
    g_b, u_b = gate_proj.get("biases"), up_proj.get("biases")
    if g_b is not None and u_b is not None:
        fused_biases = mx.concatenate([g_b, u_b], axis=0)
    elif g_b is None and u_b is None:
        fused_biases = None
    else:
        return False
    mx.eval(fused_weight, fused_scales)
    if fused_biases is not None:
        mx.eval(fused_biases)

    gate_dim = gate_proj.weight.shape[0]

    def fused_forward(x):
        y = mx.quantized_matmul(
            x, fused_weight, fused_scales, fused_biases,
            transpose=True, group_size=group_size, bits=bits, mode=mode,
        )
        gate_part = y[..., :gate_dim]
        up_part = y[..., gate_dim:]
        return down_proj(activation_fn(gate_part, up_part))

    object.__setattr__(mlp, _FUSED_ATTR, fused_forward)
    return True


def apply_gemma4_gateup_fuse_patch(model) -> bool:
    """Install gate+up fusion. Requires the loaded ``model`` so fused
    weights can be built eagerly (see module docstring for why lazy
    building is unsafe under the block-compile patch).

    Patches the relevant __call__ methods at class level (idempotent) and
    eagerly attaches a fused forward to every applicable instance reached
    from ``model``.
    """
    global _PATCHED

    # --- class-level __call__ patches (install once) ---
    if not _PATCHED:
        try:
            from mlx_lm.models.switch_layers import SwitchGLU
            _orig_sg = SwitchGLU.__call__

            def sg_call(self, x, indices):
                fn = getattr(self, _FUSED_ATTR, None)
                if fn is not None:
                    return fn(x, indices)
                return _orig_sg(self, x, indices)

            SwitchGLU.__call__ = sg_call
        except ImportError:
            pass

        try:
            from mlx_vlm.models.gemma4 import language as g4
            _orig_g4mlp = g4.MLP.__call__

            def g4mlp_call(self, x):
                fn = getattr(self, _FUSED_ATTR, None)
                if fn is not None:
                    return fn(x)
                return _orig_g4mlp(self, x)

            g4.MLP.__call__ = g4mlp_call
        except ImportError:
            pass

        try:
            from mlx_vlm.models.qwen3_5 import language as q35
            _orig_q35mlp = q35.Qwen3_5MLP.__call__

            def q35mlp_call(self, x):
                fn = getattr(self, _FUSED_ATTR, None)
                if fn is not None:
                    return fn(x)
                return _orig_q35mlp(self, x)

            q35.Qwen3_5MLP.__call__ = q35mlp_call
        except ImportError:
            pass

        _PATCHED = True

    # --- eager per-instance build by walking the model ---
    try:
        from mlx_lm.models.switch_layers import SwitchGLU
    except ImportError:
        SwitchGLU = None
    try:
        from mlx_vlm.models.gemma4 import language as g4
        from mlx_vlm.models.gemma4.language import geglu as _geglu
    except ImportError:
        g4 = None
        _geglu = None
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
        from mlx_vlm.models.qwen3_5.language import swiglu as _swiglu
    except ImportError:
        q35 = None
        _swiglu = None

    import os
    include_dense = os.environ.get("OMLX_GATEUP_DENSE", "0") == "1"

    n_fused = 0
    text_model = getattr(model, "language_model", None) or model
    root = getattr(text_model, "model", text_model)
    for module in root.modules() if hasattr(root, "modules") else []:
        cls_name = type(module).__name__
        if SwitchGLU is not None and isinstance(module, SwitchGLU):
            if _build_switchglu_fused(module):
                n_fused += 1
        elif not include_dense:
            continue
        elif g4 is not None and isinstance(module, g4.MLP) and _geglu is not None:
            if _build_dense_mlp_fused(module, _geglu):
                n_fused += 1
        elif (q35 is not None and isinstance(module, q35.Qwen3_5MLP)
              and _swiglu is not None):
            if _build_dense_mlp_fused(module, _swiglu):
                n_fused += 1

    return n_fused > 0


if __name__ == "__main__":
    print("call apply_gemma4_gateup_fuse_patch(model) with a loaded model")
