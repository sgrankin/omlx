# SPDX-License-Identifier: Apache-2.0
"""Fuse gate_proj + up_proj matmuls in MoE ``SwitchGLU`` layers.

mlx-vlm's loader splits the HF checkpoint's fused ``gate_up_proj`` tensor
into separate ``gate_proj`` / ``up_proj`` projections so they fit
``SwitchGLU``'s API. At decode that costs two sequential
``mx.gather_qmm`` calls per MoE layer per token, even though both
projections take the same input ``x``.

This patch re-fuses them. At load time it walks the model, and for every
``SwitchGLU`` whose gate/up projections are quantized with matching
parameters it concatenates ``gate_proj`` + ``up_proj``
weights/scales/biases along the per-expert output-dim axis, materialises
the result once, and installs a forward that does ONE ``gather_qmm`` and
splits the output into the gate/up halves.

The original ``gate_proj`` / ``up_proj`` submodules are then dropped. The
fused weight is a full copy of both projections, so keeping the originals
around as well would leave that much expert-weight memory duplicated for
the model's lifetime (~16 GB on a 35B oQ6 MoE).

``gather_qmm`` has a high per-call fixed cost (~150µs on an M1), so
removing one dispatch per MoE layer per token is worth ~10% decode
throughput on MoE models (measured: Qwen3.6-35B-A3B +9.4%,
Gemma 4 26B-A4B-it +11.9%). The fused result is token-identical to the
unfused path — the concatenated ``gather_qmm`` is bit-exact vs the two
separate calls.

Auto-detected and safe: dense models have no ``SwitchGLU`` and are left
untouched; a ``SwitchGLU`` whose projections are not quantized, or whose
gate/up quant parameters differ, is skipped and keeps the original path.
The dense-MLP equivalent is deliberately NOT fused — ``mx.quantized_matmul``
(dense) is compute-bound, not dispatch-bound, so fusing it gives no
speedup (and interacts badly with mx.compile); only the MoE
``gather_qmm`` path benefits.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Per-instance attribute holding the fused forward closure. Set via
# object.__setattr__ so mlx's nn.Module does not treat it as a parameter.
_FUSED_ATTR = "_omlx_gateup_fused"

# The SwitchGLU.__call__ class-level patch is installed at most once.
_CLASS_PATCHED = False


def _quant_params_match(a: Any, b: Any) -> bool:
    return (
        a.group_size == b.group_size
        and a.bits == b.bits
        and a.mode == b.mode
    )


def _build_switchglu_fused(switch_glu: Any) -> bool:
    """Build + attach a fused gate+up forward to one SwitchGLU instance.

    Returns True if the instance was fused, False if it was left on the
    original path (non-quantized or mismatched-quantization projections).
    """
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    gate_proj = switch_glu.gate_proj
    up_proj = switch_glu.up_proj
    down_proj = switch_glu.down_proj
    activation = switch_glu.activation

    if not (
        isinstance(gate_proj, QuantizedSwitchLinear)
        and isinstance(up_proj, QuantizedSwitchLinear)
    ):
        return False
    if not _quant_params_match(gate_proj, up_proj):
        return False

    group_size = gate_proj.group_size
    bits = gate_proj.bits
    mode = gate_proj.mode

    # Concatenate along the per-expert output-dim axis (axis=1):
    #   weight  (num_experts, out_dim, packed_in)
    #   scales  (num_experts, out_dim, n_groups)
    fused_weight = mx.concatenate([gate_proj["weight"], up_proj["weight"]], axis=1)
    fused_scales = mx.concatenate([gate_proj["scales"], up_proj["scales"]], axis=1)
    g_biases = gate_proj.get("biases")
    u_biases = up_proj.get("biases")
    if g_biases is not None and u_biases is not None:
        fused_biases = mx.concatenate([g_biases, u_biases], axis=1)
    elif g_biases is None and u_biases is None:
        fused_biases = None
    else:
        # Mixed bias presence across gate/up — bail rather than guess.
        return False

    # Materialise the fused tensors now, at load time — never lazily on a
    # first forward call, which could land inside an mx.compile trace.
    mx.eval(fused_weight, fused_scales)
    if fused_biases is not None:
        mx.eval(fused_biases)

    gate_dim = gate_proj.output_dims

    def fused_forward(x: mx.array, indices: mx.array) -> mx.array:
        # Mirrors mlx_lm.models.switch_layers.SwitchGLU.__call__, with the
        # two gate/up gather_qmm calls collapsed into one.
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        y = mx.gather_qmm(
            x,
            fused_weight,
            fused_scales,
            fused_biases,
            rhs_indices=idx,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=do_sort,
        )
        x_gate = y[..., :gate_dim]
        x_up = y[..., gate_dim:]
        x = down_proj(
            activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    object.__setattr__(switch_glu, _FUSED_ATTR, fused_forward)

    # Drop the now-dead originals. fused_weight is a full copy of
    # gate_proj + up_proj; keeping the originals resident as well would
    # duplicate that much expert-weight memory for the model's lifetime.
    # fused_forward closes over fused_weight / down_proj / activation
    # only — it never reaches gate_proj / up_proj again.
    del switch_glu.gate_proj
    del switch_glu.up_proj
    return True


def _patch_switchglu_class() -> bool:
    """Install the SwitchGLU.__call__ dispatch shim (idempotent)."""
    global _CLASS_PATCHED
    if _CLASS_PATCHED:
        return True

    try:
        from mlx_lm.models.switch_layers import SwitchGLU
    except ImportError:
        return False

    original_call = SwitchGLU.__call__

    def patched_call(self, x, indices):
        fused = getattr(self, _FUSED_ATTR, None)
        if fused is not None:
            return fused(x, indices)
        return original_call(self, x, indices)

    SwitchGLU.__call__ = patched_call
    _CLASS_PATCHED = True
    return True


def apply_gateup_fusion(model: Any) -> int:
    """Fuse gate+up in every eligible SwitchGLU reachable from ``model``.

    Installs the class-level dispatch shim once, then eagerly builds and
    attaches a fused forward to each eligible ``SwitchGLU`` instance.

    Args:
        model: A loaded mlx-lm / mlx-vlm model instance.

    Returns:
        The number of ``SwitchGLU`` instances fused (0 for dense models or
        when mlx-lm's switch layers are unavailable).
    """
    if not _patch_switchglu_class():
        return 0

    try:
        from mlx_lm.models.switch_layers import SwitchGLU
    except ImportError:
        return 0

    # Reach the transformer body on both mlx-lm and mlx-vlm model shapes.
    text_model = getattr(model, "language_model", None) or model
    root = getattr(text_model, "model", text_model)
    if not hasattr(root, "modules"):
        return 0

    n_fused = 0
    for module in root.modules():
        if not isinstance(module, SwitchGLU):
            continue
        # Idempotent: a re-load / re-transform should not rebuild.
        if getattr(module, _FUSED_ATTR, None) is not None:
            continue
        if _build_switchglu_fused(module):
            n_fused += 1
    return n_fused
