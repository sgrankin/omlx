# SPDX-License-Identifier: Apache-2.0
"""Tests for the MoE gate+up matmul fusion (omlx/patches/gateup_fuse.py).

The patch re-fuses SwitchGLU's gate_proj + up_proj into a single
gather_qmm. These tests use small synthetic quantized SwitchGLU layers,
so they are fast but touch MLX ops — run with the sandbox disabled on
headless macOS.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from omlx.patches.gateup_fuse import _FUSED_ATTR, apply_gateup_fusion


def _quantized_switch_glu(
    input_dims: int = 64,
    hidden_dims: int = 128,
    num_experts: int = 8,
    bits: int = 4,
    group_size: int = 64,
) -> SwitchGLU:
    """Build a SwitchGLU and quantize its projections."""
    glu = SwitchGLU(input_dims, hidden_dims, num_experts)
    nn.quantize(glu, group_size=group_size, bits=bits)
    # Sanity: quantization actually produced QuantizedSwitchLinear.
    assert isinstance(glu.gate_proj, QuantizedSwitchLinear)
    assert isinstance(glu.up_proj, QuantizedSwitchLinear)
    return glu


class _Holder(nn.Module):
    """Minimal model wrapper so apply_gateup_fusion's walk finds the GLU."""

    def __init__(self, glu: SwitchGLU):
        super().__init__()
        self.experts = glu


def _random_call_args(input_dims: int, num_experts: int, top_k: int = 2):
    mx.random.seed(0)
    x = mx.random.normal((1, 5, input_dims), dtype=mx.float32)
    # Valid expert indices in [0, num_experts).
    indices = mx.random.randint(0, num_experts, (1, 5, top_k))
    return x, indices


def test_gateup_fusion_matches_unfused():
    """Fused SwitchGLU output is numerically identical to the unfused path."""
    input_dims, num_experts = 64, 8
    glu = _quantized_switch_glu(input_dims=input_dims, num_experts=num_experts)
    x, indices = _random_call_args(input_dims, num_experts)

    # Baseline — un-fused instance (no _FUSED_ATTR set yet).
    baseline = glu(x, indices)
    mx.eval(baseline)

    n = apply_gateup_fusion(_Holder(glu))
    assert n == 1, "expected exactly one SwitchGLU to be fused"
    assert getattr(glu, _FUSED_ATTR, None) is not None

    fused = glu(x, indices)
    mx.eval(fused)

    # Same quantized weights, same gather_qmm math, just concatenated —
    # the result is bit-exact.
    assert mx.allclose(baseline, fused, atol=1e-5, rtol=1e-5), (
        f"max abs diff = {float(mx.abs(baseline - fused).max())}"
    )


def test_gateup_fusion_noop_on_dense_model():
    """A model with no SwitchGLU is left untouched (returns 0)."""

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

    assert apply_gateup_fusion(Dense()) == 0


def test_gateup_fusion_drops_original_projections():
    """After fusion the original gate_proj / up_proj are released.

    The fused weight is a full copy of both projections — keeping the
    originals resident would duplicate that much expert-weight memory.
    """
    glu = _quantized_switch_glu()
    assert apply_gateup_fusion(_Holder(glu)) == 1
    assert "gate_proj" not in glu
    assert "up_proj" not in glu
    # down_proj is still used by the fused forward and must survive.
    assert "down_proj" in glu


def test_gateup_fusion_idempotent():
    """Re-applying does not rebuild an already-fused SwitchGLU."""
    glu = _quantized_switch_glu()
    holder = _Holder(glu)

    assert apply_gateup_fusion(holder) == 1
    first = getattr(glu, _FUSED_ATTR)
    # Second application sees the instance already fused → fuses nothing new.
    assert apply_gateup_fusion(holder) == 0
    assert getattr(glu, _FUSED_ATTR) is first


def test_gateup_fusion_skips_unquantized():
    """A SwitchGLU with non-quantized projections is left on the original path."""
    glu = SwitchGLU(64, 128, 8)  # not quantized
    assert apply_gateup_fusion(_Holder(glu)) == 0
    assert getattr(glu, _FUSED_ATTR, None) is None
    # And it still runs — output is (batch, tokens, top_k, input_dims).
    x, indices = _random_call_args(64, 8, top_k=2)
    out = glu(x, indices)
    mx.eval(out)
    assert out.shape == (1, 5, 2, 64)
