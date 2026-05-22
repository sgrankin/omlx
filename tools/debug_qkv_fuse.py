# SPDX-License-Identifier: Apache-2.0
"""Numerically isolate the Q/K/V fusion bug.

Loads one gemma4 model, takes one Attention layer, and compares:
  - separate: q_proj(x), k_proj(x), v_proj(x)
  - fused:    one quantized_matmul over concatenated weights, then split

If these don't match, the bug is in the fused-weight concat or the split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model


def check_layer(attn, layer_idx: int, seq_len: int = 1) -> bool:
    use_k_eq_v = getattr(attn, "use_k_eq_v", False)
    q_proj = attn.q_proj
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if k_proj is None:
        print(f"  layer {layer_idx}: kv_shared_only, skip")
        return True

    # input dim
    in_dim = q_proj.weight.shape[1] * 32 // q_proj.bits
    x = mx.random.normal((1, seq_len, in_dim), dtype=mx.float16)

    # Separate
    q_sep = q_proj(x)
    k_sep = k_proj(x)
    v_sep = None if use_k_eq_v else v_proj(x)

    # Fused
    weights = [q_proj["weight"], k_proj["weight"]]
    scales = [q_proj["scales"], k_proj["scales"]]
    biases = [q_proj.get("biases"), k_proj.get("biases")]
    if not use_k_eq_v:
        weights.append(v_proj["weight"])
        scales.append(v_proj["scales"])
        biases.append(v_proj.get("biases"))
    fw = mx.concatenate(weights, axis=0)
    fs = mx.concatenate(scales, axis=0)
    fb = mx.concatenate(biases, axis=0) if all(b is not None for b in biases) else None

    y = mx.quantized_matmul(
        x, fw, fs, fb, transpose=True,
        group_size=q_proj.group_size, bits=q_proj.bits, mode=q_proj.mode,
    )
    q_dim = q_sep.shape[-1]
    k_dim = k_sep.shape[-1]
    q_fused = y[..., :q_dim]
    k_fused = y[..., q_dim:q_dim + k_dim]
    v_fused = None if use_k_eq_v else y[..., q_dim + k_dim:]

    q_ok = mx.allclose(q_sep, q_fused, atol=1e-2, rtol=1e-2)
    k_ok = mx.allclose(k_sep, k_fused, atol=1e-2, rtol=1e-2)
    v_ok = True if use_k_eq_v else mx.allclose(v_sep, v_fused, atol=1e-2, rtol=1e-2)

    q_diff = float(mx.abs(q_sep - q_fused).max())
    k_diff = float(mx.abs(k_sep - k_fused).max())
    v_diff = 0.0 if use_k_eq_v else float(mx.abs(v_sep - v_fused).max())

    status = "OK" if (q_ok and k_ok and v_ok) else "MISMATCH"
    print(f"  layer {layer_idx} L={x.shape[1]}: k_eq_v={use_k_eq_v} bits={q_proj.bits} "
          f"q_dim={q_dim} k_dim={k_dim}  "
          f"Q diff={q_diff:.4f} K diff={k_diff:.4f} V diff={v_diff:.4f}  [{status}]")
    return bool(q_ok and k_ok and v_ok)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26B-A4B-it-oQ6-fp16"
    path = str(Path.home() / ".omlx" / "models" / model_name)
    print(f"Loading {path}")
    model, _ = _load_model(path)
    tm = _text_model(model)
    layers = tm.model.layers

    all_ok = True
    # Check first layers at both decode (L=1) and prefill (L=20) shapes
    for seq_len in (1, 20):
        print(f"--- L={seq_len} ---")
        for i in range(min(4, len(layers))):
            attn = layers[i].self_attn
            ok = check_layer(attn, i, seq_len=seq_len)
            all_ok = all_ok and ok

    print(f"\n{'ALL OK' if all_ok else 'BUG FOUND'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
