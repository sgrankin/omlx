# SPDX-License-Identifier: Apache-2.0
"""Wrap gemma4 ProportionalRoPE.__call__ body in mx.compile.

Profile showed RoPE was 29ms / 1290 calls = 22µs/call — high for what is
~10 slice/concat ops + 1 fast.rope kernel. Most of that 22µs is Python
dispatch overhead between ops. mx.compile collapses the chain into one
graph dispatched per call.

The function is pure (no state). Inputs: (x, offset). Both can be traced
as array inputs. Per-instance compile cache keyed on id(rope).
"""

from __future__ import annotations

import mlx.core as mx

_PATCHED = False


def apply_gemma4_rope_compile_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_vlm.models.gemma4 import rope_utils as ru
    except ImportError:
        return False

    ProportionalRoPE = ru.ProportionalRoPE
    original_call = ProportionalRoPE.__call__

    _CACHE: dict[int, mx.compile] = {}

    def _build(rope):
        # Capture per-instance constants in the closure
        dims = rope.dims
        rotated_dims = rope.rotated_dims
        traditional = rope.traditional
        freqs = rope._freqs

        def body(x, offset):
            if rotated_dims <= 0:
                return x
            head = x[..., :dims]
            tail = x[..., dims:]
            half = dims // 2

            left = head[..., :half]
            right = head[..., half:]
            rotated = mx.concatenate(
                [left[..., : rotated_dims // 2], right[..., : rotated_dims // 2]],
                axis=-1,
            )
            rotated = mx.fast.rope(
                rotated, rotated_dims, traditional=traditional,
                base=None, scale=1.0, offset=offset, freqs=freqs,
            )
            left = mx.concatenate(
                [rotated[..., : rotated_dims // 2], left[..., rotated_dims // 2 :]],
                axis=-1,
            )
            right = mx.concatenate(
                [rotated[..., rotated_dims // 2 :], right[..., rotated_dims // 2 :]],
                axis=-1,
            )
            head = mx.concatenate([left, right], axis=-1)
            if tail.shape[-1] == 0:
                return head
            return mx.concatenate([head, tail], axis=-1)

        return mx.compile(body)

    def patched_call(self, x, offset=0):
        if self.rotated_dims <= 0:
            return x
        # Normalise offset to an mx.array so the trace is shape-stable
        if not isinstance(offset, mx.array):
            offset = mx.array(offset)
        key = id(self)
        fn = _CACHE.get(key)
        if fn is None:
            fn = _build(self)
            _CACHE[key] = fn
        return fn(x, offset)

    ProportionalRoPE.__call__ = patched_call
    _PATCHED = True
    return True


if __name__ == "__main__":
    apply_gemma4_rope_compile_patch()
    print("gemma4 ProportionalRoPE wrapped in mx.compile")
