# SPDX-License-Identifier: Apache-2.0
"""Tests for block compile (omlx/patches/block_compile.py).

block_compile mx.compiles the per-layer feed-forward submodules. It is
opt-in (``ModelSettings.block_compile_enabled``, default off). These
tests exercise the compile-and-dispatch mechanism on a synthetic pure
module — fast, but they touch MLX ops, so run with the sandbox disabled.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.block_compile import (
    _COMPILED_ATTR,
    _attach_compiled,
    apply_block_compile,
)


class _PureFFN(nn.Module):
    """A stateless SwiGLU-ish feed-forward — stands in for an mlp submodule."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim, bias=False)
        self.down = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


def test_attach_compiled_is_token_identical():
    """Compiling a pure submodule's forward leaves its output unchanged."""
    mx.random.seed(0)
    m = _PureFFN()
    x = mx.random.normal((1, 4, 16))

    base = m(x)
    mx.eval(base)

    assert _attach_compiled(m) is True
    assert getattr(m, _COMPILED_ATTR, None) is not None

    fused = m(x)
    mx.eval(fused)

    assert mx.allclose(base, fused, atol=1e-5, rtol=1e-5), (
        f"max abs diff = {float(mx.abs(base - fused).max())}"
    )


def test_attach_compiled_idempotent():
    """A second _attach_compiled call on the same instance is a no-op."""
    m = _PureFFN()
    assert _attach_compiled(m) is True
    first = getattr(m, _COMPILED_ATTR)
    assert _attach_compiled(m) is False
    assert getattr(m, _COMPILED_ATTR) is first


def test_dispatch_shim_falls_through_for_uncompiled_instance():
    """The class-level shim leaves un-attached instances on the original path."""
    mx.random.seed(1)
    compiled = _PureFFN()
    _attach_compiled(compiled)  # patches _PureFFN.__call__ class-wide

    # A fresh instance of the now-patched class, with no compiled forward
    # attached, must still run via the original __call__.
    plain = _PureFFN()
    x = mx.random.normal((1, 3, 16))
    out = plain(x)
    mx.eval(out)
    assert out.shape == (1, 3, 16)


def test_apply_block_compile_noop_without_targets():
    """A model containing none of the target classes compiles nothing."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

    assert apply_block_compile(Tiny()) == 0


class _ScaledFFN(nn.Module):
    """A submodule whose __call__ takes a defaulted keyword arg."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, scale=1.0):
        return self.lin(x) * scale


def test_shim_passes_kwargs_through_to_original():
    """A keyword-arg call falls through to the original, uncompiled path."""
    mx.random.seed(2)
    m = _ScaledFFN()
    x = mx.random.normal((1, 4, 16))

    expected = m(x, scale=2.0)
    mx.eval(expected)

    assert _attach_compiled(m) is True

    out = m(x, scale=2.0)
    mx.eval(out)
    assert mx.allclose(expected, out, atol=1e-5, rtol=1e-5)


def test_shim_falls_back_on_arity_mismatch():
    """A call with fewer positional args than the compiled arity falls back."""
    mx.random.seed(3)
    m = _ScaledFFN()
    x = mx.random.normal((1, 4, 16))

    expected = m(x)  # default scale=1.0
    mx.eval(expected)

    assert _attach_compiled(m) is True  # compiled for arity 2 (x, scale)

    out = m(x)  # single positional arg — must not raise
    mx.eval(out)
    assert mx.allclose(expected, out, atol=1e-5, rtol=1e-5)


def test_block_compile_not_applied_by_default(monkeypatch):
    """apply_post_load_transforms must not compile unless explicitly enabled."""
    import omlx.patches.block_compile as bc_mod
    from omlx.utils.model_loading import apply_post_load_transforms

    calls = []

    def _counting(model):
        calls.append(1)
        return 0

    monkeypatch.setattr(bc_mod, "apply_block_compile", _counting)
    apply_post_load_transforms(object(), None)
    apply_post_load_transforms(
        object(),
        SimpleNamespace(index_cache_freq=None, steering_vectors=None),
    )

    assert calls == []


def test_block_compile_applied_when_setting_enabled(monkeypatch):
    """apply_post_load_transforms compiles when block_compile_enabled=True."""
    import omlx.patches.block_compile as bc_mod
    from omlx.utils.model_loading import apply_post_load_transforms

    calls = []

    def _counting(model):
        calls.append(1)
        return 0

    monkeypatch.setattr(bc_mod, "apply_block_compile", _counting)
    apply_post_load_transforms(
        object(),
        SimpleNamespace(
            block_compile_enabled=True,
            index_cache_freq=None,
            steering_vectors=None,
        ),
    )

    assert calls == [1]
