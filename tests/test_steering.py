# SPDX-License-Identifier: Apache-2.0
"""Tests for steering (control) vector support.

Covers the native file format, the residual-stream patch, and the
contrastive generator. All tests use synthetic models, so they are fast
but do touch MLX ops — run with the sandbox disabled on headless macOS.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.patches.steering import (
    _SteeredLayer,
    apply_steering_patch,
    find_layers_container,
    model_hidden_size,
    remove_steering_patch,
)
from omlx.steering import STEERING_FORMAT_VERSION, SteeringVector
from omlx.steering_generator import generate_steering_vector

N_EMBD = 8
N_LAYERS = 4


# ---------------------------------------------------------------------------
# Synthetic model scaffolding
# ---------------------------------------------------------------------------


class FakeBlock:
    """A transformer block stand-in: applies a fixed per-layer bias."""

    def __init__(self, n_embd: int, seed: int):
        self.bias = mx.arange(n_embd, dtype=mx.float32) * 0.0 + float(seed)

    def __call__(self, h, *args, **kwargs):
        return h + self.bias


class TupleBlock:
    """A block that returns (hidden, aux) — some architectures do."""

    def __call__(self, h, *args, **kwargs):
        return h, "aux"


class FakeInner:
    def __init__(self, n_layers: int, n_embd: int):
        self.layers = [FakeBlock(n_embd, seed=i + 1) for i in range(n_layers)]


class FakeModel:
    """Minimal mlx-lm-shaped model: ``model.model.layers`` + embedding."""

    def __init__(self, n_layers: int, n_embd: int, vocab: int = 256):
        self.model = FakeInner(n_layers, n_embd)
        self.args = SimpleNamespace(hidden_size=n_embd)
        # Deterministic embedding table so hidden states are reproducible.
        self._emb = mx.arange(vocab * n_embd, dtype=mx.float32).reshape(
            vocab, n_embd
        )

    def make_cache(self):
        return [None] * len(self.model.layers)

    def __call__(self, tokens, cache=None):
        h = self._emb[tokens]  # [1, seq, n_embd]
        for layer in self.model.layers:
            h = layer(h)
        return h


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text] or [0]


class RealBlock(nn.Module):
    """A genuine nn.Module block, for parameter-tracking tests."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, h, *args, **kwargs):
        return self.proj(h)


class RealModel(nn.Module):
    """An nn.Module model with real-parameter blocks at ``model.layers``."""

    def __init__(self, n_layers: int, n_embd: int):
        super().__init__()
        self.layers = [RealBlock(n_embd) for _ in range(n_layers)]
        self.args = SimpleNamespace(hidden_size=n_embd)


# ---------------------------------------------------------------------------
# SteeringVector — native file I/O
# ---------------------------------------------------------------------------


def _sample_vector() -> SteeringVector:
    directions = {
        1: mx.arange(N_EMBD, dtype=mx.float32),
        2: mx.arange(N_EMBD, dtype=mx.float32) * -1.0,
    }
    return SteeringVector(directions, n_embd=N_EMBD, method="pca", model="fake")


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "vec.safetensors"
    original = _sample_vector()
    original.save(path)

    loaded = SteeringVector.load(path)
    assert loaded.n_embd == N_EMBD
    assert loaded.method == "pca"
    assert loaded.model == "fake"
    assert loaded.layers == [1, 2]
    for il in (1, 2):
        assert mx.allclose(loaded.directions[il], original.directions[il])


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SteeringVector.load(tmp_path / "nope.safetensors")


def test_save_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        SteeringVector(directions={}, n_embd=N_EMBD).save("/tmp/unused.safetensors")


def test_post_init_infers_n_embd():
    sv = SteeringVector(directions={3: mx.zeros(N_EMBD)})
    assert sv.n_embd == N_EMBD


def test_post_init_rejects_mismatched_size():
    with pytest.raises(ValueError, match="size"):
        SteeringVector(directions={1: mx.zeros(N_EMBD)}, n_embd=N_EMBD + 1)


def test_post_init_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        SteeringVector(directions={1: mx.zeros((2, N_EMBD))}, n_embd=N_EMBD)


def test_format_version_is_stored(tmp_path):
    path = tmp_path / "vec.safetensors"
    _sample_vector().save(path)
    _arrays, metadata = mx.load(str(path), return_metadata=True)
    assert metadata["omlx_steering_version"] == str(STEERING_FORMAT_VERSION)


def test_layer_map_scales_and_filters():
    sv = SteeringVector(
        directions={il: mx.ones(N_EMBD) for il in range(N_LAYERS)},
        n_embd=N_EMBD,
    )
    mapped = sv.layer_map(strength=2.0, layer_start=1, layer_end=2)
    assert sorted(mapped) == [1, 2]
    assert mx.allclose(mapped[1], mx.full((N_EMBD,), 2.0))


def test_layer_map_unbounded():
    sv = SteeringVector(
        directions={il: mx.ones(N_EMBD) for il in range(N_LAYERS)},
        n_embd=N_EMBD,
    )
    assert sorted(sv.layer_map()) == list(range(N_LAYERS))


# ---------------------------------------------------------------------------
# Steering patch — application / removal
# ---------------------------------------------------------------------------


def test_find_layers_container_nested():
    model = FakeModel(N_LAYERS, N_EMBD)
    container = find_layers_container(model)
    assert container is model.model
    assert len(container.layers) == N_LAYERS


def test_model_hidden_size():
    assert model_hidden_size(FakeModel(N_LAYERS, N_EMBD)) == N_EMBD


def test_apply_patch_adds_direction():
    model = FakeModel(N_LAYERS, N_EMBD)
    direction = mx.arange(N_EMBD, dtype=mx.float32) + 1.0
    patched = apply_steering_patch(model, {2: direction})
    assert patched == 1
    assert model._omlx_steering_active is True

    layer = model.model.layers[2]
    assert isinstance(layer, _SteeredLayer)
    h = mx.zeros((1, 1, N_EMBD))
    out = layer(h)
    # FakeBlock for layer index 2 has bias seed == 3.
    assert mx.allclose(out, h + 3.0 + direction)


def test_patch_handles_tuple_output():
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[1] = TupleBlock()
    direction = mx.ones(N_EMBD)
    apply_steering_patch(model, {1: direction})

    h = mx.zeros((1, 1, N_EMBD))
    out = model.model.layers[1](h)
    assert isinstance(out, tuple)
    assert mx.allclose(out[0], h + 1.0)
    assert out[1] == "aux"


def test_patch_preserves_hidden_dtype():
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[0] = TupleBlock()  # identity-ish, returns input dtype
    apply_steering_patch(model, {0: mx.ones(N_EMBD, dtype=mx.float32)})
    out, _aux = model.model.layers[0](mx.zeros((1, 1, N_EMBD), dtype=mx.bfloat16))
    assert out.dtype == mx.bfloat16


def test_remove_patch_restores_blocks():
    model = FakeModel(N_LAYERS, N_EMBD)
    originals = list(model.model.layers)
    apply_steering_patch(model, {1: mx.ones(N_EMBD), 3: mx.ones(N_EMBD)})
    assert remove_steering_patch(model) is True
    assert model.model.layers == originals
    assert model._omlx_steering_active is False
    # Removing again is a no-op.
    assert remove_steering_patch(model) is False


def test_apply_patch_is_idempotent():
    """Re-applying replaces the prior patch rather than nesting wrappers."""
    model = FakeModel(N_LAYERS, N_EMBD)
    apply_steering_patch(model, {1: mx.ones(N_EMBD)})
    apply_steering_patch(model, {1: mx.ones(N_EMBD) * 5.0})
    layer = model.model.layers[1]
    assert isinstance(layer, _SteeredLayer)
    assert not isinstance(layer["block"], _SteeredLayer)


def test_patch_keeps_wrapped_block_params_visible():
    """The wrapper must not hide the wrapped block's parameters from MLX."""
    model = RealModel(3, N_EMBD)
    before = len(tree_flatten(model.parameters()))

    patched = apply_steering_patch(model, {1: mx.ones(N_EMBD), 2: mx.ones(N_EMBD)})
    assert patched == 2

    after = tree_flatten(model.parameters())
    # Steering directions are not parameters; the wrapped Linear weights are
    # still all present — just routed through the wrapper's 'block' child.
    assert len(after) == before
    assert any("layers.1.block" in k for k, _ in after)
    # A full parameter walk must still succeed.
    mx.eval(model.parameters())


def test_patch_survives_set_dtype():
    """set_dtype walks the whole module tree — it must see through the wrapper."""
    model = RealModel(2, N_EMBD)
    apply_steering_patch(model, {1: mx.ones(N_EMBD)})
    model.set_dtype(mx.float16)
    out = model.layers[1](mx.zeros((1, 1, N_EMBD), dtype=mx.float16))
    assert out.dtype == mx.float16


def test_apply_patch_out_of_range_layer_skipped():
    model = FakeModel(N_LAYERS, N_EMBD)
    patched = apply_steering_patch(model, {1: mx.ones(N_EMBD), 99: mx.ones(N_EMBD)})
    assert patched == 1


def test_apply_patch_rejects_wrong_n_embd():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="n_embd"):
        apply_steering_patch(model, {1: mx.ones(N_EMBD + 3)})


def test_empty_layer_map_is_noop():
    model = FakeModel(N_LAYERS, N_EMBD)
    assert apply_steering_patch(model, {}) == 0
    assert getattr(model, "_omlx_steering_active", False) is False


# ---------------------------------------------------------------------------
# Generator — contrastive prompt pairs
# ---------------------------------------------------------------------------


def test_generate_mean_matches_hidden_state_diff():
    """With identity-plus-bias blocks, per-layer biases cancel in the diff,
    so the mean direction is just the normalised mean embedding difference."""
    model = FakeModel(N_LAYERS, N_EMBD)
    tok = FakeTokenizer()
    positive = ["AAA", "BBB", "CCC"]
    negative = ["aaa", "bbb", "ccc"]

    sv = generate_steering_vector(
        model, tok, positive, negative, method="mean", model_name="fake"
    )
    assert sv.layers == list(range(N_LAYERS))
    assert sv.n_embd == N_EMBD
    assert sv.method == "mean"

    # Expected: normalise(mean over pairs of emb[pos_last] - emb[neg_last]).
    diffs = [
        model._emb[ord(p[-1]) % 256] - model._emb[ord(n[-1]) % 256]
        for p, n in zip(positive, negative)
    ]
    expected = mx.stack(diffs).mean(axis=0)
    expected = expected / mx.linalg.norm(expected)
    for il in range(N_LAYERS):
        assert mx.allclose(sv.directions[il], expected, atol=1e-5)
        assert abs(float(mx.linalg.norm(sv.directions[il])) - 1.0) < 1e-5


def test_generate_pca_direction_is_oriented_and_unit():
    model = FakeModel(N_LAYERS, N_EMBD)
    tok = FakeTokenizer()
    positive = ["high", "more", "most", "peak"]
    negative = ["low", "less", "dips", "base"]

    sv = generate_steering_vector(model, tok, positive, negative, method="pca")
    for il in range(N_LAYERS):
        d = sv.directions[il]
        assert abs(float(mx.linalg.norm(d)) - 1.0) < 1e-5
        # The diffs should project positively onto the oriented direction.
        diffs = mx.stack(
            [
                model._emb[ord(p[-1]) % 256] - model._emb[ord(n[-1]) % 256]
                for p, n in zip(positive, negative)
            ]
        )
        assert float((diffs @ d).mean()) > 0.0


def test_generate_restores_model_after_run():
    model = FakeModel(N_LAYERS, N_EMBD)
    originals = list(model.model.layers)
    generate_steering_vector(
        model, FakeTokenizer(), ["a", "b"], ["c", "d"], method="mean"
    )
    assert model.model.layers == originals


def test_generate_layer_subset():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model, FakeTokenizer(), ["a", "b"], ["c", "d"], method="mean", layers=[1, 3]
    )
    assert sv.layers == [1, 3]


def test_generate_rejects_unequal_prompt_counts():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="differ"):
        generate_steering_vector(model, FakeTokenizer(), ["a", "b"], ["c"])


def test_generate_rejects_empty_prompts():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="at least one"):
        generate_steering_vector(model, FakeTokenizer(), [], [])


def test_generate_pca_needs_two_pairs():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="at least 2"):
        generate_steering_vector(
            model, FakeTokenizer(), ["a"], ["b"], method="pca"
        )


def test_generate_rejects_unknown_method():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="unknown method"):
        generate_steering_vector(
            model, FakeTokenizer(), ["a", "b"], ["c", "d"], method="bogus"
        )


def test_generated_vector_roundtrips_through_disk(tmp_path):
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model, FakeTokenizer(), ["aa", "bb", "cc"], ["xx", "yy", "zz"], method="mean"
    )
    path = tmp_path / "generated.safetensors"
    sv.save(path)
    loaded = SteeringVector.load(path)
    for il in sv.layers:
        assert mx.allclose(loaded.directions[il], sv.directions[il], atol=1e-5)
