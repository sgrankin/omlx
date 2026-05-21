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
from omlx.steering import (
    STEERING_FORMAT_VERSION,
    SteeringSpec,
    SteeringVector,
)
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


class IdentityBlock:
    """A block that returns its input unchanged."""

    def __call__(self, h, *args, **kwargs):
        return h


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text] or [0]


def _spec(
    directions,
    *,
    n_embd: int = N_EMBD,
    strength: float = 1.0,
    mode: str = "add",
    layer_start=None,
    layer_end=None,
    scaling: str = "unit",
) -> SteeringSpec:
    """Build a SteeringSpec from a ``{layer: array}`` dict."""
    return SteeringSpec(
        vector=SteeringVector(dict(directions), n_embd=n_embd, scaling=scaling),
        strength=strength,
        mode=mode,
        layer_start=layer_start,
        layer_end=layer_end,
    )


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


class NormedBlock(nn.Module):
    """Block with an RMSNorm — used to exercise the dtype probe."""

    def __init__(self, n_embd: int, norm_dtype=mx.float16):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(n_embd)
        self.input_layernorm.weight = self.input_layernorm.weight.astype(norm_dtype)
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, h, *args, **kwargs):
        return self.proj(h)


class NormedModel(nn.Module):
    """An nn.Module model whose blocks carry a (typed) layernorm."""

    def __init__(self, n_layers: int, n_embd: int, norm_dtype=mx.float16):
        super().__init__()
        self.layers = [NormedBlock(n_embd, norm_dtype) for _ in range(n_layers)]
        self.args = SimpleNamespace(hidden_size=n_embd)


class FakeVLM:
    """A VLM-shaped model: the text decoder lives under .language_model."""

    def __init__(self, n_layers: int, n_embd: int, vocab: int = 256):
        self.language_model = FakeModel(n_layers, n_embd, vocab)
        self.config = SimpleNamespace(hidden_size=n_embd)


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


def test_save_load_preserves_scaling(tmp_path):
    path = tmp_path / "vec.safetensors"
    SteeringVector(
        {1: mx.ones(N_EMBD)}, n_embd=N_EMBD, scaling="magnitude"
    ).save(path)
    assert SteeringVector.load(path).scaling == "magnitude"


def test_scaling_defaults_to_unit():
    assert SteeringVector({1: mx.ones(N_EMBD)}).scaling == "unit"


def test_get_steering_dir():
    from omlx.steering import get_steering_dir

    d = get_steering_dir()
    assert d.name == "steering"
    assert d.parent.name == ".omlx"


def test_list_datasets_includes_bundled():
    from omlx.steering import list_datasets

    ds = list_datasets()
    for name in ("joy", "calm", "assistant", "evil"):
        assert name in ds, f"bundled dataset {name!r} missing"


def test_resolve_dataset_by_name():
    from omlx.steering import resolve_dataset

    path = resolve_dataset("joy")
    assert path.name == "joy.json"
    assert path.exists()


def test_resolve_dataset_accepts_explicit_path(tmp_path):
    from omlx.steering import resolve_dataset

    f = tmp_path / "custom.json"
    f.write_text("{}")
    assert resolve_dataset(str(f)) == f


def test_resolve_dataset_unknown_raises():
    from omlx.steering import resolve_dataset

    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_dataset("no-such-steering-dataset")


def test_bundled_datasets_are_well_formed():
    import json

    from omlx.steering import get_bundled_datasets_dir

    files = list(get_bundled_datasets_dir().glob("*.json"))
    assert len(files) >= 9
    for f in files:
        data = json.loads(f.read_text())
        assert set(data) >= {"positive", "negative"}, f
        pos, neg = data["positive"], data["negative"]
        assert len(pos) == len(neg) > 0, f
        assert all(isinstance(p, str) and p.strip() for p in pos), f
        assert all(isinstance(n, str) and n.strip() for n in neg), f


def test_suggest_band_around_peak():
    from omlx.steering_generator import _suggest_band

    # peak at index 3; contiguous run >= 0.6*peak is indices 2..4.
    seps = [0.1, 0.3, 0.7, 1.0, 0.65, 0.2, 0.0]
    assert _suggest_band(seps, threshold=0.6) == (2, 4)


def test_suggest_band_excludes_last_layer():
    from omlx.steering_generator import _suggest_band

    # the genuine peak is the last layer; it must be ignored.
    assert _suggest_band([0.2, 0.9, 0.3, 5.0], threshold=0.6) == (1, 1)


def test_suggest_band_none_when_flat():
    from omlx.steering_generator import _suggest_band

    assert _suggest_band([0.0, 0.0, 0.0], threshold=0.6) is None


def test_analyze_layers_returns_per_layer_stats():
    from omlx.steering_generator import analyze_layers

    model = FakeModel(N_LAYERS, N_EMBD)
    result = analyze_layers(
        model, FakeTokenizer(), ["aa", "bb", "cc"], ["xx", "yy", "zz"]
    )
    assert len(result["layers"]) == N_LAYERS
    for r in result["layers"]:
        assert {
            "layer", "separation", "ortho_separation", "consistency"
        } <= set(r)
        assert r["separation"] >= 0.0
        assert r["ortho_separation"] >= 0.0
    band = result["suggested"]
    assert band is None or (0 <= band[0] <= band[1] < N_LAYERS - 1)


def test_analyze_layers_rejects_mismatched_counts():
    from omlx.steering_generator import analyze_layers

    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="differ"):
        analyze_layers(model, FakeTokenizer(), ["a", "b"], ["c"])


def test_analyze_layers_returns_summary_and_warnings():
    """analyze_layers now reports a verdict + optional warnings."""
    from omlx.steering_generator import analyze_layers

    model = FakeModel(N_LAYERS, N_EMBD)
    result = analyze_layers(
        model, FakeTokenizer(), ["aa", "bb", "cc"], ["xx", "yy", "zz"]
    )
    assert "summary" in result and isinstance(result["summary"], str)
    assert "separation" in result["summary"]
    assert "warnings" in result and isinstance(result["warnings"], list)


def test_quality_summary_flags_weak_axis():
    """A flat / low-separation profile triggers warnings."""
    from omlx.steering_generator import _quality_summary

    weak = [
        {"layer": i, "separation": 0.05, "consistency": 0.05}
        for i in range(4)
    ]
    summary, warnings = _quality_summary(weak)
    assert "weak" in summary or "scattered" in summary
    assert any("weak" in w for w in warnings)
    assert any("scattered" in w for w in warnings)


def test_quality_summary_passes_clean_axis():
    """A strong + coherent profile produces no warnings."""
    from omlx.steering_generator import _quality_summary

    clean = [
        {"layer": 0, "separation": 0.2, "consistency": 0.3},
        {"layer": 1, "separation": 1.2, "consistency": 0.9},
        {"layer": 2, "separation": 0.5, "consistency": 0.5},
    ]
    summary, warnings = _quality_summary(clean)
    assert "strong" in summary
    assert "coherent" in summary
    assert warnings == []


def test_quality_summary_flags_orthogonalize_confound():
    """A peak layer where ortho_sep ≪ sep should trigger a confound warning."""
    from omlx.steering_generator import _quality_summary

    confounded = [
        {"layer": 0, "separation": 0.1, "ortho_separation": 0.08,
         "consistency": 0.4},
        # Peak — Cohen's d 1.2 raw but only 0.3 after orthogonalize.
        {"layer": 1, "separation": 1.2, "ortho_separation": 0.30,
         "consistency": 0.9},
        {"layer": 2, "separation": 0.5, "ortho_separation": 0.45,
         "consistency": 0.5},
    ]
    summary, warnings = _quality_summary(confounded)
    assert any("activation-magnitude" in w for w in warnings)
    # The strong/coherent labels still hold — the warning is additive.
    assert "strong" in summary


def test_quality_summary_no_ortho_warning_when_clean():
    """ortho_sep close to sep at the peak: no confound warning."""
    from omlx.steering_generator import _quality_summary

    clean = [
        {"layer": 0, "separation": 1.2, "ortho_separation": 1.15,
         "consistency": 0.9},
        {"layer": 1, "separation": 0.5, "ortho_separation": 0.48,
         "consistency": 0.5},
    ]
    _, warnings = _quality_summary(clean)
    assert not any("activation-magnitude" in w for w in warnings)


def test_per_layer_metrics_detects_magnitude_confound():
    """When the contrast direction parallels the control mean,
    ortho_separation drops well below the raw separation."""
    import mlx.core as mx
    import numpy as np

    from omlx.steering_generator import _per_layer_metrics

    # Both classes spread isotropically around a baseline along axis 0;
    # the positive class is shifted further along that same axis. The
    # mean-diff direction is then parallel to the control mean, so
    # orthogonalize strips most of the discriminating axis.
    rng = np.random.default_rng(0)
    d, n = 16, 32
    base = np.zeros(d, dtype=np.float32)
    base[0] = 5.0
    neg_np = base + rng.normal(scale=0.5, size=(n, d)).astype(np.float32)
    pos_np = base.copy()
    pos_np[0] = 9.0
    pos_np = pos_np + rng.normal(scale=0.5, size=(n, d)).astype(np.float32)

    sep, ortho_sep, _ = _per_layer_metrics(mx.array(pos_np), mx.array(neg_np))
    assert sep > 1.0           # large raw separation
    assert ortho_sep < 0.3 * sep  # mostly stripped


def test_per_layer_metrics_keeps_orthogonal_contrast():
    """When the mean-diff direction is orthogonal to the control mean,
    ortho_separation ≈ separation (orthogonalize is a no-op)."""
    import mlx.core as mx
    import numpy as np

    from omlx.steering_generator import _per_layer_metrics

    # Both classes share a non-zero offset along axis 0 (the "control
    # mean" direction); the trait shift sits purely along axis 1. So
    # mean_diff lives on axis 1, which is orthogonal to the control mean.
    rng = np.random.default_rng(1)
    d, n = 16, 32
    neg_centre = np.zeros(d, dtype=np.float32)
    neg_centre[0] = 5.0
    pos_centre = neg_centre.copy()
    pos_centre[1] = 5.0
    neg_np = neg_centre + rng.normal(scale=0.3, size=(n, d)).astype(np.float32)
    pos_np = pos_centre + rng.normal(scale=0.3, size=(n, d)).astype(np.float32)

    sep, ortho_sep, _ = _per_layer_metrics(mx.array(pos_np), mx.array(neg_np))
    assert sep > 1.0
    assert ortho_sep > 0.9 * sep  # axis preserved


def test_quality_summary_skips_ortho_warning_when_peak_weak():
    """If everything is weak, the ortho warning is redundant — suppressed."""
    from omlx.steering_generator import _quality_summary

    weak = [
        {"layer": 0, "separation": 0.1, "ortho_separation": 0.02,
         "consistency": 0.4},
        {"layer": 1, "separation": 0.2, "ortho_separation": 0.01,
         "consistency": 0.5},
    ]
    _, warnings = _quality_summary(weak)
    # Weak-separation warning fires; ortho warning does not — the peak
    # itself is below the usable floor.
    assert any("weak" in w for w in warnings)
    assert not any("activation-magnitude" in w for w in warnings)


# ---------------------------------------------------------------------------
# Load-time apply: apply_post_load_transforms → _maybe_apply_steering
# ---------------------------------------------------------------------------


def test_post_load_applies_steering_from_settings(tmp_path):
    """settings.steering_vectors → apply_post_load_transforms wires the patch."""
    from types import SimpleNamespace

    from omlx.utils.model_loading import apply_post_load_transforms

    model = FakeModel(N_LAYERS, N_EMBD)
    vec_path = tmp_path / "v.safetensors"
    SteeringVector(directions={1: mx.ones(N_EMBD)}, n_embd=N_EMBD).save(str(vec_path))
    settings = SimpleNamespace(
        steering_vectors=[
            {"path": str(vec_path), "strength": 0.5, "mode": "add"}
        ],
        index_cache_freq=None,
    )
    apply_post_load_transforms(model, settings)
    assert getattr(model, "_omlx_steering_active", False) is True
    assert isinstance(model.model.layers[1], _SteeredLayer)


def test_post_load_no_op_when_steering_empty():
    """No steering_vectors → no patch."""
    from types import SimpleNamespace

    from omlx.utils.model_loading import apply_post_load_transforms

    model = FakeModel(N_LAYERS, N_EMBD)
    apply_post_load_transforms(
        model, SimpleNamespace(steering_vectors=None, index_cache_freq=None)
    )
    assert getattr(model, "_omlx_steering_active", False) is False


def test_post_load_no_op_when_settings_is_none():
    """settings=None — early return, no exception, model untouched."""
    from omlx.utils.model_loading import apply_post_load_transforms

    model = FakeModel(N_LAYERS, N_EMBD)
    apply_post_load_transforms(model, None)
    assert getattr(model, "_omlx_steering_active", False) is False


def test_post_load_logs_per_spec_details(tmp_path, caplog):
    """The load-time log lists each spec's name, mode, strength, band."""
    import logging
    from types import SimpleNamespace

    from omlx.utils.model_loading import apply_post_load_transforms

    model = FakeModel(N_LAYERS, N_EMBD)
    vec_path = tmp_path / "joy.safetensors"
    SteeringVector(directions={1: mx.ones(N_EMBD)}, n_embd=N_EMBD).save(str(vec_path))
    settings = SimpleNamespace(
        steering_vectors=[
            {
                "path": str(vec_path),
                "strength": 0.5,
                "mode": "add",
                "layer_start": None,
                "layer_end": None,
            }
        ],
        index_cache_freq=None,
    )
    with caplog.at_level(logging.INFO, logger="omlx.utils.model_loading"):
        apply_post_load_transforms(model, settings)
    msg = "\n".join(r.message for r in caplog.records)
    assert "joy.safetensors" in msg
    assert "mode=add" in msg
    assert "strength=0.5" in msg
    assert "layers=all" in msg


def test_post_load_isolates_one_bad_vector(tmp_path, caplog):
    """One bad vector entry warns and is skipped; the rest still apply."""
    import logging
    from types import SimpleNamespace

    from omlx.utils.model_loading import apply_post_load_transforms

    model = FakeModel(N_LAYERS, N_EMBD)
    good = tmp_path / "good.safetensors"
    SteeringVector(directions={2: mx.ones(N_EMBD)}, n_embd=N_EMBD).save(str(good))
    settings = SimpleNamespace(
        steering_vectors=[
            {"path": str(tmp_path / "missing.safetensors"), "strength": 1.0, "mode": "add"},
            {"path": str(good), "strength": 1.0, "mode": "add"},
        ],
        index_cache_freq=None,
    )
    with caplog.at_level(logging.WARNING, logger="omlx.utils.model_loading"):
        apply_post_load_transforms(model, settings)
    # The good one was still applied.
    assert getattr(model, "_omlx_steering_active", False) is True
    assert isinstance(model.model.layers[2], _SteeredLayer)
    # The bad one warned and named the missing file.
    msg = "\n".join(r.message for r in caplog.records)
    assert "missing.safetensors" in msg


def test_apply_patch_clear_error_on_vlm_dim_mismatch():
    """A VLM with wrong-dim steering vector errors clearly via language_model."""
    vlm = FakeVLM(N_LAYERS, N_EMBD)
    wrong = _spec({1: mx.ones(N_EMBD + 4)}, n_embd=N_EMBD + 4)
    with pytest.raises(ValueError, match="n_embd"):
        apply_steering_patch(vlm, [wrong])


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


def test_find_layers_container_resolves_vlm():
    """For a VLM the deepest list (language_model.model.layers) must win."""
    vlm = FakeVLM(N_LAYERS, N_EMBD)
    container = find_layers_container(vlm)
    assert container is vlm.language_model.model
    assert len(container.layers) == N_LAYERS


def test_apply_patch_adds_direction():
    model = FakeModel(N_LAYERS, N_EMBD)
    direction = mx.arange(N_EMBD, dtype=mx.float32) + 1.0
    patched = apply_steering_patch(model, [_spec({2: direction})])
    assert patched == 1
    assert model._omlx_steering_active is True

    layer = model.model.layers[2]
    assert isinstance(layer, _SteeredLayer)
    h = mx.zeros((1, 1, N_EMBD))
    out = layer(h)
    # FakeBlock for layer index 2 has bias seed == 3.
    assert mx.allclose(out, h + 3.0 + direction)


def test_apply_patch_scales_by_strength():
    model = FakeModel(N_LAYERS, N_EMBD)
    direction = mx.ones(N_EMBD)
    apply_steering_patch(model, [_spec({0: direction}, strength=2.5)])
    out = model.model.layers[0](mx.zeros((1, 1, N_EMBD)))
    # layer 0 FakeBlock seed == 1, plus 2.5 * direction.
    assert mx.allclose(out, mx.full((1, 1, N_EMBD), 1.0 + 2.5))


def test_model_hidden_size_resolves_via_language_model():
    """For a VLM, n_embd is found by falling through to language_model."""
    from omlx.patches.steering import model_hidden_size

    vlm = FakeVLM(N_LAYERS, N_EMBD)
    # Top-level config has neither .args nor .config; fallback should work.
    assert model_hidden_size(vlm) == N_EMBD


def test_model_hidden_size_prefers_text_decoder_over_stray_top_level():
    """VLM-style: top-level config has a misleading default hidden_size.

    mlx-vlm's Gemma 4 ModelConfig declares ``hidden_size: int = 1536`` at the
    top level (a stray field unrelated to the text stream); the real text
    decoder size is ``text_config.hidden_size``. ``model_hidden_size`` must
    return the text decoder's size since that's what steering wraps.
    """
    from omlx.patches.steering import model_hidden_size

    vlm = FakeVLM(N_LAYERS, N_EMBD)
    vlm.config = SimpleNamespace(
        hidden_size=1536,
        text_config=SimpleNamespace(hidden_size=N_EMBD),
    )
    assert model_hidden_size(vlm) == N_EMBD


def test_apply_patch_normalizes_add_by_layer_count():
    """Additive strength is divided across the steered layers."""
    model = FakeModel(N_LAYERS, N_EMBD)
    direction = mx.ones(N_EMBD)
    # Three add layers at strength 3.0 -> each layer's bias is 3.0/3 = 1.0.
    apply_steering_patch(
        model, [_spec({0: direction, 1: direction, 2: direction}, strength=3.0)]
    )
    out = model.model.layers[1](mx.zeros((1, 1, N_EMBD)))
    # FakeBlock layer 1 has bias seed == 2; plus the per-layer add of 1.0.
    assert mx.allclose(out, mx.full((1, 1, N_EMBD), 2.0 + 1.0))


def test_apply_patch_add_strength_band_independent():
    """A wider band splits the same strength into a smaller per-layer bias."""
    d = mx.ones(N_EMBD)
    narrow = FakeModel(N_LAYERS, N_EMBD)
    apply_steering_patch(narrow, [_spec({0: d}, strength=4.0)])
    wide = FakeModel(N_LAYERS, N_EMBD)
    apply_steering_patch(wide, [_spec({0: d, 1: d, 2: d, 3: d}, strength=4.0)])
    # 1-layer spec -> per-layer 4.0; 4-layer spec -> per-layer 1.0.
    assert mx.allclose(narrow.model.layers[0]._steer_add, d * 4.0)
    assert mx.allclose(wide.model.layers[0]._steer_add, d * 1.0)


def test_patch_handles_tuple_output():
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[1] = TupleBlock()
    apply_steering_patch(model, [_spec({1: mx.ones(N_EMBD)})])

    h = mx.zeros((1, 1, N_EMBD))
    out = model.model.layers[1](h)
    assert isinstance(out, tuple)
    assert mx.allclose(out[0], h + 1.0)
    assert out[1] == "aux"


def test_patch_preserves_hidden_dtype():
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[0] = TupleBlock()  # identity-ish, returns input dtype
    apply_steering_patch(model, [_spec({0: mx.ones(N_EMBD, dtype=mx.float32)})])
    out, _aux = model.model.layers[0](mx.zeros((1, 1, N_EMBD), dtype=mx.bfloat16))
    assert out.dtype == mx.bfloat16


def test_patch_precasts_add_bias_to_norm_dtype():
    """Probe must pick the layernorm dtype and pre-cast the additive bias."""
    model = NormedModel(2, N_EMBD, norm_dtype=mx.bfloat16)
    apply_steering_patch(model, [_spec({0: mx.ones(N_EMBD, dtype=mx.float32)})])
    layer = model.layers[0]
    assert isinstance(layer, _SteeredLayer)
    assert layer._steer_add.dtype == mx.bfloat16


def test_patch_precasts_projection_unit_to_norm_dtype():
    model = NormedModel(2, N_EMBD, norm_dtype=mx.float16)
    apply_steering_patch(
        model,
        [_spec(
            {0: mx.arange(N_EMBD, dtype=mx.float32) + 1.0},
            mode="project", strength=1.0,
        )],
    )
    layer = model.layers[0]
    assert isinstance(layer, _SteeredLayer)
    assert len(layer._steer_proj) == 1
    unit, _strength = layer._steer_proj[0]
    assert unit.dtype == mx.float16


def test_patch_falls_back_to_fp32_when_no_norm():
    """A synthetic block without any layernorm hits the fp32 fallback.

    Real models always carry layernorms, so this path only fires on
    test scaffolds or unusual architectures — fp32 is correct but pays
    a per-call downcast inside ``_steer``.
    """
    model = FakeModel(N_LAYERS, N_EMBD)  # FakeBlock has no nn.Module structure
    apply_steering_patch(model, [_spec({0: mx.ones(N_EMBD, dtype=mx.float32)})])
    assert model.model.layers[0]._steer_add.dtype == mx.float32


def test_remove_patch_restores_blocks():
    model = FakeModel(N_LAYERS, N_EMBD)
    originals = list(model.model.layers)
    apply_steering_patch(
        model, [_spec({1: mx.ones(N_EMBD), 3: mx.ones(N_EMBD)})]
    )
    assert remove_steering_patch(model) is True
    assert model.model.layers == originals
    assert model._omlx_steering_active is False
    # Removing again is a no-op.
    assert remove_steering_patch(model) is False


def test_apply_patch_is_idempotent():
    """Re-applying replaces the prior patch rather than nesting wrappers."""
    model = FakeModel(N_LAYERS, N_EMBD)
    apply_steering_patch(model, [_spec({1: mx.ones(N_EMBD)})])
    apply_steering_patch(model, [_spec({1: mx.ones(N_EMBD)}, strength=5.0)])
    layer = model.model.layers[1]
    assert isinstance(layer, _SteeredLayer)
    assert not isinstance(layer["block"], _SteeredLayer)


def test_patch_keeps_wrapped_block_params_visible():
    """The wrapper must not hide the wrapped block's parameters from MLX."""
    model = RealModel(3, N_EMBD)
    before = len(tree_flatten(model.parameters()))

    patched = apply_steering_patch(
        model, [_spec({1: mx.ones(N_EMBD), 2: mx.ones(N_EMBD)})]
    )
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
    apply_steering_patch(model, [_spec({1: mx.ones(N_EMBD)})])
    model.set_dtype(mx.float16)
    out = model.layers[1](mx.zeros((1, 1, N_EMBD), dtype=mx.float16))
    assert out.dtype == mx.float16


def test_apply_patch_out_of_range_layer_skipped():
    model = FakeModel(N_LAYERS, N_EMBD)
    patched = apply_steering_patch(
        model, [_spec({1: mx.ones(N_EMBD), 99: mx.ones(N_EMBD)})]
    )
    assert patched == 1


def test_apply_patch_rejects_wrong_n_embd():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="n_embd"):
        apply_steering_patch(
            model, [_spec({1: mx.ones(N_EMBD + 3)}, n_embd=N_EMBD + 3)]
        )


def test_empty_specs_is_noop():
    model = FakeModel(N_LAYERS, N_EMBD)
    assert apply_steering_patch(model, []) == 0
    assert getattr(model, "_omlx_steering_active", False) is False


def test_apply_patch_layer_range():
    model = FakeModel(N_LAYERS, N_EMBD)
    spec = _spec(
        {il: mx.ones(N_EMBD) for il in range(N_LAYERS)},
        layer_start=1,
        layer_end=2,
    )
    patched = apply_steering_patch(model, [spec])
    assert patched == 2
    assert isinstance(model.model.layers[1], _SteeredLayer)
    assert isinstance(model.model.layers[2], _SteeredLayer)
    assert not isinstance(model.model.layers[0], _SteeredLayer)
    assert not isinstance(model.model.layers[3], _SteeredLayer)


# ---------------------------------------------------------------------------
# Steering patch — projection mode
# ---------------------------------------------------------------------------


def test_projection_removes_component():
    """mode='project', strength=1.0 ablates the direction from the output."""
    d = mx.arange(N_EMBD, dtype=mx.float32) + 1.0
    layer = _SteeredLayer(IdentityBlock(), None, [(d / mx.linalg.norm(d), 1.0)])
    h = mx.random.normal((1, 3, N_EMBD))
    out = layer(h)
    u = d / mx.linalg.norm(d)
    residual = (out * u).sum(axis=-1)
    assert float(mx.abs(residual).max()) < 1e-4


def test_projection_partial_strength_halves_component():
    d = mx.ones(N_EMBD)
    u = d / mx.linalg.norm(d)
    layer = _SteeredLayer(IdentityBlock(), None, [(u, 0.5)])
    h = mx.random.normal((1, 4, N_EMBD))
    before = (h * u).sum(axis=-1)
    after = (layer(h) * u).sum(axis=-1)
    assert mx.allclose(after, before * 0.5, atol=1e-4)


def test_projection_negative_strength_amplifies():
    d = mx.ones(N_EMBD)
    u = d / mx.linalg.norm(d)
    layer = _SteeredLayer(IdentityBlock(), None, [(u, -1.0)])
    h = mx.random.normal((1, 4, N_EMBD))
    before = (h * u).sum(axis=-1)
    after = (layer(h) * u).sum(axis=-1)
    # h - (-1)·(u·h)·u  ->  component doubles.
    assert mx.allclose(after, before * 2.0, atol=1e-4)


def test_apply_patch_project_mode_normalizes_direction():
    """A non-unit projection direction is unit-normalised by the patch."""
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[0] = IdentityBlock()
    raw = mx.arange(N_EMBD, dtype=mx.float32) + 3.0  # deliberately not unit
    apply_steering_patch(model, [_spec({0: raw}, mode="project", strength=1.0)])
    h = mx.random.normal((1, 2, N_EMBD))
    out = model.model.layers[0](h)
    u = raw / mx.linalg.norm(raw)
    assert float(mx.abs((out * u).sum(axis=-1)).max()) < 1e-4


# ---------------------------------------------------------------------------
# Steering patch — multiple vectors
# ---------------------------------------------------------------------------


def test_multi_vector_additive_specs_sum():
    """Two additive specs on the same layer sum into one bias."""
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[0] = IdentityBlock()
    s1 = _spec({0: mx.ones(N_EMBD)}, strength=1.0)
    s2 = _spec({0: mx.ones(N_EMBD)}, strength=3.0)
    apply_steering_patch(model, [s1, s2])
    out = model.model.layers[0](mx.zeros((1, 1, N_EMBD)))
    assert mx.allclose(out, mx.full((1, 1, N_EMBD), 4.0))


def test_multi_vector_add_then_project():
    """An additive spec and a projection spec coexist on one layer."""
    model = FakeModel(N_LAYERS, N_EMBD)
    model.model.layers[0] = IdentityBlock()
    u = mx.ones(N_EMBD) / mx.linalg.norm(mx.ones(N_EMBD))
    add = _spec({0: mx.ones(N_EMBD)}, strength=2.0)
    proj = _spec({0: u}, mode="project", strength=1.0)
    apply_steering_patch(model, [add, proj])
    out = model.model.layers[0](mx.zeros((1, 1, N_EMBD)))
    # bias added first, then its component along u projected back out.
    assert float(mx.abs((out * u).sum(axis=-1)).max()) < 1e-4


def test_multi_vector_disjoint_layers():
    model = FakeModel(N_LAYERS, N_EMBD)
    s1 = _spec({0: mx.ones(N_EMBD)})
    s2 = _spec({2: mx.ones(N_EMBD)})
    patched = apply_steering_patch(model, [s1, s2])
    assert patched == 2
    assert isinstance(model.model.layers[0], _SteeredLayer)
    assert isinstance(model.model.layers[2], _SteeredLayer)


# ---------------------------------------------------------------------------
# SteeringSpec
# ---------------------------------------------------------------------------


def test_spec_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown steering mode"):
        SteeringSpec(vector=SteeringVector({0: mx.ones(N_EMBD)}), mode="bogus")


def test_spec_active_directions_filters_range():
    vec = SteeringVector({il: mx.ones(N_EMBD) for il in range(N_LAYERS)})
    spec = SteeringSpec(vector=vec, layer_start=1, layer_end=2)
    assert sorted(spec.active_directions()) == [1, 2]


def test_spec_active_directions_unbounded():
    vec = SteeringVector({il: mx.ones(N_EMBD) for il in range(N_LAYERS)})
    assert sorted(SteeringSpec(vector=vec).active_directions()) == list(
        range(N_LAYERS)
    )


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
        model, tok, positive, negative,
        method="mean", scaling="unit", orthogonalize=False, model_name="fake",
    )
    # Default skips the final layer.
    assert sv.layers == list(range(N_LAYERS - 1))
    assert sv.n_embd == N_EMBD
    assert sv.method == "mean"

    # Expected: normalise(mean over pairs of emb[pos_last] - emb[neg_last]).
    diffs = [
        model._emb[ord(p[-1]) % 256] - model._emb[ord(n[-1]) % 256]
        for p, n in zip(positive, negative)
    ]
    expected = mx.stack(diffs).mean(axis=0)
    expected = expected / mx.linalg.norm(expected)
    for il in sv.layers:
        assert mx.allclose(sv.directions[il], expected, atol=1e-5)
        assert abs(float(mx.linalg.norm(sv.directions[il])) - 1.0) < 1e-5


def test_generate_pca_direction_is_oriented_and_unit():
    model = FakeModel(N_LAYERS, N_EMBD)
    tok = FakeTokenizer()
    positive = ["high", "more", "most", "peak"]
    negative = ["low", "less", "dips", "base"]

    sv = generate_steering_vector(
        model, tok, positive, negative,
        method="pca", scaling="unit", orthogonalize=False,
    )
    for il in sv.layers:
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


def test_generate_crosscov_direction_oriented_and_unit():
    model = FakeModel(N_LAYERS, N_EMBD)
    tok = FakeTokenizer()
    positive = ["high", "more", "most", "peak", "tall", "wide"]
    negative = ["low", "less", "dips", "base", "tiny", "thin"]

    sv = generate_steering_vector(
        model, tok, positive, negative,
        method="crosscov", scaling="unit", orthogonalize=False,
    )
    assert sv.method == "crosscov"
    assert sv.layers == list(range(N_LAYERS - 1))
    for il in sv.layers:
        d = sv.directions[il]
        assert abs(float(mx.linalg.norm(d)) - 1.0) < 1e-5
        diffs = mx.stack(
            [
                model._emb[ord(p[-1]) % 256] - model._emb[ord(n[-1]) % 256]
                for p, n in zip(positive, negative)
            ]
        )
        # Oriented so the contrastive differences project positively.
        assert float((diffs @ d).mean()) > 0.0


def test_generate_crosscov_needs_two_pairs():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="at least 2"):
        generate_steering_vector(
            model, FakeTokenizer(), ["a"], ["b"], method="crosscov"
        )


def test_generate_crosscov_magnitude_scaling():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model,
        FakeTokenizer(),
        ["aa", "bb", "cc", "dd"],
        ["ww", "xx", "yy", "zz"],
        method="crosscov",
        scaling="magnitude",
    )
    assert sv.method == "crosscov"
    assert sv.scaling == "magnitude"


def test_orthogonalize_removes_parallel_component():
    from omlx.steering_generator import _orthogonalize

    base = mx.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    direction = mx.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = _orthogonalize(direction, base)
    # Orthogonal to the control mean, and unit length.
    assert abs(float(mx.sum(result * base))) < 1e-5
    assert abs(float(mx.linalg.norm(result)) - 1.0) < 1e-5


def test_orthogonalize_degenerate_base_is_noop():
    from omlx.steering_generator import _orthogonalize

    direction = mx.arange(N_EMBD, dtype=mx.float32) + 1.0
    result = _orthogonalize(direction, mx.zeros(N_EMBD))
    assert mx.allclose(result, direction)


def test_generate_with_orthogonalize():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model,
        FakeTokenizer(),
        ["aa", "bb", "cc"],
        ["xx", "yy", "zz"],
        method="mean",
        scaling="unit",
        orthogonalize=True,
    )
    assert sv.layers == list(range(N_LAYERS - 1))
    for d in sv.directions.values():
        assert abs(float(mx.linalg.norm(d)) - 1.0) < 1e-4


def test_generate_skips_last_layer_by_default():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model, FakeTokenizer(), ["aa", "bb"], ["xx", "yy"], method="mean"
    )
    assert sv.layers == list(range(N_LAYERS - 1))
    assert (N_LAYERS - 1) not in sv.directions


def test_generate_explicit_layers_can_include_last():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model, FakeTokenizer(), ["aa", "bb"], ["xx", "yy"],
        method="mean", layers=[0, N_LAYERS - 1],
    )
    assert sv.layers == [0, N_LAYERS - 1]


def test_generate_default_scaling_is_magnitude():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model, FakeTokenizer(), ["aa", "bb"], ["xx", "yy"], method="mean"
    )
    assert sv.scaling == "magnitude"


def test_generate_default_method_is_mean():
    model = FakeModel(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        model, FakeTokenizer(), ["aa", "bb"], ["xx", "yy"]
    )
    assert sv.method == "mean"


def test_generate_orthogonalize_default_on():
    model = FakeModel(N_LAYERS, N_EMBD)
    pos, neg = ["aa", "bb", "cc"], ["xx", "yy", "zz"]
    default = generate_steering_vector(model, FakeTokenizer(), pos, neg)
    on = generate_steering_vector(
        model, FakeTokenizer(), pos, neg, orthogonalize=True
    )
    off = generate_steering_vector(
        model, FakeTokenizer(), pos, neg, orthogonalize=False
    )
    assert all(
        mx.allclose(default.directions[il], on.directions[il])
        for il in default.layers
    )
    assert any(
        not mx.allclose(default.directions[il], off.directions[il])
        for il in default.layers
    )


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


def test_generate_rejects_unknown_scaling():
    model = FakeModel(N_LAYERS, N_EMBD)
    with pytest.raises(ValueError, match="unknown scaling"):
        generate_steering_vector(
            model,
            FakeTokenizer(),
            ["a", "b"],
            ["c", "d"],
            method="mean",
            scaling="bogus",
        )


def test_generate_magnitude_scaling_equals_raw_mean_diff():
    """method=mean + scaling=magnitude reproduces the un-normalised mean diff.

    With unit u = meandiff/||meandiff||, the magnitude |mean(diff·u)| equals
    ||meandiff||, so unit·magnitude == the raw mean difference.
    """
    model = FakeModel(N_LAYERS, N_EMBD)
    tok = FakeTokenizer()
    positive = ["AAA", "BBB", "CCC"]
    negative = ["aaa", "bbb", "ccc"]

    sv = generate_steering_vector(
        model, tok, positive, negative,
        method="mean", scaling="magnitude", orthogonalize=False,
    )
    assert sv.scaling == "magnitude"

    diffs = [
        model._emb[ord(p[-1]) % 256] - model._emb[ord(n[-1]) % 256]
        for p, n in zip(positive, negative)
    ]
    expected = mx.stack(diffs).mean(axis=0)  # raw mean difference
    for il in sv.layers:
        assert mx.allclose(sv.directions[il], expected, atol=1e-2, rtol=1e-4)
    # And the directions are not unit-norm (that is the point of "magnitude").
    norms = [float(mx.linalg.norm(d)) for d in sv.directions.values()]
    assert not all(abs(n - 1.0) < 1e-3 for n in norms)


def test_generate_on_vlm_uses_language_model():
    """A VLM-shaped model is steered via its text decoder, not rejected."""
    vlm = FakeVLM(N_LAYERS, N_EMBD)
    sv = generate_steering_vector(
        vlm,
        FakeTokenizer(),
        ["aa", "bb", "cc"],
        ["xx", "yy", "zz"],
        method="mean",
        model_name="fake-vlm",
    )
    assert sv.layers == list(range(N_LAYERS - 1))
    assert sv.n_embd == N_EMBD
    # The text decoder's blocks must be restored after capture.
    assert all(
        isinstance(b, FakeBlock) for b in vlm.language_model.model.layers
    )


def test_drop_mtp_weights_on_load_filters_mtp_tensors():
    """The MTP-drop shim removes mtp.* tensors from a load_weights call."""
    import mlx.nn as nn

    from omlx.cli import _drop_mtp_weights_on_load

    received: list = []
    original = nn.Module.load_weights
    nn.Module.load_weights = lambda self, w, *a, **k: received.append(list(w))
    try:
        with _drop_mtp_weights_on_load():
            nn.Module.load_weights(
                object(),
                [
                    ("language_model.layers.0.self_attn.q_proj.weight", 1),
                    ("language_model.mtp.layers.0.self_attn.q_proj.weight", 2),
                    ("mtp.norm.weight", 3),
                ],
            )
    finally:
        nn.Module.load_weights = original

    assert received == [[("language_model.layers.0.self_attn.q_proj.weight", 1)]]
    # The shim restores the original load_weights on exit.
    assert nn.Module.load_weights is original


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
