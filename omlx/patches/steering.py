# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch: apply a steering vector to a loaded mlx-lm / mlx-vlm model.

Each transformer block is wrapped so that, after it produces its
residual-stream output ``h``, one or more steering specs are applied —
an additive bias and/or directional projections (see
:class:`~omlx.steering.SteeringSpec` and :class:`_SteeredLayer`).

Applied once at load time by ``apply_post_load_transforms`` and reversible
via :func:`remove_steering_patch`, so a cached model object can be re-used
without steering after the setting is cleared.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..steering import SteeringSpec

logger = logging.getLogger(__name__)


def find_layers_container(model: Any) -> Any | None:
    """Return the object whose ``.layers`` is the transformer block list.

    mlx-lm wraps the block list one level down (``model.model.layers``);
    mlx-vlm wraps it deeper (``model.language_model.model.layers``). A few
    omlx-custom models expose ``.layers`` at the top. Probe in
    most-specific-first order so the deepest real list wins.
    """
    candidates = (
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
        model,
    )
    for root in candidates:
        if root is None:
            continue
        layers = getattr(root, "layers", None)
        if isinstance(layers, (list, tuple)) and len(layers) > 0:
            return root
    return None


def model_hidden_size(model: Any) -> int | None:
    """Best-effort lookup of the model's hidden size (``n_embd``).

    On a VLM the top-level ``config`` only carries vision/text sub-configs;
    fall through to ``model.language_model`` so the steering vector's
    n_embd can still be validated against the text decoder.
    """
    for attr in ("args", "config"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            hs = obj.get("hidden_size") or obj.get("n_embd")
        else:
            hs = getattr(obj, "hidden_size", None) or getattr(obj, "n_embd", None)
        if hs:
            return int(hs)
    lm = getattr(model, "language_model", None)
    if lm is not None and lm is not model:
        return model_hidden_size(lm)
    return None


class _SteeredLayer(nn.Module):
    """Wraps a transformer block, steering its residual-stream output.

    Applies, in order, an optional additive bias and any number of
    directional projections::

        h <- h + add_bias
        h <- h - strength · (d̂ · h) · d̂      (for each projection)

    ``add_bias`` is the pre-summed contribution of every additive spec for
    this layer; projections are kept separate because they are not linear in
    the activation. See :func:`apply_steering_patch`.

    A real ``nn.Module`` (not a bare proxy): the wrapped block is registered
    as a child module, so its parameters stay visible to MLX's tree
    traversal — ``parameters()``, ``set_dtype``, weight save — even though
    the wrapper sits in the model's ``.layers`` list for the model's whole
    lifetime. Unlike the transient ``_AttentionCapture`` in
    ``patches/specprefill`` (removed in a ``finally``), this patch is
    persistent, so correct parameter tracking matters.

    Steering data is stored via ``object.__setattr__`` so MLX does not
    register it as a trainable parameter — it is fixed configuration, not a
    weight, and must not be quantized or serialised with the model.
    """

    def __init__(
        self,
        block: Any,
        add_bias: mx.array | None,
        projections: list[tuple[mx.array, float]],
    ):
        super().__init__()
        # Register the block as a dict-child directly: when it is a real
        # nn.Module, MLX's traversal recurses into it for parameters();
        # when it is a plain object it is just a leaf entry.
        self["block"] = block
        object.__setattr__(self, "_steer_add", add_bias)
        object.__setattr__(self, "_steer_proj", projections)

    def _steer(self, h: mx.array) -> mx.array:
        # Cast steering data to the hidden dtype so a bf16/fp16 residual
        # stream is not silently promoted to f32.
        if self._steer_add is not None:
            h = h + self._steer_add.astype(h.dtype)
        for unit, strength in self._steer_proj:
            u = unit.astype(h.dtype)
            # Per-token component of h along the (unit) direction.
            coeff = (h * u).sum(axis=-1, keepdims=True)
            h = h - strength * coeff * u
        return h

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self["block"](*args, **kwargs)
        # A block returns either the hidden state directly or a tuple whose
        # first element is the hidden state (some architectures also return
        # cache/aux entries).
        if isinstance(out, tuple):
            return (self._steer(out[0]),) + tuple(out[1:])
        return self._steer(out)

    def __getattr__(self, name: str) -> Any:
        # Reached for names absent from the wrapper; delegate to the block
        # so callers probing layer internals (e.g. ``layer.self_attn``)
        # still see through the wrapper.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self["block"], name)


def apply_steering_patch(
    model: Any,
    specs: list[SteeringSpec],
    *,
    n_embd: int | None = None,
) -> int:
    """Wrap transformer blocks to steer their output per the given specs.

    Args:
        model: A loaded mlx-lm / mlx-vlm model object.
        specs: Steering specs to apply together. Additive ("add") specs sum
            into a single per-layer bias; projection ("project") specs apply
            in sequence after it.
        n_embd: Optional expected hidden size; when given and derivable
            from the model, a mismatch raises rather than failing later
            inside the forward pass.

    Returns:
        The number of layers actually patched.

    Removes any prior steering patch first, so this is safe to call again
    with new specs (e.g. after a settings change).
    """
    remove_steering_patch(model)
    if not specs:
        return 0

    container = find_layers_container(model)
    if container is None:
        raise ValueError("could not locate the transformer layer list on model")
    layers = container.layers
    n_layers = len(layers)

    expected = model_hidden_size(model) or n_embd
    if expected is not None:
        for spec in specs:
            if spec.vector.n_embd != expected:
                raise ValueError(
                    f"steering vector n_embd {spec.vector.n_embd} does not "
                    f"match model n_embd {expected}"
                )

    # Resolve specs into per-layer operations: additive contributions sum;
    # projections accumulate as an ordered list.
    add_bias: dict[int, mx.array] = {}
    projections: dict[int, list[tuple[mx.array, float]]] = {}
    for spec in specs:
        active = spec.active_directions()
        if spec.mode == "add":
            # Normalise additive strength by the steered-layer count. An
            # additive bias compounds down the residual stream, so the same
            # `strength` over a wide band perturbs far more than over a
            # narrow one; dividing makes `strength` a band-width-independent
            # total budget. Projection is self-calibrating and per-layer
            # meaningful, so it is left unnormalised.
            per_layer = spec.strength / max(len(active), 1)
            for il, direction in active.items():
                contrib = (direction * per_layer).astype(mx.float32)
                add_bias[il] = (
                    contrib if il not in add_bias else add_bias[il] + contrib
                )
        else:  # "project" — the formula needs a unit direction
            for il, direction in active.items():
                norm = mx.linalg.norm(direction)
                unit = direction / norm if float(norm) > 1e-8 else direction
                projections.setdefault(il, []).append(
                    (unit.astype(mx.float32), spec.strength)
                )

    patched = 0
    for il in sorted(set(add_bias) | set(projections)):
        if il < 0 or il >= n_layers:
            logger.warning(
                "steering: layer %d out of range (model has %d layers), skipping",
                il,
                n_layers,
            )
            continue
        layers[il] = _SteeredLayer(
            layers[il], add_bias.get(il), projections.get(il, [])
        )
        patched += 1

    if patched:
        model._omlx_steering_active = True
        logger.info(
            "Steering patch applied to %d/%d layers (%d spec%s)",
            patched,
            n_layers,
            len(specs),
            "" if len(specs) == 1 else "s",
        )
    return patched


def remove_steering_patch(model: Any) -> bool:
    """Unwrap any steering-patched layers, restoring the original blocks.

    Returns True if a patch was present and removed.
    """
    if not getattr(model, "_omlx_steering_active", False):
        return False
    container = find_layers_container(model)
    if container is not None:
        layers = container.layers
        for i in range(len(layers)):
            # Unwrap defensively in case wrappers ever nested.
            while isinstance(layers[i], _SteeredLayer):
                layers[i] = layers[i]["block"]
    model._omlx_steering_active = False
    return True
