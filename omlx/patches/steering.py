# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch: apply a steering vector to a loaded mlx-lm / mlx-vlm model.

Each transformer block in the model is wrapped so that, after it produces
its residual-stream output ``h``, the layer's steering direction is added::

    h <- h + direction[il]

The directions passed in here are already scaled by strength and filtered
to the configured layer range (see :meth:`SteeringVector.layer_map`).

Applied once at load time by ``apply_post_load_transforms`` and reversible
via :func:`remove_steering_patch`, so a cached model object can be re-used
without steering after the setting is cleared.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

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
    """Best-effort lookup of the model's hidden size (``n_embd``)."""
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
    return None


class _SteeredLayer(nn.Module):
    """Wraps a transformer block, adding a steering direction to its output.

    A real ``nn.Module`` (not a bare proxy): the wrapped block is registered
    as a child module, so its parameters stay visible to MLX's tree
    traversal — ``parameters()``, ``set_dtype``, weight save — even though
    the wrapper sits in the model's ``.layers`` list for the model's whole
    lifetime. Unlike the transient ``_AttentionCapture`` in
    ``patches/specprefill`` (removed in a ``finally``), this patch is
    persistent, so correct parameter tracking matters.

    The steering direction is stored as a plain instance attribute via
    ``object.__setattr__`` so MLX does not register it as a trainable
    parameter — it is a fixed bias, not a weight, and must not be quantized
    or serialised with the model.
    """

    def __init__(self, block: Any, direction: mx.array):
        super().__init__()
        # Register the block as a dict-child directly: when it is a real
        # nn.Module, MLX's traversal recurses into it for parameters();
        # when it is a plain object it is just a leaf entry.
        self["block"] = block
        object.__setattr__(self, "_steer_direction", direction)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self["block"](*args, **kwargs)
        # A block returns either the hidden state directly or a tuple whose
        # first element is the hidden state (some architectures also return
        # cache/aux entries). Cast the direction to the hidden dtype so a
        # bf16/fp16 residual stream is not silently promoted to f32.
        if isinstance(out, tuple):
            h = out[0]
            return (h + self._steer_direction.astype(h.dtype),) + tuple(out[1:])
        return out + self._steer_direction.astype(out.dtype)

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
    layer_map: dict[int, mx.array],
    *,
    n_embd: int | None = None,
) -> int:
    """Wrap transformer blocks so configured layers add their direction.

    Args:
        model: A loaded mlx-lm / mlx-vlm model object.
        layer_map: Maps absolute layer index -> already-scaled direction
            array (see :meth:`SteeringVector.layer_map`).
        n_embd: Optional expected hidden size; when given and derivable
            from the model, a mismatch raises rather than failing later
            inside the forward pass.

    Returns:
        The number of layers actually patched.

    Removes any prior steering patch first, so this is safe to call again
    with a new ``layer_map`` (e.g. after a settings change).
    """
    remove_steering_patch(model)
    if not layer_map:
        return 0

    container = find_layers_container(model)
    if container is None:
        raise ValueError("could not locate the transformer layer list on model")
    layers = container.layers
    n_layers = len(layers)

    expected = model_hidden_size(model) or n_embd
    if expected is not None:
        for il, direction in layer_map.items():
            if direction.shape[-1] != expected:
                raise ValueError(
                    f"steering direction for layer {il} has size "
                    f"{direction.shape[-1]}, but model n_embd is {expected}"
                )

    patched = 0
    for il, direction in layer_map.items():
        if il < 0 or il >= n_layers:
            logger.warning(
                "steering: layer %d out of range (model has %d layers), skipping",
                il,
                n_layers,
            )
            continue
        layers[il] = _SteeredLayer(layers[il], direction)
        patched += 1

    if patched:
        model._omlx_steering_active = True
        logger.info("Steering patch applied to %d/%d layers", patched, n_layers)
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
