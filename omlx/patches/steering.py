# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch: apply a steering vector to a loaded mlx-lm / mlx-vlm model.

Each transformer block's *class* is swapped for a per-base-class steered
subclass so that, after the block produces its residual-stream output
``h``, one or more steering specs are applied — an additive bias and/or
directional projections (see :class:`~omlx.steering.SteeringSpec` and
:func:`_steer_block`).

Applied once at load time by ``apply_post_load_transforms`` and reversible
via :func:`remove_steering_patch`, so a cached model object can be re-used
without steering after the setting is cleared.
"""

from __future__ import annotations

import contextlib
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


def _residual_dtype(container: Any) -> Any:
    """Probe the model's residual-stream compute dtype.

    Looks for an ``nn.RMSNorm`` / ``nn.LayerNorm`` inside the first
    transformer block — these are never quantized, so their weight's
    dtype is the model's compute dtype even on int4/int8 checkpoints.
    Falls back to ``fp32`` when no norm module is found (e.g. on
    synthetic test blocks): correct but slower (per-call downcast in
    :func:`_steer`, the module-level steering body). Real models hit the
    probe path.
    """
    layers = getattr(container, "layers", None) or []
    if not layers:
        return mx.float32
    block = layers[0]
    walker = getattr(block, "modules", None)
    if not callable(walker):
        return mx.float32
    fp_dtypes = (mx.float16, mx.bfloat16, mx.float32)
    for sub in walker():
        if isinstance(sub, (nn.RMSNorm, nn.LayerNorm)):
            w = getattr(sub, "weight", None)
            if w is not None and getattr(w, "dtype", None) in fp_dtypes:
                return w.dtype
    return mx.float32


def model_hidden_size(model: Any) -> int | None:
    """Best-effort lookup of the model's hidden size (``n_embd``).

    On a VLM the top-level config may carry its own ``hidden_size`` field —
    a dataclass default unrelated to the text decoder (e.g. mlx-vlm's Gemma 4
    sets ``ModelConfig.hidden_size=1536`` while the actual text stream is
    ``text_config.hidden_size=2816``). Steering wraps the text decoder's
    blocks, so prefer the language model's own hidden_size: recurse into
    ``model.language_model`` first, then check a ``text_config`` sub-config,
    and only fall back to the top-level field for pure LLMs.
    """
    lm = getattr(model, "language_model", None)
    if lm is not None and lm is not model:
        hs = model_hidden_size(lm)
        if hs:
            return hs
    for attr in ("args", "config"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        text_cfg = (
            obj.get("text_config") if isinstance(obj, dict)
            else getattr(obj, "text_config", None)
        )
        if text_cfg is not None:
            if isinstance(text_cfg, dict):
                hs = text_cfg.get("hidden_size") or text_cfg.get("n_embd")
            else:
                hs = getattr(text_cfg, "hidden_size", None) or getattr(
                    text_cfg, "n_embd", None
                )
            if hs:
                return int(hs)
        if isinstance(obj, dict):
            hs = obj.get("hidden_size") or obj.get("n_embd")
        else:
            hs = getattr(obj, "hidden_size", None) or getattr(obj, "n_embd", None)
        if hs:
            return int(hs)
    return None


# Cache of base class -> steered subclass. One subclass per block class is
# enough; all per-layer state lives on the instance.
_STEERED_CLASS_CACHE: dict[type, type] = {}


def _steer(layer: Any, h: mx.array) -> mx.array:
    """Apply ``layer``'s steering to a residual-stream tensor.

    Module-level rather than a method so the class swap in
    :func:`_steered_class` stays a two-line trampoline, and so the dev tools
    under ``tools/`` can A/B alternative bodies by rebinding this name.
    """
    fn = layer._steer_fn
    if fn is not None:
        try:
            return fn(h)
        except Exception as e:  # noqa: BLE001 — perf path must never fail a forward
            # shapeless compilation covers a changing sequence length but not
            # a changing rank. Drop to the traced-Python body permanently for
            # this layer rather than failing the forward pass.
            logger.warning(
                "steering: compiled projection body failed (%s); "
                "falling back to the uncompiled path",
                e,
            )
            object.__setattr__(layer, "_steer_fn", None)
    body = layer._steer_body
    if body is not None:
        return body(h)
    # Add-only layer (or no-op). Steering data is pre-cast to the model's
    # compute dtype at patch time (see apply_steering_patch); the cast here is
    # a safety net for the rare path where set_dtype() ran after the patch.
    add = layer._steer_add
    if add is not None:
        return h + (add if add.dtype == h.dtype else add.astype(h.dtype))
    return h


def _steered_class(cls: type) -> type:
    """Return (and memoize) a subclass of ``cls`` that steers its output."""
    steered = _STEERED_CLASS_CACHE.get(cls)
    if steered is not None:
        return steered

    def _steered_call(self, *args: Any, **kwargs: Any) -> Any:
        # Resolved dynamically (not closed over at mint time) so a later
        # class-level __call__ shim on the base class — e.g.
        # patches/block_compile.py's ``cls.__call__ = patched`` — is still
        # picked up by every steered instance, including ones minted before
        # the shim runs.
        base_call = type(self)._omlx_steer_base.__call__
        out = base_call(self, *args, **kwargs)
        # A block returns either the hidden state directly or a tuple whose
        # first element is the hidden state (some architectures also return
        # cache/aux entries).
        if isinstance(out, tuple):
            return (_steer(self, out[0]),) + tuple(out[1:])
        return _steer(self, out)

    steered = type(
        f"Steered{cls.__name__}",
        (cls,),
        {"__call__": _steered_call, "_omlx_steer_base": cls},
    )
    _STEERED_CLASS_CACHE[cls] = steered
    return steered


def is_steered(obj: Any) -> bool:
    """True when ``obj`` is a transformer block carrying a steering patch."""
    return getattr(type(obj), "_omlx_steer_base", None) is not None


def _unsteer_block(block: Any) -> Any:
    """Restore ``block``'s original class and drop its steering state."""
    base = getattr(type(block), "_omlx_steer_base", None)
    if base is None:
        return block
    block.__class__ = base
    for name in ("_steer_add", "_steer_proj", "_steer_body", "_steer_fn"):
        with contextlib.suppress(AttributeError):
            object.__delattr__(block, name)
    return block


def _steer_block(
    block: Any,
    add_bias: mx.array | None,
    projections: list[tuple[mx.array, float]],
) -> Any:
    """Steer ``block``'s residual-stream output in place; returns ``block``.

    Applies, in order, an optional additive bias and any number of
    directional projections::

        h <- h + add_bias
        h <- h - strength · (d̂ · h) · d̂      (for each projection)

    ``add_bias`` is the pre-summed contribution of every additive spec for
    this layer; projections are kept separate because they are not linear in
    the activation. See :func:`apply_steering_patch`.

    The block's *class* is swapped for a steered subclass rather than the
    block being wrapped in a proxy object. A proxy delegates reads but
    swallows writes — ``patches/specprefill`` installs its attention capture
    with ``layer.self_attn = module`` and ``Scheduler.deep_reset`` clears
    ``layer.cache``; both silently no-opped against a wrapper. The swap also
    leaves parameter paths and ``isinstance`` dispatch pointing at the real
    block.

    Steering data is stored with ``object.__setattr__`` so MLX does not
    register it as a parameter — it is fixed configuration, not a weight, and
    must not be quantized or serialised with the model.
    """
    _unsteer_block(block)
    proj_copy = [(u, float(s)) for u, s in projections]
    body = None
    compiled = None
    # Build a compiled forward fn for the projection chain. Each projection
    # contributes (mul, sum, mul, sub) — four dispatches — and at P ×
    # N_layers per forward pass that adds up to a 3–20 % decode regression on
    # tested models (Qwen3.6, Gemma 4). mx.compile traces the loop into a
    # single graph with the projection arrays as constants. shapeless=True so
    # prefill ([B, S, D]) and decode ([B, 1, D]) share one trace instead of
    # retracing per sequence length; the body is elementwise plus a last-axis
    # reduction, so it is rank- but not shape-dependent. Add-only layers are
    # not compiled: the trace cost exceeds the saving on a single elementwise
    # op.
    if proj_copy:
        add_const = add_bias

        if add_const is not None:

            def body(h: mx.array) -> mx.array:
                a = (
                    add_const
                    if add_const.dtype == h.dtype
                    else add_const.astype(h.dtype)
                )
                h = h + a
                for unit, strength in proj_copy:
                    u = unit if unit.dtype == h.dtype else unit.astype(h.dtype)
                    coeff = (h * u).sum(axis=-1, keepdims=True)
                    h = h - strength * coeff * u
                return h

        else:

            def body(h: mx.array) -> mx.array:
                for unit, strength in proj_copy:
                    u = unit if unit.dtype == h.dtype else unit.astype(h.dtype)
                    coeff = (h * u).sum(axis=-1, keepdims=True)
                    h = h - strength * coeff * u
                return h

        compiled = mx.compile(body, shapeless=True)

    # Class swap first: on an exotic block (__slots__, C type) this raises
    # TypeError before any steering state is written, so a refused swap
    # leaves the block completely untouched rather than half-steered.
    block.__class__ = _steered_class(type(block))
    object.__setattr__(block, "_steer_add", add_bias)
    object.__setattr__(block, "_steer_proj", proj_copy)
    object.__setattr__(block, "_steer_body", body)
    object.__setattr__(block, "_steer_fn", compiled)
    return block


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
    compute_dtype = _residual_dtype(container)

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
                    (unit.astype(compute_dtype), spec.strength)
                )

    patched = 0
    steer_arrays: list[mx.array] = []
    for il in sorted(set(add_bias) | set(projections)):
        if il < 0 or il >= n_layers:
            logger.warning(
                "steering: layer %d out of range (model has %d layers), skipping",
                il,
                n_layers,
            )
            continue
        # Additive contribs are accumulated in fp32 for precision, then
        # cast once to the compute dtype before being handed to the layer.
        bias = add_bias.get(il)
        if bias is not None:
            bias = bias.astype(compute_dtype)
            steer_arrays.append(bias)
        layer_projections = projections.get(il, [])
        steer_arrays.extend(unit for unit, _ in layer_projections)
        try:
            _steer_block(layers[il], bias, layer_projections)
        except TypeError as e:
            logger.warning(
                "steering: layer %d block %s does not accept a class swap "
                "(%s); skipping",
                il,
                type(layers[il]).__name__,
                e,
            )
            continue
        patched += 1

    if patched:
        # Materialize on the calling (loader) thread. These arrays are
        # hidden from MLX parameter traversal (object.__setattr__ storage,
        # tuples inside a plain list), so materialize_lazy_state never
        # reaches them; left lazy they stay bound to this thread's stream
        # and the first forward on a per-engine inference thread dies with
        # "There is no Stream(gpu, N) in current thread".
        mx.eval(steer_arrays)
        model._omlx_steering_active = True
        logger.info(
            "Steering patch applied to %d/%d layers (%d spec%s)",
            patched,
            n_layers,
            len(specs),
            "" if len(specs) == 1 else "s",
        )

    # Written unconditionally (not just on success): specs non-empty but
    # patched == 0 — every target layer out of range, or every class swap
    # refused — is exactly the silently-unsteered case this status exists to
    # surface. See ``_engine_steering_status`` / admin's model list.
    object.__setattr__(
        model,
        "_omlx_steering_status",
        {
            "active": patched > 0,
            "layers": patched,
            "specs": len(specs),
            "error": None if patched else "no layer could be steered",
        },
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
        for layer in layers:
            _unsteer_block(layer)
    model._omlx_steering_active = False
    object.__setattr__(model, "_omlx_steering_status", None)
    object.__setattr__(model, "_omlx_steering_digest", "")
    return True
