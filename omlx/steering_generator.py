# SPDX-License-Identifier: Apache-2.0
"""Generate a steering vector from contrastive prompt pairs.

Given pairs of (positive, negative) prompts that differ only in the
behaviour we want to steer toward, this module:

  1. runs the model on each prompt and captures, per transformer layer,
     the residual-stream hidden state at the final token;
  2. forms the per-pair difference ``h_pos - h_neg`` for each layer;
  3. reduces the stack of differences to a single direction per layer,
     either by averaging ("mean") or by taking the leading principal
     component ("pca");
  4. normalises each direction and packages them as a
     :class:`~omlx.steering.SteeringVector`.

This mirrors llama.cpp's ``cvector-generator`` (and the "repeng"
approach), reimplemented natively against MLX.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .patches.steering import find_layers_container
from .steering import SteeringVector

logger = logging.getLogger(__name__)


class _HiddenCapture(nn.Module):
    """Wraps a transformer block to record its final-token hidden state.

    Forward behaviour is unchanged; the layer's output for the last
    position is written into ``sink[layer_idx]`` on every call. A real
    ``nn.Module`` (see :class:`~omlx.patches.steering._SteeredLayer`) so the
    wrapped block's parameters stay visible during the capture pass; the
    wrapper is transient and removed once capture finishes.
    """

    def __init__(self, block: Any, sink: dict[int, mx.array], layer_idx: int):
        super().__init__()
        self["block"] = block
        object.__setattr__(self, "_cap_sink", sink)
        object.__setattr__(self, "_cap_layer_idx", layer_idx)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self["block"](*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if h.ndim != 3 or h.shape[0] != 1:
            raise RuntimeError(
                f"hidden-state capture expects a [1, seq, n_embd] block "
                f"output, got shape {tuple(h.shape)}"
            )
        # Keep the final-token row of the single batch element.
        self._cap_sink[self._cap_layer_idx] = h[0, -1, :]
        return out

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self["block"], name)


def _collect_hidden(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    sink: dict[int, mx.array],
    n_layers: int,
) -> dict[int, mx.array]:
    """Run the model on each prompt; return ``{layer: [n_prompts, n_embd]}``."""
    from mlx_lm.models.cache import make_prompt_cache

    acc: dict[int, list[mx.array]] = {il: [] for il in range(n_layers)}
    for idx, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            raise ValueError(f"prompt {idx} tokenized to an empty sequence")
        tokens = mx.array(ids)[None]  # [1, seq]
        sink.clear()
        model(tokens, cache=make_prompt_cache(model))
        mx.eval(list(sink.values()))
        missing = [il for il in range(n_layers) if il not in sink]
        if missing:
            raise RuntimeError(
                f"no hidden state captured for layers {missing} on prompt {idx}"
            )
        for il in range(n_layers):
            acc[il].append(sink[il])
    return {il: mx.stack(acc[il], axis=0) for il in range(n_layers)}


def _pca_direction(diff: mx.array) -> mx.array:
    """Leading principal component of a ``[n_pairs, n_embd]`` difference set.

    Computed with NumPy's SVD on the mean-centred differences. The sign is
    oriented so the (uncentred) differences project positively — i.e. the
    direction points from "negative" toward "positive".
    """
    import numpy as np

    raw = np.asarray(diff, dtype=np.float32)
    centred = raw - raw.mean(axis=0, keepdims=True)
    # full_matrices=False keeps this cheap: vt is [min(n,d), n_embd].
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    if float((raw @ direction).mean()) < 0.0:
        direction = -direction
    return mx.array(direction)


def generate_steering_vector(
    model: Any,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
    method: str = "pca",
    model_name: str = "",
    layers: list[int] | None = None,
) -> SteeringVector:
    """Build a :class:`SteeringVector` from contrastive prompt pairs.

    Args:
        model: A loaded mlx-lm model object.
        tokenizer: Its tokenizer (must expose ``.encode``).
        positive: Prompts exhibiting the target behaviour.
        negative: Prompts exhibiting the opposite; must be the same length
            as ``positive`` (pair ``i`` is ``positive[i]`` vs ``negative[i]``).
        method: "pca" (leading principal component) or "mean".
        model_name: Free-form provenance string stored in the vector.
        layers: Restrict generation to these layer indices (default: all).

    Returns:
        A normalised SteeringVector with one direction per requested layer.
    """
    if method not in ("pca", "mean"):
        raise ValueError(f"unknown method {method!r}; expected 'pca' or 'mean'")
    if len(positive) != len(negative):
        raise ValueError(
            f"positive/negative prompt counts differ: "
            f"{len(positive)} vs {len(negative)}"
        )
    if len(positive) == 0:
        raise ValueError("need at least one contrastive prompt pair")
    if method == "pca" and len(positive) < 2:
        raise ValueError("PCA needs at least 2 prompt pairs; use method='mean'")
    if getattr(model, "language_model", None) is not None:
        raise ValueError(
            "steering vector generation supports text LLMs only; got a "
            "VLM-shaped model (has a .language_model submodule)"
        )

    container = find_layers_container(model)
    if container is None:
        raise ValueError("could not locate the transformer layer list on model")
    block_list = container.layers
    n_layers = len(block_list)

    target_layers = sorted(layers) if layers is not None else list(range(n_layers))
    for il in target_layers:
        if il < 0 or il >= n_layers:
            raise ValueError(f"layer {il} out of range (model has {n_layers})")

    # Wrap every block to capture hidden states, capture, then always
    # restore the originals — even if the forward pass raises.
    sink: dict[int, mx.array] = {}
    originals = list(block_list)
    for i in range(n_layers):
        block_list[i] = _HiddenCapture(block_list[i], sink, i)
    try:
        logger.info("Capturing hidden states for %d positive prompts", len(positive))
        pos_hidden = _collect_hidden(model, tokenizer, positive, sink, n_layers)
        logger.info("Capturing hidden states for %d negative prompts", len(negative))
        neg_hidden = _collect_hidden(model, tokenizer, negative, sink, n_layers)
    finally:
        for i in range(n_layers):
            block_list[i] = originals[i]

    directions: dict[int, mx.array] = {}
    for il in target_layers:
        diff = pos_hidden[il].astype(mx.float32) - neg_hidden[il].astype(mx.float32)
        vec = diff.mean(axis=0) if method == "mean" else _pca_direction(diff)
        norm = mx.linalg.norm(vec)
        if float(norm) < 1e-8:
            logger.warning("layer %d direction is near-zero; leaving unnormalised", il)
        else:
            vec = vec / norm
        directions[il] = vec.astype(mx.float32)

    n_embd = int(next(iter(directions.values())).shape[-1])
    logger.info(
        "Generated steering vector: %d layers, n_embd=%d, method=%s",
        len(directions),
        n_embd,
        method,
    )
    return SteeringVector(
        directions=directions,
        n_embd=n_embd,
        method=method,
        model=model_name,
    )
