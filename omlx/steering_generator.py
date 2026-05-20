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
    text_model: Any,
    tokenizer: Any,
    prompts: list[str],
    sink: dict[int, mx.array],
    n_layers: int,
) -> dict[int, mx.array]:
    """Run a text forward on each prompt; return ``{layer: [n_prompts, n_embd]}``.

    ``text_model`` is the object whose forward pass is a pure-text decode —
    the model itself for an LLM, or ``model.language_model`` for a VLM.
    """
    from mlx_lm.models.cache import make_prompt_cache

    acc: dict[int, list[mx.array]] = {il: [] for il in range(n_layers)}
    for idx, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            raise ValueError(f"prompt {idx} tokenized to an empty sequence")
        tokens = mx.array(ids)[None]  # [1, seq]
        sink.clear()
        text_model(tokens, cache=make_prompt_cache(text_model))
        mx.eval(list(sink.values()))
        missing = [il for il in range(n_layers) if il not in sink]
        if missing:
            raise RuntimeError(
                f"no hidden state captured for layers {missing} on prompt {idx}"
            )
        for il in range(n_layers):
            acc[il].append(sink[il])
    return {il: mx.stack(acc[il], axis=0) for il in range(n_layers)}


def _capture_layer_states(
    model: Any,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
) -> tuple[dict[int, mx.array], dict[int, mx.array], int]:
    """Capture per-layer final-token hidden states for both prompt sets.

    Wraps every transformer block, runs a text forward on each prompt, and
    always restores the original blocks — even if a forward pass raises.
    Returns ``(pos_hidden, neg_hidden, n_layers)`` where each hidden map is
    ``{layer: [n_prompts, n_embd]}``. Shared by :func:`generate_steering_vector`
    and :func:`analyze_layers`.
    """
    # For a VLM, drive the text decoder directly so the forward pass is a
    # pure-text decode (the top-level forward expects vision inputs).
    text_model = getattr(model, "language_model", None) or model

    container = find_layers_container(model)
    if container is None:
        raise ValueError("could not locate the transformer layer list on model")
    block_list = container.layers
    n_layers = len(block_list)

    sink: dict[int, mx.array] = {}
    originals = list(block_list)
    for i in range(n_layers):
        block_list[i] = _HiddenCapture(block_list[i], sink, i)
    try:
        logger.info("Capturing hidden states for %d positive prompts", len(positive))
        pos_hidden = _collect_hidden(text_model, tokenizer, positive, sink, n_layers)
        logger.info("Capturing hidden states for %d negative prompts", len(negative))
        neg_hidden = _collect_hidden(text_model, tokenizer, negative, sink, n_layers)
    finally:
        for i in range(n_layers):
            block_list[i] = originals[i]
    return pos_hidden, neg_hidden, n_layers


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


def _crosscov_direction(
    pos: mx.array, neg: mx.array, n_candidates: int = 3
) -> mx.array:
    """Cross-covariance contrastive axis (jukofyork-style).

    Unlike mean/PCA — which work on the per-pair differences and so fold in
    whatever else differs between paired prompts — this centres both classes
    on their joint mean and forms the symmetrized cross-covariance
    ``C = (AᵀB + BᵀA) / 2``. For a direction ``d``, ``dᵀC d`` is the
    sample sum of ``(posᵢ·d)·(negᵢ·d)``: it is most negative when the two
    classes deviate *oppositely* from the baseline along ``d`` (a genuine
    contrastive axis) and positive for directions they share (confounds).

    So the steering axis is the eigenvector with the most-negative
    eigenvalue. Among the ``n_candidates`` most-negative eigenvectors the
    one with the best Fisher discriminant ratio is chosen, covering the
    case where the trait is not cleanly the single top eigenvector.

    Needs many samples to estimate ``C`` well (ideally on the order of
    ``n_embd`` prompt pairs); degrades gracefully with fewer.
    """
    import numpy as np

    p = np.asarray(pos, dtype=np.float32)  # [n_pairs, n_embd]
    n = np.asarray(neg, dtype=np.float32)
    mu = np.concatenate([p, n], axis=0).mean(axis=0, keepdims=True)
    a = p - mu
    b = n - mu
    cov = a.T @ b
    cov = 0.5 * (cov + cov.T)
    # eigh returns eigenvalues ascending — index 0 is the most negative.
    _eigvals, eigvecs = np.linalg.eigh(cov)

    best_d = eigvecs[:, 0]
    best_ratio = -1.0
    for i in range(min(n_candidates, eigvecs.shape[1])):
        d = eigvecs[:, i]
        proj_p, proj_n = p @ d, n @ d
        within = float(proj_p.var() + proj_n.var())
        ratio = (float(proj_p.mean() - proj_n.mean())) ** 2 / (within + 1e-8)
        if ratio > best_ratio:
            best_ratio, best_d = ratio, d

    direction = best_d
    if float(((p - n) @ direction).mean()) < 0.0:
        direction = -direction
    return mx.array(direction)


def _orthogonalize(direction: mx.array, control_mean: mx.array) -> mx.array:
    """Remove the component of ``direction`` parallel to the control mean.

    ds4's default generator step: a contrastive direction can pick up a
    component aligned with where the control-class activations sit in
    general — steering along that merely scales activations rather than
    moving the trait. Projecting it out leaves a cleaner contrastive axis.
    Returns a unit vector (or ``direction`` unchanged if degenerate).
    """
    base_norm = mx.linalg.norm(control_mean)
    if float(base_norm) < 1e-8:
        return direction
    base = control_mean / base_norm
    direction = direction - mx.sum(direction * base) * base
    norm = mx.linalg.norm(direction)
    if float(norm) < 1e-8:
        return direction
    return direction / norm


def generate_steering_vector(
    model: Any,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
    method: str = "mean",
    model_name: str = "",
    layers: list[int] | None = None,
    scaling: str = "magnitude",
    orthogonalize: bool = True,
) -> SteeringVector:
    """Build a :class:`SteeringVector` from contrastive prompt pairs.

    Args:
        model: A loaded mlx-lm LLM, or an mlx-vlm VLM — for a VLM the text
            decoder (``model.language_model``) is driven directly so the
            forward pass is pure text.
        tokenizer: Its tokenizer (must expose ``.encode``).
        positive: Prompts exhibiting the target behaviour.
        negative: Prompts exhibiting the opposite; must be the same length
            as ``positive`` (pair ``i`` is ``positive[i]`` vs ``negative[i]``).
        method: "mean" (default — average of per-pair differences; robust
            with few pairs), "crosscov" (cross-covariance contrastive axis
            — cleaner separation of the trait from confounds, but wants
            many prompt pairs ~n_embd), or "pca" (leading principal
            component of the differences; discouraged — its axis is
            confound-dominated).
        model_name: Free-form provenance string stored in the vector.
        layers: Restrict generation to these layer indices. Default: all
            layers except the last — steering the final layer perturbs the
            pre-logit state directly and its direction is a magnitude
            outlier. An explicit list is honoured exactly (last included).
        scaling: "magnitude" (default) — each direction scaled by its mean
            projection magnitude, so one additive strength behaves
            consistently across layers (the residual stream grows ~35x in
            norm with depth; magnitude scaling tracks it, corr ~0.99). Or
            "unit" — every direction normalised to unit length.
        orthogonalize: Project each layer's direction orthogonal to that
            layer's control-class (negative) mean before scaling — strips
            the component aligned with general activation drift, leaving a
            cleaner trait axis. Default True, matching ds4's generator.

    Returns:
        A SteeringVector with one direction per requested layer.
    """
    if method not in ("pca", "mean", "crosscov"):
        raise ValueError(
            f"unknown method {method!r}; expected 'mean', 'pca' or 'crosscov'"
        )
    if scaling not in ("unit", "magnitude"):
        raise ValueError(
            f"unknown scaling {scaling!r}; expected 'unit' or 'magnitude'"
        )
    if len(positive) != len(negative):
        raise ValueError(
            f"positive/negative prompt counts differ: "
            f"{len(positive)} vs {len(negative)}"
        )
    if len(positive) == 0:
        raise ValueError("need at least one contrastive prompt pair")
    if method in ("pca", "crosscov") and len(positive) < 2:
        raise ValueError(
            f"method={method!r} needs at least 2 prompt pairs; use method='mean'"
        )

    pos_hidden, neg_hidden, n_layers = _capture_layer_states(
        model, tokenizer, positive, negative
    )

    if layers is not None:
        target_layers = sorted(layers)
    else:
        # Skip the final layer: steering it perturbs the pre-logit state
        # directly, and its direction is a magnitude outlier.
        target_layers = list(range(n_layers - 1)) or [0]
    for il in target_layers:
        if il < 0 or il >= n_layers:
            raise ValueError(f"layer {il} out of range (model has {n_layers})")

    directions: dict[int, mx.array] = {}
    for il in target_layers:
        pos_h = pos_hidden[il].astype(mx.float32)
        neg_h = neg_hidden[il].astype(mx.float32)
        diff = pos_h - neg_h
        if method == "mean":
            vec = diff.mean(axis=0)
        elif method == "pca":
            vec = _pca_direction(diff)
        else:  # crosscov
            vec = _crosscov_direction(pos_h, neg_h)
        norm = mx.linalg.norm(vec)
        if float(norm) < 1e-8:
            logger.warning("layer %d direction is near-zero; leaving unscaled", il)
            directions[il] = vec.astype(mx.float32)
            continue
        unit = vec / norm
        if orthogonalize:
            unit = _orthogonalize(unit, neg_h.mean(axis=0))
        if scaling == "magnitude":
            # Scale the unit direction by the mean signed projection of the
            # per-pair differences onto it — compensates for the residual
            # stream growing with depth, so one strength knob works for all
            # layers (jukofyork's per-layer scaling).
            magnitude = float(mx.abs((diff @ unit).mean()))
            directions[il] = (unit * magnitude).astype(mx.float32)
        else:
            directions[il] = unit.astype(mx.float32)

    n_embd = int(next(iter(directions.values())).shape[-1])
    logger.info(
        "Generated steering vector: %d layers, n_embd=%d, method=%s, "
        "scaling=%s, orthogonalize=%s",
        len(directions),
        n_embd,
        method,
        scaling,
        orthogonalize,
    )

    # Score the captured axis over the kept layers so a poor dataset shows
    # up at generate time rather than at first eval.
    kept_stats = [
        {
            "layer": il,
            "separation": (m := _per_layer_metrics(pos_hidden[il], neg_hidden[il]))[0],
            "consistency": m[1],
        }
        for il in target_layers
    ]
    summary, quality_warnings = _quality_summary(kept_stats)
    logger.info("Axis quality: %s", summary)
    for w in quality_warnings:
        logger.warning("Axis quality: %s", w)

    return SteeringVector(
        directions=directions,
        n_embd=n_embd,
        method=method,
        scaling=scaling,
        model=model_name,
    )


def _per_layer_metrics(pos: mx.array, neg: mx.array) -> tuple[float, float]:
    """Score a layer's contrast: ``(separation, consistency)``.

    - **separation** — Cohen's-d effect size of the two classes projected
      onto their mean-difference direction. Standardized; unbounded above.
    - **consistency** — ``|mean(diff)| / mean|diff|`` over the per-pair
      differences. 1.0 if every pair points the same way, →0 if scattered.

    Shared by :func:`analyze_layers` and :func:`generate_steering_vector`
    so the "is this dataset clean?" signal is computed the same way.
    """
    import numpy as np

    p = np.asarray(pos, dtype=np.float32)
    n = np.asarray(neg, dtype=np.float32)
    diff = p - n
    mean_diff = diff.mean(axis=0)
    md_norm = float(np.linalg.norm(mean_diff))
    if md_norm < 1e-8:
        return 0.0, 0.0
    unit = mean_diff / md_norm
    proj_p, proj_n = p @ unit, n @ unit
    pooled = float(np.sqrt(0.5 * (proj_p.var() + proj_n.var()))) + 1e-8
    sep = float(abs(proj_p.mean() - proj_n.mean()) / pooled)
    cons = md_norm / (float(np.linalg.norm(diff, axis=1).mean()) + 1e-8)
    return sep, cons


def _quality_summary(layer_stats: list[dict]) -> tuple[str, list[str]]:
    """One-line verdict + any warnings from per-layer separation/consistency.

    Thresholds are empirical ("usable steering axis" floor):

    - separation: ≥0.8 strong, ≥0.5 good, ≥0.3 marginal, <0.3 weak
    - consistency: ≥0.5 coherent, ≥0.3 mixed, <0.3 scattered

    Returns ``(summary_line, [warning, ...])``.
    """
    if not layer_stats:
        return "no layers scored", []
    peak_sep = max(layer_stats, key=lambda r: r["separation"])
    peak_cons = max(layer_stats, key=lambda r: r["consistency"])
    sep_word = (
        "strong" if peak_sep["separation"] >= 0.8
        else "good" if peak_sep["separation"] >= 0.5
        else "marginal" if peak_sep["separation"] >= 0.3
        else "weak"
    )
    cons_word = (
        "coherent" if peak_cons["consistency"] >= 0.5
        else "mixed" if peak_cons["consistency"] >= 0.3
        else "scattered"
    )
    summary = (
        f"peak separation {peak_sep['separation']:.3f} ({sep_word}) at "
        f"layer {peak_sep['layer']}; peak consistency "
        f"{peak_cons['consistency']:.3f} ({cons_word}) at layer {peak_cons['layer']}"
    )
    warnings: list[str] = []
    if peak_sep["separation"] < 0.3:
        warnings.append(
            "weak separation at every layer — the prompt pairs may not "
            "isolate a consistent trait. Check pair matching (same topic, "
            "only the trait varying), add more pairs, or revisit polarity."
        )
    if peak_cons["consistency"] < 0.3:
        warnings.append(
            "scattered per-pair differences — a few outlier pairs likely "
            "dominate the mean direction. Tighter prompt-pair matching "
            "usually helps."
        )
    return summary, warnings


def _suggest_band(
    separations: list[float], threshold: float = 0.6
) -> tuple[int, int] | None:
    """Pick a contiguous layer band from per-layer separation scores.

    ``separations[i]`` is layer ``i``'s score. The last layer is excluded
    from the suggestion — steering it perturbs the pre-logit state. Returns
    the contiguous run containing the peak-separation layer over which
    separation stays at or above ``threshold`` of the peak, or ``None`` if
    every score is ~0.
    """
    scored = separations[:-1] if len(separations) > 1 else list(separations)
    if not scored or max(scored) <= 1e-8:
        return None
    peak = max(scored)
    peak_layer = scored.index(peak)
    cutoff = threshold * peak
    start = end = peak_layer
    while start - 1 >= 0 and scored[start - 1] >= cutoff:
        start -= 1
    while end + 1 < len(scored) and scored[end + 1] >= cutoff:
        end += 1
    return (start, end)


def analyze_layers(
    model: Any,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
    *,
    threshold: float = 0.6,
) -> dict:
    """Score each layer's separability for a contrastive prompt set.

    Captures per-layer activations once (no generation) and, for each
    layer, measures how well the positive and negative prompts separate:

    - **separation** — a Cohen's-d effect size: the standardized gap
      between the two classes' projections onto the layer's mean-difference
      direction. Large where the layer cleanly encodes the trait.
    - **consistency** — ``|mean(diff)| / mean|diff|`` over the per-pair
      differences; 1.0 if every pair points the same way, →0 if scattered.

    Returns ``{"layers": [{"layer", "separation", "consistency"}, ...],
    "suggested": (start, end) | None}``. The suggested band is the
    contiguous run around the peak-separation layer (last layer excluded)
    where separation stays above ``threshold`` of the peak — read straight
    from one capture, rather than a blind generate/evaluate sweep of ranges.
    """
    if len(positive) != len(negative):
        raise ValueError(
            f"positive/negative prompt counts differ: "
            f"{len(positive)} vs {len(negative)}"
        )
    if len(positive) == 0:
        raise ValueError("need at least one contrastive prompt pair")

    pos_hidden, neg_hidden, n_layers = _capture_layer_states(
        model, tokenizer, positive, negative
    )

    layer_stats: list[dict] = []
    separations: list[float] = []
    for il in range(n_layers):
        sep, cons = _per_layer_metrics(pos_hidden[il], neg_hidden[il])
        separations.append(sep)
        layer_stats.append(
            {"layer": il, "separation": sep, "consistency": cons}
        )

    summary, warnings = _quality_summary(layer_stats)
    return {
        "layers": layer_stats,
        "suggested": _suggest_band(separations, threshold),
        "summary": summary,
        "warnings": warnings,
    }
