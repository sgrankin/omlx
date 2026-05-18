# SPDX-License-Identifier: Apache-2.0
"""Evaluate a steering vector by generating at several strengths.

A debugging aid (in the spirit of ds4's evaluation tooling): apply one
steering vector at a sweep of strength values and print the generated text
for each, so the effect — and the strength at which output degrades — is
directly visible.

Generation is greedy (deterministic), so differences between scales are
attributable to the steering vector rather than sampling noise.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .patches.steering import apply_steering_patch, remove_steering_patch
from .steering import SteeringSpec, SteeringVector

logger = logging.getLogger(__name__)


def _eos_ids(tokenizer: Any) -> set[int]:
    """Collect every end-of-sequence token id the tokenizer exposes."""
    ids: set[int] = set()
    for attr in ("eos_token_ids", "eos_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is None:
            continue
        if isinstance(val, (set, list, tuple)):
            ids.update(int(v) for v in val)
        else:
            ids.add(int(val))
    return ids


def _logits(out: Any) -> mx.array:
    """Extract the logits array from a forward result (array or output obj)."""
    return out.logits if hasattr(out, "logits") else out


def _greedy_generate(
    text_model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_tokens: int,
) -> str:
    """Greedily decode up to ``max_tokens`` tokens from ``prompt_ids``."""
    from mlx_lm.models.cache import make_prompt_cache

    eos = _eos_ids(tokenizer)
    cache = make_prompt_cache(text_model)
    logits = _logits(text_model(mx.array(prompt_ids)[None], cache=cache))[:, -1, :]

    generated: list[int] = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits[0]).item())
        if nxt in eos:
            break
        generated.append(nxt)
        logits = _logits(
            text_model(mx.array([[nxt]]), cache=cache)
        )[:, -1, :]
    return tokenizer.decode(generated)


def evaluate_steering(
    model: Any,
    tokenizer: Any,
    vector: SteeringVector,
    prompt: str,
    *,
    scales: list[float],
    mode: str = "add",
    layer_start: int | None = None,
    layer_end: int | None = None,
    max_tokens: int = 200,
) -> list[tuple[float, str]]:
    """Generate ``prompt`` at each strength in ``scales``.

    A strength of 0 is generated with steering removed entirely (the
    baseline). Returns ``[(scale, generated_text), ...]`` in input order.
    """
    text_model = getattr(model, "language_model", None) or model

    prompt_ids: list[int] | None = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Template to a string, then encode — apply_chat_template's
            # tokenized return type varies (list vs BatchEncoding).
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = list(tokenizer.encode(templated))
        except Exception as e:  # noqa: BLE001 - fall back to a raw prompt
            logger.warning("chat template failed (%s); using raw prompt", e)
    if prompt_ids is None:
        prompt_ids = list(tokenizer.encode(prompt))

    results: list[tuple[float, str]] = []
    for scale in scales:
        if scale == 0.0:
            remove_steering_patch(model)
        else:
            apply_steering_patch(
                model,
                [
                    SteeringSpec(
                        vector=vector,
                        strength=scale,
                        mode=mode,
                        layer_start=layer_start,
                        layer_end=layer_end,
                    )
                ],
            )
        logger.info("Generating at steering strength %.3g", scale)
        results.append((scale, _greedy_generate(text_model, tokenizer, prompt_ids, max_tokens)))

    remove_steering_patch(model)
    return results
