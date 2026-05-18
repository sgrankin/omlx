# SPDX-License-Identifier: Apache-2.0
"""Steering vectors (a.k.a. control vectors) for oMLX.

A steering vector is a per-layer additive bias on the residual stream.
After transformer block ``il`` produces its output ``h``, steering applies::

    h <- h + strength * direction[il]

This nudges the model's activations toward a behaviour captured by the
vector (see :mod:`omlx.steering_generator`) without retraining or LoRA.

Steering is configured per loaded model and applied uniformly to every
request the model serves (matching llama.cpp's per-context control
vectors); see :mod:`omlx.patches.steering` for the application patch.

Native on-disk format — a safetensors file holding:

  - one F32 tensor per steered layer, named ``direction.<layer_index>``
  - string metadata: ``omlx_steering_version``, ``n_embd``, ``method``,
    ``model``

Layer indices are absolute model layer indices. Not every layer needs a
direction; layers without one are simply left unsteered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)

# Bump when the on-disk layout changes incompatibly.
STEERING_FORMAT_VERSION = 1

_TENSOR_PREFIX = "direction."


@dataclass
class SteeringVector:
    """A set of per-layer steering directions for one model.

    Attributes:
        directions: Maps absolute layer index -> 1-D direction array of
            length ``n_embd``.
        n_embd: Model hidden size the directions were built for.
        method: How the vector was produced ("pca", "mean", "manual").
        model: Identifier of the source model (free-form, for provenance).
    """

    directions: dict[int, mx.array] = field(default_factory=dict)
    n_embd: int = 0
    method: str = "manual"
    model: str = ""

    def __post_init__(self) -> None:
        if not self.n_embd and self.directions:
            self.n_embd = int(next(iter(self.directions.values())).shape[-1])
        for il, vec in self.directions.items():
            if vec.ndim != 1:
                raise ValueError(
                    f"steering direction for layer {il} must be 1-D, "
                    f"got shape {tuple(vec.shape)}"
                )
            if vec.shape[0] != self.n_embd:
                raise ValueError(
                    f"steering direction for layer {il} has size "
                    f"{vec.shape[0]}, expected n_embd={self.n_embd}"
                )

    @property
    def layers(self) -> list[int]:
        """Sorted list of layer indices that carry a direction."""
        return sorted(self.directions)

    def save(self, path: str | Path) -> None:
        """Write the steering vector to a native safetensors file."""
        if not self.directions:
            raise ValueError("refusing to save an empty steering vector")
        arrays = {
            f"{_TENSOR_PREFIX}{il}": vec.astype(mx.float32)
            for il, vec in self.directions.items()
        }
        metadata = {
            "omlx_steering_version": str(STEERING_FORMAT_VERSION),
            "n_embd": str(self.n_embd),
            "method": self.method,
            "model": self.model,
        }
        mx.save_safetensors(str(path), arrays, metadata=metadata)
        logger.info(
            "Saved steering vector (%d layers, n_embd=%d, method=%s) to %s",
            len(self.directions),
            self.n_embd,
            self.method,
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> SteeringVector:
        """Load a steering vector from a native safetensors file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"steering vector file not found: {path}")

        arrays, metadata = mx.load(str(path), return_metadata=True)
        metadata = metadata or {}

        directions: dict[int, mx.array] = {}
        for name, vec in arrays.items():
            if not name.startswith(_TENSOR_PREFIX):
                logger.warning("ignoring unexpected tensor %r in %s", name, path)
                continue
            suffix = name[len(_TENSOR_PREFIX):]
            try:
                il = int(suffix)
            except ValueError as e:
                raise ValueError(
                    f"malformed steering tensor name {name!r} in {path} "
                    "(expected 'direction.<int>')"
                ) from e
            directions[il] = vec.astype(mx.float32)

        if not directions:
            raise ValueError(f"no 'direction.*' tensors found in {path}")

        n_embd_meta = metadata.get("n_embd")
        n_embd = (
            int(n_embd_meta)
            if n_embd_meta
            else int(next(iter(directions.values())).shape[-1])
        )
        return cls(
            directions=directions,
            n_embd=n_embd,
            method=metadata.get("method", "manual") or "manual",
            model=metadata.get("model", "") or "",
        )

    def layer_map(
        self,
        strength: float = 1.0,
        layer_start: int | None = None,
        layer_end: int | None = None,
    ) -> dict[int, mx.array]:
        """Return ``{layer_index: scaled direction}`` for the active range.

        ``layer_start`` and ``layer_end`` are inclusive bounds; ``None``
        means unbounded on that side. Each direction is scaled by
        ``strength`` and cast to F32.
        """
        result: dict[int, mx.array] = {}
        for il, vec in self.directions.items():
            if layer_start is not None and il < layer_start:
                continue
            if layer_end is not None and il > layer_end:
                continue
            result[il] = (vec * strength).astype(mx.float32)
        return result
