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


def get_steering_dir() -> Path:
    """Return the directory where steering vector files are kept.

    A sibling of the models and cache directories under the oMLX home —
    ``~/.omlx/steering``. Generated vectors land here by default and the
    admin UI lists it; callers that write to it create it lazily.
    """
    from .settings import DEFAULT_BASE_PATH

    return DEFAULT_BASE_PATH / "steering"


def get_bundled_datasets_dir() -> Path:
    """Directory of contrastive prompt datasets shipped with oMLX."""
    return Path(__file__).parent / "data" / "steering"


def get_user_datasets_dir() -> Path:
    """User-local contrastive prompt datasets (``~/.omlx/steering/datasets``)."""
    return get_steering_dir() / "datasets"


def list_datasets() -> dict[str, Path]:
    """Map dataset name -> path, merging bundled and user-local datasets.

    A user-local dataset shadows a bundled one of the same name.
    """
    out: dict[str, Path] = {}
    for directory in (get_bundled_datasets_dir(), get_user_datasets_dir()):
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                out[path.stem] = path
    return out


def resolve_dataset(spec: str) -> Path:
    """Resolve a ``--prompts`` value to a dataset file.

    An existing path is returned as-is; otherwise ``spec`` is treated as a
    dataset name and looked up among the user-local and bundled datasets.
    Raises FileNotFoundError (listing what is available) on a miss.
    """
    path = Path(spec)
    if path.exists():
        return path
    name = spec[:-5] if spec.endswith(".json") else spec
    datasets = list_datasets()
    if name in datasets:
        return datasets[name]
    available = ", ".join(sorted(datasets)) or "(none)"
    raise FileNotFoundError(
        f"steering dataset {spec!r} not found — not an existing path, and "
        f"not a known dataset name. Available datasets: {available}"
    )


@dataclass
class SteeringVector:
    """A set of per-layer steering directions for one model.

    Attributes:
        directions: Maps absolute layer index -> 1-D direction array of
            length ``n_embd``.
        n_embd: Model hidden size the directions were built for.
        method: How the vector was produced ("pca", "mean", "manual").
        scaling: Per-layer scaling applied at generation time — "unit"
            (each direction normalised to unit length) or "magnitude"
            (scaled by the mean projection magnitude, so a single strength
            knob behaves consistently across layers).
        model: Identifier of the source model (free-form, for provenance).
    """

    directions: dict[int, mx.array] = field(default_factory=dict)
    n_embd: int = 0
    method: str = "manual"
    scaling: str = "unit"
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
            "scaling": self.scaling,
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
            scaling=metadata.get("scaling", "unit") or "unit",
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


# Supported application modes (see SteeringSpec.mode).
STEERING_MODES = ("add", "project")


@dataclass
class SteeringSpec:
    """One steering vector applied with a mode, strength and layer range.

    A model may be steered by several specs at once (see
    :func:`omlx.patches.steering.apply_steering_patch`): additive specs sum
    into a single per-layer bias; projection specs apply in sequence.

    Attributes:
        vector: The steering directions to apply.
        strength: Scale factor. For "add" mode it is a band-width-independent
            total budget — :func:`~omlx.patches.steering.apply_steering_patch`
            divides it by the number of steered layers, so a given strength
            perturbs comparably whether applied over a narrow or wide band.
            For "project" mode it is per-layer (not normalised): 1.0 fully
            removes the direction's component from the activation, 0 is a
            no-op, <0 amplifies it, >1 flips it.
        mode: "add" — additive residual-stream bias (``h += strength·d``);
            "project" — directional projection
            (``h -= strength·(d̂·h)·d̂``), which is self-calibrating across
            layers because it scales with the activation itself.
        layer_start: First layer to steer (inclusive; None = unbounded).
        layer_end: Last layer to steer (inclusive; None = unbounded).
    """

    vector: SteeringVector
    strength: float = 1.0
    mode: str = "add"
    layer_start: int | None = None
    layer_end: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in STEERING_MODES:
            raise ValueError(
                f"unknown steering mode {self.mode!r}; "
                f"expected one of {STEERING_MODES}"
            )

    def active_directions(self) -> dict[int, mx.array]:
        """Directions within the configured layer range (unscaled)."""
        out: dict[int, mx.array] = {}
        for il, vec in self.vector.directions.items():
            if self.layer_start is not None and il < self.layer_start:
                continue
            if self.layer_end is not None and il > self.layer_end:
                continue
            out[il] = vec
        return out
