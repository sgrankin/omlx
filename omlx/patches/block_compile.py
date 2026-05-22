# SPDX-License-Identifier: Apache-2.0
"""mx.compile the per-layer feed-forward submodules of a decoder.

The transformer block's post-attention feed-forward path is a chain of
many small ops (norms, residual adds, gate/router, expert matmuls). Each
is its own MLX dispatch; collapsing the chain into one compiled graph
removes the Python-level per-op dispatch overhead. Worth ~1-2% decode
throughput on MoE models. Opt-in via ``ModelSettings.block_compile_enabled``.

WHY THE SUBMODULE, NOT THE WHOLE BLOCK
--------------------------------------
An earlier version reimplemented ``DecoderLayer.__call__`` to compile the
whole post-attn body. That collides with the native-MTP patch, which
*also* replaces ``DecoderLayer.__call__`` (to thread ``n_confirmed``
through linear attention). Two patches, one method — last one wins, the
other breaks.

This version instead compiles the feed-forward **submodule**
(``mlp`` / ``experts``), which neither MTP nor anything else patches. It
composes cleanly: MTP keeps owning ``DecoderLayer.__call__``; the layer
still calls ``self.mlp(...)`` / ``self.experts(...)``, which now dispatch
a compiled graph. It also composes with the (upstream, engine-side)
gate+up fusion regardless of patch application order: ``mx.compile``
traces lazily on first forward, which happens after the engine has
already rewritten the ``SwitchGLU``, so the compiled trace captures the
fused ``gather_qmm`` (over captured-constant fused weights) no matter
when this patch was applied relative to the fusion.

The feed-forward submodules are pure (no cache, no in-place state — the
KV cache lives in attention), so compiling them is token-identical.

Targets:
  - mlx_vlm.models.qwen3_5_moe.language.Qwen3_5MoeSparseMoeBlock
  - mlx_vlm.models.gemma4.language.MLP
  - mlx_vlm.models.gemma4.language.Experts
Dense Qwen3_5MLP is intentionally not targeted — block compile measured
~0% on dense models (the single large matmul dominates dispatch cost).
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Per-instance attribute holding the compiled forward.
_COMPILED_ATTR = "_omlx_block_compiled"
# Class-level markers set when the dispatch shim / saved original are installed.
_PATCHED_FLAG = "_omlx_block_compile_patched"
_ORIG_CALL = "_omlx_block_orig_call"
# Per-instance positional arity the compiled forward was built for.
_ARITY_ATTR = "_omlx_block_compile_arity"


def _patch_class(cls: type) -> None:
    """Install a one-time dispatch shim on ``cls.__call__``.

    The shim runs the per-instance compiled forward when one has been
    attached, else the original. The original is stashed on the class so
    the compiled closures can call it.
    """
    if getattr(cls, _PATCHED_FLAG, False):
        return

    original = cls.__call__

    def patched(self, *args, **kwargs):
        # The compiled closure is built for one exact positional arity and
        # takes no keywords (mx.compile keys its trace cache on the wrapped
        # signature). Anything else falls through to the original.
        fn = getattr(self, _COMPILED_ATTR, None)
        if (
            fn is not None
            and not kwargs
            and len(args) == getattr(self, _ARITY_ATTR, -1)
        ):
            return fn(*args)
        return original(self, *args, **kwargs)

    setattr(cls, _ORIG_CALL, original)
    cls.__call__ = patched
    setattr(cls, _PATCHED_FLAG, True)


def _attach_compiled(module: Any) -> bool:
    """Compile ``module``'s forward and attach it. Returns True on success.

    The compiled closure is built with the original ``__call__``'s exact
    positional arity. This matters: ``mx.compile`` keys its trace cache on
    the wrapped function's signature — a ``*args`` wrapper defeats the
    cache and forces a re-trace on every call (catastrophically slow).
    """
    if getattr(module, _COMPILED_ATTR, None) is not None:
        return False  # idempotent
    cls = type(module)
    original = getattr(cls, _ORIG_CALL, None) or cls.__call__

    # Positional params of the original __call__, excluding `self`.
    n_args = sum(
        1
        for p in inspect.signature(original).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ) - 1

    if n_args == 1:
        def _forward(a, _m=module, _o=original):
            return _o(_m, a)
    elif n_args == 2:
        def _forward(a, b, _m=module, _o=original):
            return _o(_m, a, b)
    elif n_args == 3:
        def _forward(a, b, c, _m=module, _o=original):
            return _o(_m, a, b, c)
    else:
        return False  # unsupported arity — leave on the original path

    _patch_class(cls)
    object.__setattr__(module, _ARITY_ATTR, n_args)
    object.__setattr__(module, _COMPILED_ATTR, mx.compile(_forward))
    return True


def apply_block_compile(model: Any) -> int:
    """Compile the feed-forward submodules of every decoder layer.

    Args:
        model: A loaded mlx-lm / mlx-vlm model instance.

    Returns:
        The number of submodules compiled (0 when none of the target
        classes are importable / present).
    """
    targets: list[type] = []
    try:
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
        targets.append(Qwen3_5MoeSparseMoeBlock)
    except ImportError:
        pass
    try:
        from mlx_vlm.models.gemma4.language import MLP as _Gemma4MLP
        from mlx_vlm.models.gemma4.language import Experts as _Gemma4Experts
        targets.append(_Gemma4MLP)
        targets.append(_Gemma4Experts)
    except ImportError:
        pass

    if not targets:
        return 0
    target_tuple = tuple(targets)

    text_model = getattr(model, "language_model", None) or model
    root = getattr(text_model, "model", text_model)
    if not hasattr(root, "modules"):
        return 0

    n = 0
    for module in root.modules():
        if isinstance(module, target_tuple):
            if _attach_compiled(module):
                n += 1
    return n
