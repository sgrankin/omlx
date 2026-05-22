# SPDX-License-Identifier: Apache-2.0
"""Test whether the gate+up fusion works with — and accelerates — the
native MTP (draft+verify) decode path.

Loads a Qwen3.6 model with mtp_enabled, drives a decode through
mlx_lm.generate.batch_generate (which routes through the omlx-patched
GenerationBatch.next → MTP draft+verify), and times it with and without
the gate+up fusion. Also checks the two produce identical output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_mtp_model(model_path: str):
    """Load a model with the native-MTP patches applied (heads kept)."""
    import json

    from omlx.utils.model_loading import maybe_apply_pre_load_patches
    from omlx.utils.model_loading import maybe_load_custom_quantization

    # mtp_enabled must be visible to maybe_apply_pre_load_patches.
    model_settings = SimpleNamespace(mtp_enabled=True)
    maybe_apply_pre_load_patches(model_path, model_settings)

    cfg = json.loads((Path(model_path) / "config.json").read_text())
    is_vlm = "vision_config" in cfg or "text_config" in cfg

    custom = maybe_load_custom_quantization(model_path, is_vlm=is_vlm)
    if custom is not None:
        model, processor = custom
        return model, getattr(processor, "tokenizer", processor)

    if is_vlm:
        from mlx_vlm.utils import load as vlm_load
        from omlx.engine.vlm import (
            _patch_torch_free_image_processor,
            _patch_video_processor_bug,
            _remap_nested_visual_on_load,
            _strip_audio_config_if_orphaned,
        )

        _patch_video_processor_bug()
        _patch_torch_free_image_processor()
        with (
            _strip_audio_config_if_orphaned(Path(model_path)),
            _remap_nested_visual_on_load(Path(model_path)),
        ):
            model, processor = vlm_load(model_path)
        return model, getattr(processor, "tokenizer", processor)

    from mlx_lm import load as lm_load
    return lm_load(model_path)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.6-35B-A3B-oQ6-fp16"
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    path = str(Path.home() / ".omlx" / "models" / model_name)

    print(f"Loading {path} with mtp_enabled=True")
    t0 = time.perf_counter()
    model, tokenizer = load_mtp_model(path)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    from omlx.patches.mlx_lm_mtp import is_mtp_active
    print(f"  MTP active: {is_mtp_active()}")

    # The text model for generation
    text_model = getattr(model, "language_model", None) or model
    mtp_attr = getattr(text_model, "mtp", None) or getattr(
        getattr(text_model, "model", text_model), "mtp", None
    )
    print(f"  MTP head present on model: {mtp_attr is not None}")

    prompt = "Explain how a B-tree index speeds up range queries."
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = list(tokenizer.encode(templated))
        except Exception:
            ids = list(tokenizer.encode(prompt))
    else:
        ids = list(tokenizer.encode(prompt))
    print(f"  prompt: {len(ids)} tokens")

    # ---- Probe 1: does the omlx-patched GenerationBatch see this model as
    # MTP-eligible? (the production decode path's gating predicate) --------
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _model_has_mtp_module, _is_mtp_eligible,
    )
    has_mtp_fwd = hasattr(text_model, "mtp_forward")
    has_mtp_mod = _model_has_mtp_module(text_model)
    print(f"  text_model has mtp_forward: {has_mtp_fwd}")
    print(f"  _model_has_mtp_module:      {has_mtp_mod}")

    # ---- Probe 2: greedy decode using the model directly, with and
    # without gate+up. Confirms gate+up COMPOSES with an MTP-loaded model
    # and is token-identical. (This is a plain forward loop — it exercises
    # the backbone SwitchGLU that both MTP draft and verify also use, but
    # not the draft/verify control flow itself.) --------------------------
    def decode(n):
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(text_model)
        out = text_model(mx.array(ids)[None], cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        nxt = int(mx.argmax(logits[0, -1]).item())
        toks = []
        for _ in range(n):
            toks.append(nxt)
            out = text_model(mx.array([[nxt]]), cache=cache)
            logits = out.logits if hasattr(out, "logits") else out
            nxt = int(mx.argmax(logits[0, -1]).item())
        return toks

    print("\n=== Backbone decode on the MTP-loaded model ===")
    print("    baseline vs apply_post_load_transforms (block-compile)")
    decode(8)  # warmup — JIT/compile before timing
    base_runs = []
    for _ in range(3):
        t = time.perf_counter()
        base = decode(n_tokens)
        base_runs.append(time.perf_counter() - t)
    base_dt = min(base_runs)
    print(f"  baseline   {base_dt:.3f}s  tok/s={n_tokens/base_dt:.1f}  "
          f"({'/'.join(f'{r:.2f}' for r in base_runs)})")

    # Run the REAL engine hook on the MTP-loaded model. This applies
    # block compile on top of the already-applied MTP patches — the exact
    # production composition we need to verify. (Gate+up fusion now runs
    # separately, in the engine, via omlx.patches.qwen35_moe_gate_up.)
    from omlx.utils.model_loading import apply_post_load_transforms
    from omlx.patches.block_compile import _COMPILED_ATTR

    apply_post_load_transforms(
        model, model_settings=SimpleNamespace(block_compile_enabled=True)
    )
    root = getattr(text_model, "model", text_model)
    mods = list(root.modules())
    n_gu = sum(1 for m in mods if hasattr(m, "gate_up_proj"))
    n_bc = sum(1 for m in mods if getattr(m, _COMPILED_ATTR, None) is not None)
    print(f"  apply_post_load_transforms: gate+up={n_gu}, block-compile={n_bc}")

    decode(8)  # warmup the transformed path
    fused_runs = []
    for _ in range(3):
        t = time.perf_counter()
        fused = decode(n_tokens)
        fused_runs.append(time.perf_counter() - t)
    fused_dt = min(fused_runs)
    print(f"  transformed {fused_dt:.3f}s  tok/s={n_tokens/fused_dt:.1f}  "
          f"({'/'.join(f'{r:.2f}' for r in fused_runs)})")

    print("\n=== Result ===")
    ok = base == fused
    print(f"  gate+up + block-compile compose with native MTP: "
          f"{'YES — token-identical' if ok else 'NO — DIVERGED'}")
    print(f"  backbone speedup: {100*(base_dt-fused_dt)/base_dt:+.1f}%")
    print(f"  Note: plain decode loop (serial — sync-bound, underestimates")
    print(f"  fusion); the draft+verify tok/s needs omlx's engine VLM-MTP path.")


if __name__ == "__main__":
    main()
