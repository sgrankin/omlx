# SPDX-License-Identifier: Apache-2.0
"""Profile a decode loop to find non-attention hot spots.

Two views:

  - Python profile (cProfile): cumulative + self time per Python function.
    Surfaces Python-level overhead (control flow, type checks, attribute
    walks) that GPU profilers miss.

  - Section timers: wall-time around named sections of the loop, with an
    explicit ``mx.eval()`` between sections to fold pending GPU work into
    the measurement. Adds a per-section sync penalty (pessimistic per
    section), but the *relative* split is honest because every section
    pays the same penalty.

Run with one of the four oQ6-fp16 target models from CLAUDE.local.md.
Defaults to Qwen3.6-35B-A3B because it's fast (~50 tok/s) so the loop
runs many times within a profile window, giving stable percentages.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_model(model_path: str):
    """VLM- and oQ-aware load (lifted from omlx CLI helpers).

    Lives here because this branch is rooted on the base layer and does not
    have the steering-feature's _load_steering_model helper available.
    """
    import json

    from omlx.utils.model_loading import (
        maybe_apply_pre_load_patches,
        maybe_load_custom_quantization,
    )

    is_vlm = False
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            is_vlm = "vision_config" in cfg or "text_config" in cfg
        except (OSError, json.JSONDecodeError):
            pass

    maybe_apply_pre_load_patches(model_path)
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


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", None) or model


def _logits(out: Any) -> mx.array:
    return out.logits if hasattr(out, "logits") else out


class SectionTimer:
    """Accumulates wall time per named section with mx.eval() boundaries."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._t0 = 0.0
        self._name = ""

    def __enter__(self) -> "SectionTimer":
        return self

    def __exit__(self, *a) -> None:
        pass

    def begin(self, name: str) -> None:
        self._name = name
        self._t0 = time.perf_counter()

    def end(self, *evals: mx.array) -> None:
        if evals:
            mx.eval(*evals)
        dt = time.perf_counter() - self._t0
        self.totals[self._name] = self.totals.get(self._name, 0.0) + dt
        self.calls[self._name] = self.calls.get(self._name, 0) + 1


def decode_loop(
    model: Any, tokenizer: Any, prompt_ids: list[int], n_steps: int,
    timer: SectionTimer | None = None,
) -> int:
    from mlx_lm.models.cache import make_prompt_cache

    tm = _text_model(model)

    if timer is not None:
        timer.begin("setup_cache")
        cache = make_prompt_cache(tm)
        timer.end()
    else:
        cache = make_prompt_cache(tm)

    # Prefill
    if timer is not None:
        timer.begin("prefill_forward")
        out = tm(mx.array(prompt_ids)[None], cache=cache)
        logits = _logits(out)[:, -1, :]
        timer.end(logits)

        timer.begin("prefill_argmax")
        nxt_arr = mx.argmax(logits[0])
        timer.end(nxt_arr)
        nxt = int(nxt_arr.item())
    else:
        out = tm(mx.array(prompt_ids)[None], cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())

    for _ in range(n_steps):
        if timer is not None:
            timer.begin("decode_input_array")
            inp = mx.array([[nxt]])
            timer.end(inp)

            timer.begin("decode_forward")
            out = tm(inp, cache=cache)
            logits = _logits(out)[:, -1, :]
            timer.end(logits)

            timer.begin("decode_argmax")
            nxt_arr = mx.argmax(logits[0])
            timer.end(nxt_arr)

            timer.begin("decode_item")
            nxt = int(nxt_arr.item())
            timer.end()
        else:
            inp = mx.array([[nxt]])
            out = tm(inp, cache=cache)
            logits = _logits(out)[:, -1, :]
            nxt = int(mx.argmax(logits[0]).item())
    return nxt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen3.6-35B-A3B-oQ6-fp16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--top", type=int, default=30, help="Top N rows for cProfile")
    parser.add_argument(
        "--prompt",
        default="Write a one-paragraph explanation of how a B-tree works.",
    )
    args = parser.parse_args()

    path = (
        args.model if Path(args.model).is_absolute()
        else str(Path.home() / ".omlx" / "models" / args.model)
    )

    print(f"Loading {path}")
    t0 = time.perf_counter()
    model, tokenizer = _load_model(path)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = list(tokenizer.encode(templated))
        except Exception:
            prompt_ids = list(tokenizer.encode(args.prompt))
    else:
        prompt_ids = list(tokenizer.encode(args.prompt))

    print(f"  prompt_ids: {len(prompt_ids)} tokens")

    # Warmup (untimed)
    decode_loop(model, tokenizer, prompt_ids, args.warmup)

    # 1) Section timing pass — coarse split with mx.eval boundaries.
    timer = SectionTimer()
    t0 = time.perf_counter()
    decode_loop(model, tokenizer, prompt_ids, args.steps, timer=timer)
    section_wall = time.perf_counter() - t0
    print(f"\n=== Section timing ({args.steps} decode steps, "
          f"wall={section_wall:.2f}s, tok/s={args.steps / section_wall:.2f}) ===")
    print(f"  {'section':<22s}  {'calls':>6s}  {'total s':>10s}  "
          f"{'per-call ms':>13s}  {'share %':>8s}")
    total = sum(timer.totals.values())
    for name in sorted(timer.totals, key=lambda n: -timer.totals[n]):
        t = timer.totals[name]
        c = timer.calls[name]
        share = 100.0 * t / total
        print(f"  {name:<22s}  {c:>6d}  {t:>10.4f}  "
              f"{1000.0 * t / c:>13.4f}  {share:>7.1f}%")

    # 2) Python profile pass — cProfile around the same loop (no section timer).
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    decode_loop(model, tokenizer, prompt_ids, args.steps)
    pr.disable()
    profile_wall = time.perf_counter() - t0
    print(f"\n=== cProfile ({args.steps} decode steps, wall={profile_wall:.2f}s, "
          f"tok/s={args.steps / profile_wall:.2f}) ===")
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(args.top)
    print(s.getvalue())

    # Repeat sorted by tottime to surface self-time hotspots
    s2 = StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(args.top)
    print("=== cProfile sorted by tottime ===")
    print(s2.getvalue())

    return 0


if __name__ == "__main__":
    sys.exit(main())
