# SPDX-License-Identifier: Apache-2.0
"""Capture a Metal .gputrace for a short decode segment.

mx.metal.start_capture() requires the
MTL_CAPTURE_ENABLED=1 env var (set before the Metal driver loads) and a
.gputrace path that does not yet exist. The capture covers everything
between start_capture() and stop_capture(), so we keep the window narrow
(a single decode forward pass) to avoid producing a multi-GB bundle.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the loader from the profile script
from tools.profile_decode import _load_model, _logits, _text_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="gemma-4-26B-A4B-it-oQ6-fp16")
    parser.add_argument(
        "--out", default="/tmp/omlx_decode.gputrace",
        help="Output .gputrace path (must not already exist)",
    )
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--capture-steps", type=int, default=1,
                        help="Number of decode steps to capture")
    parser.add_argument("--prompt", default="What is kernel fusion?")
    args = parser.parse_args()

    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        print("ERROR: MTL_CAPTURE_ENABLED must be set to 1 BEFORE this process "
              "starts. Re-run with `MTL_CAPTURE_ENABLED=1 uv run python "
              "tools/capture_metal_trace.py ...`")
        return 2

    if Path(args.out).exists():
        print(f"ERROR: {args.out} already exists. Remove it or pick a new path.")
        return 2

    path = (
        args.model if Path(args.model).is_absolute()
        else str(Path.home() / ".omlx" / "models" / args.model)
    )
    print(f"Loading {path}")
    model, tokenizer = _load_model(path)
    tm = _text_model(model)
    print(f"  loaded.  n_layers={len(tm.model.layers)}")

    # Build a short prompt.
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

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(tm)

    # Prefill (untraced)
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    logits = _logits(out)[:, -1, :]
    nxt = int(mx.argmax(logits[0]).item())

    # Warmup decode steps (untraced) so Metal caches / JIT'd kernels are warm.
    for _ in range(args.warmup_steps):
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())

    # CAPTURE: window the trace to N decode steps.
    print(f"Capturing {args.capture_steps} decode step(s) to {args.out}")
    mx.metal.start_capture(args.out)
    for _ in range(args.capture_steps):
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt_arr = mx.argmax(logits[0])
        mx.eval(nxt_arr)  # force flush inside the capture window
        nxt = int(nxt_arr.item())
    mx.metal.stop_capture()
    print("Capture complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
