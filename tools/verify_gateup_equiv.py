# SPDX-License-Identifier: Apache-2.0
"""Rigorous token-equivalence check for the gate+up fusion patch.

The bench's built-in check uses a tiny synthetic prompt [1..10]. The
fused matmul can differ from the unfused by ~1 ulp fp16 at prefill
(L>1) due to different gemm kernel tiling. Whether that flips a token
is input-dependent — so verify with a realistic chat prompt and a long
decode run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model, _logits


def decode(model, prompt_ids, n):
    from mlx_lm.models.cache import make_prompt_cache
    tm = _text_model(model)
    cache = make_prompt_cache(tm)
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    logits = _logits(out)[:, -1, :]
    nxt = int(mx.argmax(logits[0]).item())
    toks = []
    for _ in range(n):
        toks.append(nxt)
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())
    return toks


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26B-A4B-it-oQ6-fp16"
    arch = sys.argv[2] if len(sys.argv) > 2 else "gemma4"
    n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    path = str(Path.home() / ".omlx" / "models" / model_name)
    print(f"Loading {path}")
    model, tokenizer = _load_model(path)

    # A realistic, longer prompt — exercises prefill at L>1.
    prompt = (
        "Explain, in detail, how a B-tree database index works and why it "
        "is preferred over a hash index for range queries. Cover node "
        "structure, splitting, and the height/fanout tradeoff."
    )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = list(tokenizer.encode(templated))
        except Exception:
            prompt_ids = list(tokenizer.encode(prompt))
    else:
        prompt_ids = list(tokenizer.encode(prompt))
    print(f"  prompt: {len(prompt_ids)} tokens; decoding {n_tokens}")

    print("Baseline decode...")
    base = decode(model, prompt_ids, n_tokens)

    print("Applying gate+up fusion...")
    from tools.patch_gemma4_gateup_fuse import apply_gemma4_gateup_fuse_patch
    ok = apply_gemma4_gateup_fuse_patch(model)
    print(f"  applied: {ok}")

    print("Patched decode...")
    patched = decode(model, prompt_ids, n_tokens)

    if base == patched:
        print(f"\n✓ gate+up: token-identical over {n_tokens} tokens "
              f"on a realistic {len(prompt_ids)}-token prompt")
        return 0
    diff = next((i for i in range(min(len(base), len(patched)))
                 if base[i] != patched[i]), None)
    print(f"\n✗ gate+up DIVERGES at token {diff} / {n_tokens}")
    print(f"  baseline[{max(0,diff-2)}:{diff+3}] = {base[max(0,diff-2):diff+3]}")
    print(f"  patched [{max(0,diff-2)}:{diff+3}] = {patched[max(0,diff-2):diff+3]}")
    # How many tokens match before divergence — a measure of severity
    print(f"  {diff}/{n_tokens} tokens matched before divergence")
    return 1


if __name__ == "__main__":
    sys.exit(main())
