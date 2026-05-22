# SPDX-License-Identifier: Apache-2.0
"""Per-section timing inside gemma4 DecoderLayer + Attention.

Monkey-patches Attention.__call__ and DecoderLayer.__call__ to time each
sub-section (q_proj, k_proj, v_proj, norms, ropes, cache_update, sdpa,
o_proj, residual+norms+mlp). mx.eval at each boundary forces GPU work to
complete so the measurement reflects actual wall time of that section.

This is informational, not perf-improving — it tells us where the
remaining 8.4% Python overhead in the patched forward actually lives.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_decode import _load_model, _text_model, _logits


_timings: defaultdict = defaultdict(float)
_counts: defaultdict = defaultdict(int)


def _t(name, *evals):
    if evals:
        mx.eval(*evals)
    now = time.perf_counter()
    _timings[name] = _timings.get(name, 0.0) + (now - _timings["__last__"])
    _counts[name] += 1
    _timings["__last__"] = now


def _t_start():
    _timings["__last__"] = time.perf_counter()


def patch_for_timing():
    from mlx_vlm.models.gemma4 import language as g4

    Attention_orig = g4.Attention.__call__

    def attn_timed(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        _t_start()
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        _t("attn:q_proj", queries)
        queries = self.q_norm(queries)
        _t("attn:q_norm", queries)

        if shared_kv is not None:
            keys, values = shared_kv
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            _t("attn:k_proj", keys)
            if self.use_k_eq_v:
                values = keys
            else:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
                _t("attn:v_proj", values)
            offset = mx.array(cache.offset) if cache is not None else 0
            keys = self.k_norm(keys)
            _t("attn:k_norm", keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)
            _t("attn:k_rope", keys)
            values = self.v_norm(values)
            _t("attn:v_norm", values)
            values = values.transpose(0, 2, 1, 3)

            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
                _t("attn:cache_update", keys, values)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        _t("attn:q_rope", queries)

        from mlx_vlm.models.base import scaled_dot_product_attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        _t("attn:sdpa", output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(output)
        _t("attn:o_proj", out)

        return out, (keys, values), offset

    g4.Attention.__call__ = attn_timed

    DL_orig = g4.DecoderLayer.__call__

    def dl_timed(self, x, mask=None, cache=None, per_layer_input=None,
                 shared_kv=None, offset=None):
        _t_start()
        residual = x
        h = self.input_layernorm(x)
        _t("blk:input_ln", h)
        h, shared_kv, offset = self.self_attn(
            h, mask, cache, shared_kv=shared_kv, offset=offset
        )
        _t("blk:self_attn", h)
        h = self.post_attention_layernorm(h)
        _t("blk:post_attn_ln", h)
        h = residual + h
        residual = h
        _t("blk:residual1", h)
        if self.enable_moe:
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)
            _t("blk:mlp_main", h1)
            top_k_indices, top_k_weights = self.router(h)
            _t("blk:router", top_k_indices, top_k_weights)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)
            _t("blk:experts", h2)
            h = h1 + h2
            _t("blk:moe_combine", h)
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)
            _t("blk:mlp_main", h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        _t("blk:residual2", h)
        if (
            self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            import mlx.nn as nn
            residual3 = h
            gate = self.per_layer_input_gate(h)
            gate = nn.gelu_approx(gate)
            gate = mx.multiply(gate, per_layer_input)
            gate = self.per_layer_projection(gate)
            gate = self.post_per_layer_input_norm(gate)
            h = residual3 + gate
            _t("blk:plgate", h)
        if self.layer_scalar is not None:
            h = h * self.layer_scalar
            _t("blk:scalar", h)
        return h, shared_kv, offset

    g4.DecoderLayer.__call__ = dl_timed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="gemma-4-26B-A4B-it-oQ6-fp16")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    path = (
        args.model if Path(args.model).is_absolute()
        else str(Path.home() / ".omlx" / "models" / args.model)
    )
    print(f"Loading {path}")
    model, _ = _load_model(path)
    tm = _text_model(model)
    n_layers = len(tm.model.layers)
    print(f"  n_layers={n_layers}")

    patch_for_timing()

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(tm)

    prompt_ids = list(range(20))
    out = tm(mx.array(prompt_ids)[None], cache=cache)
    logits = _logits(out)[:, -1, :]
    nxt = int(mx.argmax(logits[0]).item())

    # Warmup
    for _ in range(args.warmup):
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())

    # Reset timers
    _timings.clear()
    _counts.clear()

    # Timed decode
    t0 = time.perf_counter()
    for _ in range(args.steps):
        out = tm(mx.array([[nxt]]), cache=cache)
        logits = _logits(out)[:, -1, :]
        nxt = int(mx.argmax(logits[0]).item())
    wall = time.perf_counter() - t0

    print(f"\nWall: {wall:.3f}s for {args.steps} steps => {args.steps/wall:.2f} tok/s")
    print(f"Note: section timing FORCES sync at each section, slowing things "
          f"by ~{(wall * args.steps - sum(_timings.values()))*1000:.0f}ms\n")

    print(f"\n{'section':<22s}  {'calls':>7s}  {'total ms':>10s}  "
          f"{'per-call us':>13s}  {'per-token ms':>13s}")
    print("-" * 80)
    items = sorted(((k, v) for k, v in _timings.items() if not k.startswith("__")),
                   key=lambda kv: -kv[1])
    for name, t in items:
        c = _counts[name]
        per_call_us = 1e6 * t / c
        per_token_ms = 1000.0 * t / args.steps
        print(f"  {name:<22s}  {c:>7d}  {t * 1000:>10.2f}  "
              f"{per_call_us:>13.2f}  {per_token_ms:>13.3f}")
    print()
    total = sum(t for k, t in _timings.items() if not k.startswith("__"))
    print(f"Total measured: {total * 1000:.0f}ms ({total / wall * 100:.0f}% of wall)")


if __name__ == "__main__":
    main()
