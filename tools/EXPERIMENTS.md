# Decode-perf experiments — running ledger

Every experiment gets a row. Append-only. Record method, result, conclusion.
Reread before starting a new experiment to avoid re-running dead ends.

Models (all oQ6-fp16):
- `g26m` = gemma-4-26B-A4B-it (MoE, 30 layers, n_embd=2816)
- `g31`  = gemma-4-31B-it (dense, 60 layers, n_embd=5376)
- `q35m` = Qwen3.6-35B-A3B (MoE, 40 layers, n_embd=2048)
- `q27`  = Qwen3.6-27B (dense, 64 layers, n_embd=5120)

Baseline tok/s (no patches, decode_overlap, 64 steps, async overlap):
- g26m: 42.22
- g31:  11.10
- q35m: 53.96
- q27:  13.77

## Experiment table

| # | Hypothesis | Patch / change | g26m | g31 | q35m | q27 | Equiv | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | Async-overlap decode loop is faster than naive serial | None — comparison of two decode-loop patterns | serial 37.4 vs overlap 42.1 (+12.6%) | n/a | n/a | n/a | n/a | mlx-lm already uses overlap → omlx benefits via GenerationBatch._step |
| 2 | Per-block mx.compile of post-attn body | `tools/patch_gemma4_compile.py` (gemma4) | +1.7% | 0.0% | n/a | n/a | ✓ | Modest win, MoE only |
| 3 | Per-block mx.compile, qwen3_5 (dense+MoE) | `tools/patch_qwen3_5_compile.py` (qwen3_5 AND qwen3_5_moe) | n/a | n/a | +1.6% | +0.1% | ✓ | Same pattern: MoE wins, dense flat |

## Experiments in progress

(none currently)

## Completed experiments (continued)

| # | Hypothesis | Patch / change | g26m | g31 | q35m | q27 | Equiv | Outcome |
|---|---|---|---|---|---|---|---|---|
| 4 | Extending compile into attention (pre + q_rope + out_proj separate compiles, SDPA stays uncompiled to preserve TurboQuant cache paths) | `tools/patch_gemma4_attn_compile.py` on top of #2 | +1.6% | - | - | - | ✓ | NEUTRAL: same as block-only. The q_rope and out_proj wrappers are too small (1-2 ops each) for compile to help. Confirms block granularity is right. |
| 5 | Cache bound-method refs in `__dict__` to skip __getattr__ | Edit `patched_call` to store input_layernorm + self_attn refs in `self.__dict__` on first call | +1.5% | - | - | - | ✓ | NEUTRAL/slight regression. The __getattr__ overhead is in upstream mlx-vlm but accessing through mlx Module's `_children` dict isn't bypassed by storing in `__dict__` (mlx Module __getattr__ still runs first). Reverted. |
| 6 | Compile gemma4 RoPE (ProportionalRoPE.__call__ — ~8-10 slice/concat ops + mx.fast.rope) | `tools/patch_gemma4_rope_compile.py` on top of #2 | +1.3% | - | - | - | ✓ | NEUTRAL/slight regression. Most "slices" are zero-cost views, not kernel dispatches, so there's no Python frame overhead to fuse. Compile dispatch overhead > savings. |
| 7 | Forced-sync per-section timer to identify expensive GPU regions | `tools/section_time_gemma4.py` (measurement, not optimization) | - | - | - | - | - | **MISLEADING** (not a real perf change anyway): the forced sync at each boundary added ~250µs overhead per section, making everything look ~similar. The "GPU compute floor" conclusion I drew from this was wrong — see experiment #8. |
| 8 | **Gate+up matmul fusion**: HF checkpoint ships gate+up as one fused tensor; mlx-vlm splits on load. Re-fuse at runtime by concatenating weights/scales/biases and doing one matmul, split output. | `tools/patch_gemma4_gateup_fuse.py` — patches SwitchGLU (MoE), gemma4 MLP and Qwen3_5MLP (dense). | **+12.5%** | +1.5% | **+10.7%** | +0.4% | ✓ | **MAJOR WIN on MoE.** Token-identical. The gather_qmm kernel has high per-call fixed overhead (~150µs), and saving one dispatch per layer per token compounds. Dense models get a small win because their MLP matmul itself dominates the dispatch overhead. This invalidates the earlier "98% of floor" claim from exp #7. |
| 9 | **Q/K/V matmul fusion**: same idea applied to attention's q_proj/k_proj/v_proj (all take the same input x). | `tools/patch_qkv_fuse.py` (REMOVED — buggy) | +13.6%* | - | - | - | ✗ | PROMISING but BUGGY. Showed +13.6% (≈ +2.5% on top of gate+up) but output DIVERGED at token 0 (garbage repeated token). Bug not found by inspection in the session budget — gemma4 attention has per-layer-varying config (k_eq_v full-attn layers use num_global_key_value_heads=2 + possibly global_head_dim; sliding layers differ; mixed 6/8-bit quant per layer). Patch deleted; documented here as a real future lever worth ~2-3% if the Q/K/V split bug is fixed. Suggest: write a numerical isolation test (fused matmul output vs separate q/k/v on random input) to localize the bug fast. |

\* exp #9 delta includes gate+up; QKV's own marginal contribution ≈ +2.5%. Output incorrect.

## Cumulative profile of patched gemma 4 26B-A4B-it (128 steps)
After applying experiment #2 (block compile), cProfile shows where Python
overhead still lives (sorted by tottime, items > 0.001s):

  decode_loop:                3.187s (.item() sync blocked)
  patched_call:               0.074s  ← MY patch's orchestration cost
  attention __call__:         0.041s
  cache._update_in_place:     0.029s
  gemma4 rope_utils:          0.029s
  QuantizedLinear.__call__:   0.018s
  RMSNorm:                    0.010s
  Module __getattr__:         0.007s  (was 0.017 before, patch reduced)
  SDPA (mlx_lm):              0.006s
  cache.update_and_fetch:     0.005s
  outer model.__call__:       0.005s

Total visible non-blocked Python: ~0.25s out of 2.97s wall = 8.4%.
Even eliminating ALL of it would be max 8.4% improvement.
Most of it is in attention (compute-bound subcalls) and cache (the
kvcache `[..., prev:offset, :] = keys` slice assignment is GPU work,
attributed to Python frames here).

## Experiment ideas (queue)

- **idea-A** — Hoist `nn.Module.__getattr__` cost via __dict__ cache.
  **Tried (exp #5): neutral / slight regression.** Reason: mlx Module's
  __getattr__ goes through `_children`, not __dict__, so the cache doesn't
  bypass it.
- **idea-B** — Cache `mx.array([[nxt]])` allocation per decode step.
  Today every step allocates a new (1,1) int32 array.
  Status: low-value (per-step cost ~1µs, negligible compared to forward).
- **idea-C** — Lower scheduler.py Python overhead in the decode hot loop.
  Look at `_step_*` paths to see if there's anything Python-heavy.
  Status: deferred — needs end-to-end measurement through scheduler, not
  direct model calls.
- **idea-D** — Per-expert MoE dispatch fusion. The Experts class dispatches
  many small gather_mm calls. If we can batch them...
  Status: partly captured by experiment #2 (block compile wraps
  Experts call, so its dispatches are inside the compile graph).
- **idea-E** — Look for any `.item()` syncs in the inference path beyond
  the known grammar one. Each unnecessary sync costs ~3ms.
  Status: deferred — needs scheduler audit.
- **idea-F** — KV cache `_update_in_place` was 25ms in cProfile across 3200
  calls. Check if there's a faster path (preallocated cache, no resize).
  Status: cache already pre-allocates in 256-step chunks. The 25ms is
  the in-place GPU slice assignment, attributed to a Python frame in cProfile.
  Probably not improvable without a custom compile-friendly cache class.
- **idea-G (NEW)** — Inline cache update into compiled body via custom
  KVCache that exposes functional get/set. Compile would include SDPA.
  Status: multi-day fork project — out of scope for this session.

## Dead-end experiments

- **dead-1** — Pre-stacking steering projections into (units, strengths)
  arrays. Tried in steering perf work. Sequential indexing was a small
  regression vs Python tuple iteration. Pattern: array indexing has its
  own dispatch cost.

## Method notes

- Use `tools/bench_gemma4_compile.py <model> --arch {gemma4|qwen3_5}`
  for A/B comparison with token equivalence check.
- 3-run minimum, take min. Most runs are very stable (±0.5%).
- `dangerouslyDisableSandbox: true` required for any Metal-touching code.
- Models with MTP weights need `_drop_mtp_weights_on_load` (already in profile_decode.py).
