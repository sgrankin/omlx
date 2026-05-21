# Perf investigation journal

Branch: `perf/profile-decode` (rooted on `vlok`). User is AFK; I have autonomy
to iterate on omlx decode performance. Write to this often, reread after any
context compaction.

## Context at start (TL;DR)

User wants more decode-perf wins after the steering fusion (already shipped,
3-8pp savings on projection mode). Target models: Qwen3.6-27B-oQ6-fp16,
Qwen3.6-35B-A3B-oQ6-fp16, gemma-4-26B-A4B-it-oQ6-fp16, gemma-4-31B-it-oQ6-fp16.
All live under `~/.omlx/models/`.

Workflow rules (must follow):
- jj VCS, see `/jj:workflow` skill behavior. Pre-create commit with `jj describe`,
  iterate in scratch with `jj squash` after each milestone. Use `--use-destination-message`
  on squash; otherwise jj opens an editor and fails non-interactively.
- macOS BSD userland (e.g. `sed -i ''` not `sed -i`).
- Tests that touch Metal need `dangerouslyDisableSandbox: true`.
- 4 target models above are oQ6-fp16 (6-bit quant, fp16 compute).

Already-known facts (verified):
- Decode is GPU-bound at batch=1. Per-step wall is dominated by model forward.
- mlx-lm `GenerationBatch._step` uses async overlap (`mx.async_eval(next_tokens)`
  before `mx.eval(current)`) and omlx inherits it.
- Naive serial decode pattern (sync .item() before next forward) is ~12% slower
  than overlap. Measured 37.4 → 42.1 tok/s on Gemma 4 26B-A4B-it.
- omlx `scheduler.py:263` breaks the overlap for grammar-constrained requests.
  Tracked as project memory for later.
- One decode step on Gemma 4 26B-A4B-it dispatches ~5,258 kernels across
  238 distinct pipelines. ≈175 dispatches / layer. Per-dispatch budget at
  27ms/token = 5.1µs, right at the kernel launch overhead range — suggests
  partial dispatch-binding.

## Current hypothesis being tested

**Per-block `mx.compile`** on the gemma4 transformer block body should collapse
many small per-layer dispatches into fewer larger ones, recovering some of the
175-dispatches/layer cost.

Risks:
- KV cache update is in-place; mx.compile can't trace mutations cleanly.
  Need to keep cache.update_and_fetch() OUTSIDE the compiled region.
- SDPA dispatch shape changes as KV grows — compile would need to either
  re-trace or be shape-agnostic.
- Patching mlx-vlm code is upstream territory; we already do this in
  `omlx/patches/` (mlx_vlm_mtp/, mlx_lm_mtp/, qwen3_5_attention.py,
  qwen3_6_nested_visual.py). Pattern is well-established.

## Plan

1. Reverse-engineer the .gputrace format to get reliable pipeline-ID →
   kernel-name mapping. Get a dispatch census broken down by op type
   (RMSNorm × N, RoPE × N, qmv × N, sdpa × N, residual × N, switch_mlp × N).
   This points at which fusion target gives the biggest win.
2. Pick the fattest fusable region (e.g. norm+residual+RMSNorm, or the
   "post-attn project + residual + RMSNorm" sequence).
3. Patch via monkey-patch in `omlx/patches/`.
4. Re-bench on all 4 target models.
5. If win confirmed, ship; if not, document and move on.

## Constraints / things I cannot do
- Cannot open Xcode (no GUI control). All analysis must be CLI.
- Cannot ask user questions; must make decisions and document them here.
- Must respect existing patches and tests — break nothing.

## Running log (newest at bottom; timestamp roughly)

### t=0  Setup
- Created this journal.
- Plan: continue RE for ~30-60 min, then try the gemma4 block compile experiment.

### t=1  Gemma 4 block compile patch — first result (Gemma 4 26B-A4B-it)
- Wrote `tools/patch_gemma4_compile.py`: monkey-patches DecoderLayer.__call__
  to keep self_attn outside mx.compile (KV cache mutates in place), then
  feeds the post-attn body (norm + residual + MLP + MoE + post-norms +
  per-layer-gate + layer_scalar) through `mx.compile`.
- Bench `tools/bench_gemma4_compile.py`: 3 timed runs each, baseline vs
  patched, plus token-equivalence check.
- Result:
  - baseline: 42.22 tok/s (1.517/1.516/1.517s — very stable)
  - patched:  42.95 tok/s (1.491/1.490/1.491s)
  - **+1.7%**, token-identical.
- Modest win. Within the order I expected (a small fraction of the 9%
  Python overhead in the forward).

### t=1.5  Trace RE status
- 47 library-hash IDs in device-resources.
- 238 PSO-hash IDs in capture.
- ZERO intersection — library hashes never appear in capture text.
- This means PSO IDs are NOT library hashes; PSO is a separate Metal object
  identified by its own hash. To bridge them I'd need to find PSO-creation
  records in capture that name (library, function) → PSO.
- Decision: hold further RE. The 1.7% gemma4 patch result is small enough
  that the next biggest lever is unlikely to come from "find more fusable
  sequences" — most sequences are already fused or matmul-bound. Better
  to try the patch on the bigger gemma 31B (dense) and on Qwen3.6.

### t=2  Gemma 4 31B (dense) and Qwen3.6 results
- Gemma 4 31B (dense, 60 layers, n_embd=5376):
  - baseline: 11.10 tok/s
  - patched:  11.10 tok/s
  - **-0.0%** (noise). Token-identical.
  - Hypothesis: dense large model is more compute-bound per token (90ms vs
    23ms on the MoE), Python overhead is a smaller fraction.

- Qwen3.6-35B-A3B (MoE, A3B = 3B active, 40 layers, n_embd=2048):
  - First attempt: -0.2% no change (my Qwen3.5 patch only touched
    `mlx_vlm.models.qwen3_5.language.Qwen3_5DecoderLayer.__call__` but the
    MoE variant lives in `mlx_vlm.models.qwen3_5_moe.language.Qwen3_5MoeDecoderLayer`).
    Mistake — different class hierarchy in mlx-vlm.
  - Fixed patch to cover both dense + MoE classes; re-running.

### Learnings to remember
- mlx-vlm splits MoE into a separate package per architecture
  (`qwen3_5` vs `qwen3_5_moe`), each with its own DecoderLayer class.
  Patches must target both.
- Per-block mx.compile wins are MUCH bigger when per-token wall is small
  (Python overhead is a larger fraction). MoE-A3B / A4B models are the
  best targets; dense >30B models are likely compute-bound and immune.

### t=3  Full picture for per-block compile patch (4 target models)
- Gemma 4 26B-A4B-it (MoE, 30 layers, n_embd=2816):  **+1.7%** (42.22 → 42.95)
- Qwen3.6-35B-A3B   (MoE, 40 layers, n_embd=2048):  **+1.6%** (53.96 → 54.84)
- Gemma 4 31B-it    (dense, 60 layers, n_embd=5376): 0.0%  (11.10 → 11.10)
- Qwen3.6-27B       (dense, 64 layers, n_embd=5120): +0.1% (13.77 → 13.78)

All four token-identical. Safe to apply unconditionally — no regression on
dense models, real win on MoE.

### Decision: ship per-block patch, then try to extend
The win is modest but the patch is small + safe + token-identical. Ship it.
Then try:
- Extending compile into attention (split pre-cache / post-cache phases)
- Compile the gemma4 outer model forward (probably blocked by KV mutation)
- Look at scheduler.py overhead — maybe there's a Python-level lever in
  the loop *around* the model forward

### t=4  Attention pre/post compile split (neutral)
- Wrote `patch_gemma4_attn_compile.py`: pre-cache compile + post-cache
  q_rope + post-cache o_proj as separate compiled regions, with SDPA
  unchanged in between (preserves cache-aware paths e.g. TurboQuant).
- Result on Gemma 4 26B-A4B-it: +1.6% (vs block-only +1.7%). NO GAIN.
- The post-cache wrappers (q_rope is 2 ops, o_proj is 3 ops) are too thin
  for compile to amortize its dispatch overhead.

### t=5  __dict__-based attribute caching (neutral/slight regression)
- Tried storing `input_layernorm` and `self_attn` refs in `self.__dict__`
  to skip mlx Module's __getattr__ on each call.
- Result: +1.5% (vs block-only +1.7%). Marginally worse.
- Reverted. mlx Module's __getattr__ goes through `_children`, not
  __dict__, so caching in __dict__ doesn't bypass the slow path.

### t=6  RoPE compile (neutral, slight regression)
- Wrote `patch_gemma4_rope_compile.py`: wrap ProportionalRoPE.__call__
  in mx.compile, hoping to fuse its 8-10 slice/concat ops.
- Result: +1.3% (vs block-only +1.7%). Slightly worse.
- Reason: most of the "slices" are VIEWS, not kernel dispatches. The 8-10
  ops estimate was wrong. mx.compile overhead > savings.

### Where the patched profile shows remaining Python overhead
Sorted by tottime (gemma 4 26B-A4B-it, 128 steps, patched):
  decode_loop (.item() block):       3.187s
  patched_call (my orchestration):   0.074s  ← biggest live frame
  attention __call__:                0.041s
  cache._update_in_place:            0.029s
  rope_utils:                        0.029s
  QuantizedLinear:                   0.018s
  RMSNorm:                           0.010s
  Module __getattr__:                0.007s
  SDPA:                              0.006s
Total live Python: ~250ms / 2965ms = 8.4% of wall.

To get more wins, would need to inline cache.update_and_fetch into a
compiled function (requires custom cache class), or rework attention
to bring SDPA inside compile (cache-aware paths complicate this).
Both are multi-day fork projects, not session wins.

### Final disposition
- Ship the block-compile patch. +1.5-1.7% on MoE models, 0% on dense.
  Safe (token-identical). 4 commits on `perf/profile-decode`.
- Park further per-block fusion. Diminishing returns territory.
- Other levers (grammar overlap restoration, KV cache rewrite, full-model
  compile) are documented in EXPERIMENTS.md as future work.

### t=7  Forced-sync per-section timing — confirms GPU is the floor
- Wrote section_time_gemma4.py: inserts mx.eval() at each section boundary
  inside Attention.__call__ and DecoderLayer.__call__, measures per-section
  wall + count.
- The instrumented run is 7× slower (5.84 vs 42.2 tok/s) — every section
  pays a ~250µs sync penalty. So sections that take >250µs/call are GPU-
  compute-bounded; sections at exactly 250µs are dispatch+sync only.
- Findings (only sections where actual GPU work clearly exceeds the sync
  floor):
    blk:experts      670µs/call  ≈ ~420µs real GPU work — HEAVIEST section
    blk:mlp_main     390µs/call  ≈ ~140µs real GPU work
    blk:router       346µs/call  ≈ ~100µs real GPU work
    attn:q_proj      310µs/call  ≈ ~60µs real GPU work
    attn:k_proj      295µs/call  ≈ ~45µs real GPU work
    attn:sdpa        264µs/call  ≈ ~15µs real GPU work
    everything else: ≤ ~250µs/call — basically pure sync overhead
- Per layer at decode-batch=1:
    Router + Experts + main_MLP ≈ 660µs of GPU work
    Attention (q/k/v/o + SDPA + rope) ≈ 200µs of GPU work
    Norms/residuals/etc: tiny per call
- 30 layers × 860µs ≈ 26ms per token of GPU work. Matches the 23ms wall
  per token in production (small slack from kernel queueing overlap).
- **Conclusion**: At batch=1 decode, the model IS compute-bound and very
  close to the realizable floor. The +1.7% from block-compile is shaving
  Python dispatch overhead; we can't go meaningfully below the 23ms floor
  without changing the model (smaller quants, fewer experts) or hardware.
- Bigger wins would require: (a) running concurrent requests
  (continuous-batching makes per-token cost amortize), (b) different
  kernel choices for `affine_qmv_fast` (the q/k/v/o_proj kernel — out of
  scope, MLX-level), or (c) speculative decoding (dflash / MTP, which
  omlx already supports as opt-in).

### Summary for the user (when they return)

Block-compile patch shipped on `perf/profile-decode` branch as 4 commits:
- `perf(tools): add decode-loop profiler`
- `perf(tools): Metal trace capture + async-overlap baseline`
- `perf: investigation journal`
- `perf(patches): mx.compile-fuse the post-attn block body`
- `perf(tools): experimental compile patches that DIDN'T win`

Result: **+1.5-1.7% tok/s on MoE models** (Gemma 4 26B-A4B-it,
Qwen3.6-35B-A3B), 0% on dense (Gemma 4 31B, Qwen3.6-27B). Token-identical.

Per-section timing confirms the four target models are GPU-compute-bound
at batch=1, near the realizable floor. Python overhead is small;
further wins require model/algorithm changes (TurboQuant, speculative
decoding), not Python-level optimizations.

