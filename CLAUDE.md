# oMLX — developer notes

LLM inference server for Apple Silicon. OpenAI/Anthropic-compatible HTTP API over `mlx-lm` / `mlx-vlm` / `mlx-embeddings`, with continuous batching, paged KV cache, and a hot (RAM) + cold (SSD) tier. FastAPI server + admin dashboard + menubar app.

This checkout is a **local fork** — upstream is `jundot/omlx`. Feel free to diverge from upstream conventions when it makes local work easier, but flag anything that would be awkward to upstream later.

## Environment

- macOS, Apple Silicon (M1). Python ≥ 3.10.
- VCS is **jj** (colocated with git). Always load `/jj:workflow` before VCS ops.
- Install deps:
  ```
  uv sync --extra={grammar,mcp,modelscope}
  ```
  (Audio stack lives in the `audio` extra; skip unless needed — pulls espeak/spacy.)
- Run the server:
  ```
  uv run omlx serve
  ```
- Several `mlx-*` deps are pinned to specific git commits in `pyproject.toml`. If you bump one, check the comment above its line — we pin for specific fixes (Gemma 4 tool parser, TQ race fix, dflash temperature sampling, etc.).

## Layout

```
omlx/
  server.py / cli.py          entry points
  scheduler.py, engine_core.py, engine_pool.py
  engine/                     per-modality engines (batched, vlm, dflash, embedding, reranker, audio)
  models/                     model wrappers (llm, vlm, embedding, reranker, xlm_roberta)
  cache/                      paged + prefix + SSD cache
  adapter/                    OpenAI / Anthropic / Harmony / Gemma4 request+output adaptation
  api/                        FastAPI routes + schemas
  admin/                      web UI (templates, static, i18n)
  mcp/                        Model Context Protocol client/executor
  patches/                    monkey-patches into mlx-lm / mlx-vlm internals
  oq.py, turboquant_kv.py     on-the-fly quantization (oQ / TurboQuant)
  integrations/               OpenClaw, OpenCode, Codex, Pi wiring
tests/                        pytest suite, mirrors module layout
docs/experimental/            design notes (e.g. dflash integration)
packaging/                    PyObjC menubar app
```

## Testing

```
pytest -m "not slow"                # fast path; default during dev
pytest tests/test_<module>.py -v    # single module
pytest -m slow                      # requires real model files
pytest -m integration               # requires running server
```

Test naming mirrors source: `omlx/<x>.py` → `tests/test_<x>.py`.

**Metal + sandbox**: tests that touch Metal (anything loading a real model, or pulling MLX ops onto the GPU) crash Python when run inside the Claude Code sandbox. Run those with `dangerouslyDisableSandbox: true`, or restrict to pure-Python tests (`pytest -m "not slow"` usually stays off Metal, but not guaranteed — if a fast-marked test imports a model, it'll still crash).

## Things worth knowing before editing

- **`omlx/patches/`** is active monkey-patching of upstream mlx-lm / mlx-vlm. When upstream changes under us (we pin to commits, not releases), these are the first suspects for regressions.
- **Gemma 4** has a lot of special-case code: `adapter/gemma4.py`, `adapter/output_parser.py`, tool-parser detection in the text engine, plus SpecPrefill compatibility shims. Recent work in this tree has been stabilizing malformed-channel-marker handling and think-tag trimming — check recent jj log before touching.
- **dflash** (`engine/dflash.py`) is the block-diffusion speculative decoding engine — newer, less battle-tested than `engine/batched.py`. See `docs/experimental/dflash_mlx_integration.md`.
- **Quantization**: `oq.py` (oQ, uses `oq_calibration_data.json`) and `turboquant_kv.py` are the local paths. fp16 is natively fast on M1 — for local quantizing experiments, fp16 is often the right target rather than int4/int8.
- **Transformers is pinned `<5.4.0`** because 5.4 made `Qwen2VLImageProcessor` hard-require torch (breaks VLM). Don't casually widen this.
- **License header**: every source file gets `# SPDX-License-Identifier: Apache-2.0`.
- **`Scheduler._extract_cache_states`** output is frozen-safe: list state (ArraysCache/CacheList sub-lists) is shallow-copied at capture, so callers can hold the returned dict across prefill/decode steps without seeing in-place mutation. Boundary-snapshot storage depends on this. Guarded by `tests/test_scheduler.py::TestExtractCacheStatesFreeze`.

## Style

- ruff + black, line length 88. `ruff` config lives in `pyproject.toml` (`E,F,W,I,N,UP,B,SIM`, ignores `E501`, `B905`).
- mypy is configured but not strict — don't add type-ignore comments just to silence it; fix the type if it's real.
