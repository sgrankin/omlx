# HF downloader — stall investigation

User-observed symptom: a queued HF model download in the admin UI stops
making progress (no bytes written for minutes), but as soon as the user
sends Ctrl-C to the server, the stalled threads suddenly start writing
again before shutdown completes. Related: when calibration data
(`omlx/oq.py:_load_hf_calibration` → `datasets.load_dataset`) hits a
mirror returning 502s, cancelling the parent oQ task does not unblock
the underlying thread for the full retry budget (~60–90s).

This file captures the architecture and concrete suspects so a follow-up
commit on this branch can fix the underlying issue rather than the
symptom.

## Architecture

The admin HF downloader (`omlx/admin/hf_downloader.py`) uses an
asyncio + threadpool model:

- `start_download()` records a task and queues async work.
- `_run_download()` runs as an asyncio task. It acquires
  `self._download_sem` (single-flight), calls `api.model_info()` with a
  per-request timeout (no overall timeout), then runs
  `huggingface_hub.snapshot_download` twice via `asyncio.to_thread()`:
  once as a `dry_run` (size estimate), then for real.
- `_poll_progress()` runs as a sibling async task. Every 2s it computes
  directory size + latest mtime via `Path.rglob("*")` and reports
  progress + a stall flag (300s without mtime advancement).
- Cancellation: `cancel_download()` sets `self._cancelled` and cancels
  the asyncio wrappers. The actual thread cooperatively notices the
  flag in the progress callback path. Force-cancel happens after a
  timeout but the thread can still be blocked inside huggingface_hub.

There are **no** explicit progress callbacks registered with HF — all
progress comes from the filesystem poller.

The `fix/hf-downloader-polling` topic already addresses one related bug
(start the poller before `dry_run` so the UI doesn't freeze during
size estimate). It does not change cancellation or timeout semantics.

## Concrete suspects

1. **`snapshot_download` has no overall timeout.** Both the dry_run
   and the real download run inside `asyncio.to_thread` with no
   wall-clock guard. `etag_timeout=30` only bounds individual HEAD
   requests. A mirror that accepts the TCP connection and then sits
   on the request can hold the worker thread indefinitely.

2. **The progress poller swallows exceptions silently.** If
   `_get_dir_size()` / `_get_latest_mtime()` hit a `FileNotFoundError`
   (file vanished mid-iteration during `rglob`), the poll task can
   die. The outer task only handles `asyncio.CancelledError`. Once
   the poller dies, stall detection stops firing — explaining why a
   visibly-stalled download never trips the 300s flag.

3. **`Path.rglob("*")` on a partially-downloaded model can be slow.**
   For a model with thousands of partial files (HF's content-addressed
   blob store leaves many `.lock` / `.incomplete` artifacts), the
   rglob scan can take seconds and starves the asyncio loop.

4. **Cancel flag is checked after `snapshot_download` returns,** not
   during. A cancelled download stays in `downloading` state until the
   thread eventually unwinds.

5. **`datasets.load_dataset()` (in `omlx/oq.py:_load_hf_calibration`)
   has no cooperative cancel hook at all** — its retry loop is owned
   by `huggingface_hub` and ignores the oQ progress callback's
   `_QuantCancelled`.

6. **Why does Ctrl-C kick threads back to life?** Likely because the
   server's signal handler triggers `loop.stop()`, which cascades into
   the threadpool executor's shutdown path. That shutdown sends
   pending futures' results, unblocks any thread waiting on a future
   completion, and the HF library's own atexit / cleanup paths flush
   buffered writes before the process dies. The writes were already
   queued in memory; the network read had completed; only the
   shutdown sequence drained them.

## Open questions

- What `huggingface_hub` version are we actually running? `pyproject.toml`
  pins it; check whether the version has the threadpool deadlock
  reported upstream around the 0.24–0.26 transition.
- Is the user pointing at `hf.tail172cc.ts.net` (the tailscale-fronted
  mirror)? That endpoint returns 502s during the failures we observed.
  How does HF behave when the endpoint accepts TCP but stalls the
  HTTP response?
- Reproducer: does the issue happen on a 1 GB model with a healthy
  mirror, or only on multi-GB downloads / a flaky mirror?
- Trace: `lsof` / `fs_usage` / `sample` on the stalled python process
  to confirm whether threads are blocked in `read()` (network) vs.
  `pthread_cond_wait` (Python lock) vs. some HF-internal lock.

## Likely fix shape (sketch — verify with reproducer first)

- Wrap `snapshot_download` in `asyncio.wait_for(asyncio.to_thread(...))`
  with a generous overall timeout (e.g. 30 min for size > 50 GB) so a
  truly hung thread can be force-killed and surfaced.
- Add a per-file progress callback (HF supports one) so progress
  updates don't depend on rglob scans, and so cancellation has a
  cheap polling point inside the thread.
- Wrap `_get_dir_size` / `_get_latest_mtime` per-file iteration in a
  try/except so vanished files don't kill the poller.
- For `_load_hf_calibration`, set `download_config.max_retries=0` (or
  similar) so a 502 fails fast instead of retrying for ~90s. The
  built-in calibration data is the primary path; the HF fallback is
  only there for correctness on tokenizers that produce too few
  tokens from the built-in corpus, and even there a long retry
  budget doesn't help.
