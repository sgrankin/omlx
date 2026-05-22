# HF downloader — stall investigation

User-observed symptom: a queued HF model download in the admin UI stops
making progress (no bytes written for minutes), but as soon as the user
sends Ctrl-C to the server, the stalled threads suddenly start writing
again before shutdown completes. Related: when calibration data
(`omlx/oq.py:_load_hf_calibration` → `datasets.load_dataset`) hits a
mirror returning 502s, cancelling the parent oQ task does not unblock
the underlying thread for the full retry budget (~60-90s).

This file captures the architecture and the current leading suspect so a
follow-up investigation can confirm it with a real reproducer.

## Architecture

The admin HF downloader (`omlx/admin/hf_downloader.py`) uses an
asyncio + threadpool model:

- `start_download()` records a task and queues async work.
- `_run_download()` runs as an asyncio task. It acquires
  `self._download_sem` (single-flight), calls `api.model_info()` with a
  per-request timeout, starts the progress poller, then runs
  `huggingface_hub.snapshot_download` twice via `asyncio.to_thread()`:
  once as a bounded (`asyncio.wait_for`, 120s) `dry_run` for the size
  estimate, then for real (unbounded — see below).
- `_poll_progress()` runs as a sibling async task. Every 2s it computes
  directory size + latest mtime via `Path.rglob("*")` (now off the event
  loop via `asyncio.to_thread`) and reports progress + a stall flag
  (300s without mtime advancement).
- Cancellation: `cancel_download()` sets `self._cancelled` and calls
  `abort_xet_session()`. The tqdm callback path (non-xet transfers)
  cooperatively raises `_DownloadCancelled` on the next chunk once the
  flag is set.

Already handled by upstream / this fork, not open suspects:

- `OSError` during the progress scan (`_get_dir_size` /
  `_get_latest_mtime`) is caught per-file, so a vanished file mid-`rglob`
  (`FileNotFoundError` is an `OSError`) can't kill the poller.
- `_make_cancellable_tqdm` + `abort_xet_session()` (called on cancel,
  stall, and shutdown) is upstream's + this fork's mechanism for reaping
  a wedged transfer thread; the earlier "cancel flag only checked after
  `snapshot_download` returns" theory is stale.
- The "Ctrl-C kicks stalled threads back to life" theory (signal handler
  cascading into threadpool shutdown flushing buffered writes) was never
  confirmed and isn't the current lead — the xet theory below better
  explains a stall that produces zero bytes rather than buffered-but-
  unflushed ones.

## Leading suspect: hf-xet ignores the HTTP timeout

`huggingface_hub` 1.x downloads large files through `hf-xet`, a Rust
extension that reconstructs blobs from content-addressed chunks fetched
from a separate **CAS (content-addressed storage) endpoint**, not the
Hub API endpoint. Two things compound:

1. **`hf-xet`'s Rust reconstructor does not honor
   `HF_HUB_DOWNLOAD_TIMEOUT`.** That env var (and the library's
   `etag_timeout`) bounds requests made through `huggingface_hub`'s own
   Python `requests`/`httpx` layer. The xet path's chunk-fetch loop is
   a separate Rust HTTP client with its own (much longer, or absent)
   timeout behavior. A stalled xet request doesn't trip any of the
   timeouts this codebase sets.

2. **A mirror that proxies the Hub API but not the CAS endpoint** would
   explain the exact symptom: `model_info()`, the dry run's metadata
   fetch, and the initial file listing all go through the (proxied) Hub
   API and succeed. The actual chunk transfer then tries to reach the
   xet CAS endpoint — either a different hostname the mirror doesn't
   proxy, or one it proxies but the backend never answers. TCP connects
   (mirror accepts the connection), the HTTP request is sent, and then
   nothing: no error, no timeout, no bytes. That's a silent stall, not a
   failure — which is why nothing in `_run_download`'s exception
   handling ever fires.

This reframes the earlier "no overall timeout on `snapshot_download`"
suspect: the *real* download call is intentionally left unbounded (see
the comment in `_run_download`) because `asyncio.wait_for` can't kill
the underlying thread — wrapping it would just leak a second wedged
thread on every stall instead of one. The fix has to stop the xet
transfer from stalling in the first place, or route around it.

## First repro step

Set `HF_HUB_DISABLE_XET=1` (falls back to the plain HTTP transfer path)
and retry the same download against the same mirror. If the stall
disappears, that confirms the xet CAS path as the cause and turns this
from a theory into a bug report against `hf-xet` (or a routing gap in
the mirror config) rather than something fixable in this codebase alone.

Follow-ups if the xet theory confirms:

- Check whether the configured mirror actually proxies the CAS endpoint
  hf-xet talks to (it may be a distinct hostname/path from the Hub API
  the mirror was set up for).
- Consider defaulting `HF_HUB_DISABLE_XET=1` when a non-default
  `endpoint` is configured, since mirrors are the common case where this
  bites.

## Retry-hygiene note: `_cleanup_partial` may be targeting the wrong path

`_cleanup_partial` (~line 1090) removes `local_dir / "._____temp"` to
clear in-progress shards before a retry. That matches an older
`huggingface_hub` staging convention. Hub 1.26 stages in-progress
downloads under `local_dir/.cache/huggingface/` instead (the standard
cache layout, not a dot-underscore temp dir). If the installed hub
version has already moved to that layout, `_cleanup_partial` is
silently a no-op on real partial downloads — worth confirming against
the pinned `huggingface-hub` version and, if stale, retargeting the
cleanup path (or covering both, for older/newer hub compatibility)
before relying on it for retry correctness.

## Open questions

- Confirm hf-xet's timeout behavior against source/changelog for the
  pinned `huggingface_hub` version rather than inferring it — "ignores
  `HF_HUB_DOWNLOAD_TIMEOUT`" is the working theory, not yet verified
  against this exact version.
- Is the user's configured endpoint a mirror that was only set up to
  proxy `/api/models/*` and file resolution, not the separate xet CAS
  host? Get the actual `endpoint` value and CAS host from a stalled
  session.
- Does disabling xet fix it on a small model too, or only large ones
  (where xet's chunked reconstruction kicks in vs. a plain HTTP GET for
  small files)?
- Verify the actual staging path `huggingface_hub` uses for the pinned
  version, to settle the `_cleanup_partial` question above.
