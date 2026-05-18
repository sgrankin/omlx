#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models --max-model-memory 32GB

    # With pinned models
    omlx serve --model-dir /path/to/models --max-model-memory 48GB --pin llama-3b,qwen-7b
"""

import argparse
import contextlib
import faulthandler
import sys


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved.

    All argparse defaults are None, so `is not None` means the user
    explicitly passed the flag on the command line.
    """
    if hasattr(args, "model_dir") and args.model_dir is not None:
        return True
    if hasattr(args, "port") and args.port is not None:
        return True
    if hasattr(args, "max_model_memory") and args.max_model_memory is not None:
        return True
    if hasattr(args, "max_process_memory") and args.max_process_memory is not None:
        return True
    if hasattr(args, "host") and args.host is not None:
        return True
    if hasattr(args, "log_level") and args.log_level is not None:
        return True
    if hasattr(args, "mcp_config") and args.mcp_config is not None:
        return True
    if hasattr(args, "hf_endpoint") and args.hf_endpoint is not None:
        return True
    if hasattr(args, "ms_endpoint") and args.ms_endpoint is not None:
        return True
    if hasattr(args, "http_proxy") and args.http_proxy is not None:
        return True
    if hasattr(args, "https_proxy") and args.https_proxy is not None:
        return True
    if hasattr(args, "no_proxy") and args.no_proxy is not None:
        return True
    if hasattr(args, "ca_bundle") and args.ca_bundle is not None:
        return True
    return False


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import logging
    import os
    import uvicorn

    from ._version import __version__
    from .settings import init_settings, get_settings
    from .logging_config import configure_file_logging, AdminStatsAccessFilter

    try:
        from ._build_info import build_number
    except ImportError:
        build_number = None

    # Print version banner
    print(f"\033[33moMLX - LLM inference, optimized for your Mac\033[0m")
    print(f"\033[33m├─ https://github.com/jundot/omlx\033[0m")
    if build_number:
        print(f"\033[33m├─ Version: {__version__}\033[0m")
        print(f"\033[33m└─ Build: {build_number}\033[0m")
    else:
        print(f"\033[33m└─ Version: {__version__}\033[0m")
    print()

    # Initialize global settings first (to get log_level from file if not specified)
    settings = init_settings(base_path=args.base_path, cli_args=args)

    # Register TRACE level (5) — includes full message content
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    # Configure logging (use settings value which has proper priority)
    level_name = settings.server.log_level.upper()
    log_level = TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set omlx loggers
    for name in ["omlx", "omlx.scheduler", "omlx.paged_ssd_cache",
                 "omlx.memory_monitor", "omlx.paged_cache", "omlx.prefix_cache",
                 "omlx.engine_pool", "omlx.model_discovery"]:
        logging.getLogger(name).setLevel(log_level)

    # Suppress repetitive admin stats access logs
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    if log_level > TRACE:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Ensure required directories exist
    settings.ensure_directories()

    # Apply HuggingFace endpoint if configured
    if settings.huggingface.endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface.endpoint

    # Apply ModelScope endpoint if configured
    if settings.modelscope.endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = settings.modelscope.endpoint

    # Apply proxy/TLS settings if configured
    if settings.network.http_proxy:
        os.environ["HTTP_PROXY"] = settings.network.http_proxy
        os.environ["http_proxy"] = settings.network.http_proxy
    if settings.network.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.network.https_proxy
        os.environ["https_proxy"] = settings.network.https_proxy
    if settings.network.no_proxy:
        os.environ["NO_PROXY"] = settings.network.no_proxy
        os.environ["no_proxy"] = settings.network.no_proxy
    if settings.network.ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = settings.network.ca_bundle
        os.environ["SSL_CERT_FILE"] = settings.network.ca_bundle

    # Save CLI args to settings.json if non-default values provided
    if _has_cli_overrides(args):
        try:
            settings.save()
            print("Saved CLI arguments to settings.json")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    # Configure file logging (writes to {base_path}/logs/server.log)
    log_dir = settings.logging.get_log_dir(settings.base_path)
    configure_file_logging(
        log_dir=log_dir,
        level=settings.server.log_level,
        include_request_id=True,
        retention_days=settings.logging.retention_days,
    )
    print(f"Log directory: {log_dir}")

    # Enable native crash diagnostics (SIGABRT, SIGSEGV, SIGFPE, SIGBUS).
    # On Metal/MLX crashes (#511, #520), this dumps all Python thread
    # tracebacks to the server log before the process terminates.
    crash_log_path = log_dir / "crash.log"
    _crash_file = open(crash_log_path, "a")
    faulthandler.enable(file=_crash_file, all_threads=True)

    # Validate settings
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Import server and config
    from .server import app, init_server
    from .config import parse_size

    model_dirs = settings.model.get_model_dirs(settings.base_path)
    print(f"Base path: {settings.base_path}")
    print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
    print(f"Max model memory: {settings.model.max_model_memory}")
    print(f"Max process memory: {settings.memory.max_process_memory}")

    # Store MCP config path for FastAPI startup
    # Priority: CLI arg > settings.json
    mcp_config = args.mcp_config or settings.mcp.config_path
    if mcp_config:
        print(f"MCP config: {mcp_config}")
        os.environ["OMLX_MCP_CONFIG"] = mcp_config

    # Determine paged SSD cache directory
    # Priority: --no-cache > CLI arg > settings file
    if args.no_cache:
        paged_ssd_cache_dir = None
    elif args.paged_ssd_cache_dir:
        # CLI argument takes precedence
        paged_ssd_cache_dir = args.paged_ssd_cache_dir
    elif settings.cache.enabled:
        # Use settings file value (resolved path or default)
        paged_ssd_cache_dir = str(settings.cache.get_ssd_cache_dir(settings.base_path))
    else:
        # Cache explicitly disabled in settings
        paged_ssd_cache_dir = None

    # Build scheduler config for BatchedEngine
    scheduler_config = settings.to_scheduler_config()
    # Set paged SSD cache options
    scheduler_config.paged_ssd_cache_dir = paged_ssd_cache_dir
    # Determine cache max size: CLI arg > settings (with auto resolution)
    if paged_ssd_cache_dir:
        if args.paged_ssd_cache_max_size:
            # CLI argument specified explicitly
            cache_max_size_bytes = parse_size(args.paged_ssd_cache_max_size)
        else:
            # Use settings value (handles "auto" -> 10% of SSD capacity)
            cache_max_size_bytes = settings.cache.get_ssd_cache_max_size_bytes(settings.base_path)
        scheduler_config.paged_ssd_cache_max_size = cache_max_size_bytes
    else:
        scheduler_config.paged_ssd_cache_max_size = 0
        cache_max_size_bytes = 0

    # Hot cache: CLI arg > settings
    if paged_ssd_cache_dir:
        if args.hot_cache_max_size:
            hot_cache_max_bytes = parse_size(args.hot_cache_max_size)
        else:
            hot_cache_max_bytes = settings.cache.get_hot_cache_max_size_bytes()
        scheduler_config.hot_cache_max_size = hot_cache_max_bytes
    else:
        scheduler_config.hot_cache_max_size = 0

    if args.no_cache:
        print("Mode: Multi-model serving (no oMLX cache, mlx-lm BatchGenerator only)")
    elif paged_ssd_cache_dir:
        print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
        # Format cache size for display
        cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
        print(f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})")
        if scheduler_config.hot_cache_max_size > 0:
            hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
            print(f"Hot cache: {hot_display} (in-memory)")
    else:
        print("Mode: Multi-model serving (continuous batching, no cache)")

    # Set MLX buffer cache limit high to prevent the allocator from
    # immediately releasing Metal buffers when the cache is full.
    # Without this, allocator::free() can call buf->release() while the
    # GPU is still using the buffer, causing kernel panics on M4.
    # With a large cache limit, freed buffers always stay in the pool
    # and are only released via mx.clear_cache() (which we protect
    # with mx.synchronize()). See issue #300.
    import mlx.core as mx
    total_mem = mx.device_info().get("memory_size", 0)
    if total_mem > 0:
        mx.set_cache_limit(total_mem)

    # Initialize server
    # Note: pinned_models and default_model are managed via admin page (model_settings.json)
    # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
    init_server(
        model_dirs=[str(d) for d in model_dirs],
        max_model_memory=settings.model.get_max_model_memory_bytes(),
        scheduler_config=scheduler_config,
        api_key=settings.auth.api_key,
        global_settings=settings,
    )

    # Start server
    print(f"Starting server at http://{settings.server.host}:{settings.server.port}")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = "debug" if settings.server.log_level == "trace" else settings.server.log_level
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
    )



def launch_command(args, extra_args: list[str] | None = None):
    """Launch an external tool integrated with oMLX.

    extra_args are unknown CLI tokens forwarded to the underlying tool binary
    (e.g. ``-r`` / ``--resume <id>`` for Claude Code).
    """
    import requests

    from .integrations import get_integration, list_integrations
    from .settings import GlobalSettings

    tool_name = args.tool

    if tool_name == "list":
        print("Available integrations:")
        for integ in list_integrations():
            installed = "installed" if integ.is_installed() else "not installed"
            print(f"  {integ.name:12s} {integ.display_name} ({installed})")
        return

    integration = get_integration(tool_name)
    if integration is None:
        print(f"Unknown integration: {tool_name}")
        print("Available: " + ", ".join(i.name for i in list_integrations()))
        sys.exit(1)

    # Resolve host/port: CLI args > env vars > settings.json > defaults
    settings = GlobalSettings.load()
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    # 0.0.0.0 is a valid bind address but not a valid connect address.
    # Fall back to localhost so launch can reach the server regardless
    # of which interface it was bound to.
    connect_host = host if host and host != "0.0.0.0" else "127.0.0.1"

    # Check if oMLX server is running
    base_url = f"http://{connect_host}:{port}"
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx serve")
        sys.exit(1)

    # Get API key: CLI args > settings.json > empty
    api_key = getattr(args, "api_key", None) or settings.auth.api_key or ""

    # Build headers for authenticated requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Pre-fetch model status (context_window, max_tokens, model_type per model)
    models_status_map: dict[str, dict] = {}
    try:
        resp = requests.get(f"{base_url}/v1/models/status", headers=headers, timeout=5)
        if resp.ok:
            for m in resp.json().get("models", []):
                models_status_map[m["id"]] = m
    except Exception:
        pass

    # Determine model
    model = args.model
    if not model:
        # Fetch available models from server
        try:
            resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("model_type") in ("llm", "vlm", None)
            ]
        except Exception:
            models = []

        if not models:
            print("No models available. Load a model first.")
            sys.exit(1)

        if len(models) == 1:
            model = models[0]
            print(f"Using model: {model}")
        else:
            models_info_list = [
                {"id": m_id, **models_status_map.get(m_id, {})}
                for m_id in models
            ]
            model = integration.select_model(
                models_info_list, integration.display_name
            )

    # Check if tool is installed
    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    # Resolve model limits from pre-fetched status
    model_info = models_status_map.get(model, {})
    context_window = model_info.get("max_context_window")
    max_tokens = model_info.get("max_tokens")
    model_type = model_info.get("model_type")

    # Launch
    print(f"Launching {integration.display_name} with model {model}...")
    tools_profile = getattr(args, "tools_profile", "coding")
    integration.launch(
        port=port,
        api_key=api_key,
        model=model,
        host=connect_host,
        tools_profile=tools_profile,
        context_window=context_window,
        max_tokens=max_tokens,
        model_type=model_type,
        extra_args=extra_args,
    )


def diagnose_menubar() -> int:
    """Diagnose why the oMLX menubar icon might be missing.

    Reports macOS version, app install path, running menubar process, and the
    most recent visibility warning from the log. Prints manual recovery steps
    since Tahoe's ControlCenter doesn't expose a public API to re-enable a
    hidden status item.
    """
    import platform
    import subprocess
    from pathlib import Path

    print("oMLX menubar diagnostics")
    print("=" * 40)

    mac_ver = platform.mac_ver()[0] or "unknown"
    print(f"macOS:          {mac_ver}")
    print(f"Bundle ID:      com.omlx.app")

    app_path = Path("/Applications/oMLX.app")
    print(f"App installed:  {'yes' if app_path.exists() else 'NO (install DMG first)'}")

    try:
        res = subprocess.run(
            ["pgrep", "-af", "omlx_app"],
            capture_output=True, text=True, timeout=5,
        )
        running = bool(res.stdout.strip())
        print(f"Menubar app:    {'running' if running else 'NOT running'}")
        if running:
            first_line = res.stdout.strip().splitlines()[0]
            pid = first_line.split()[0] if first_line else "?"
            print(f"PID:            {pid}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Menubar app:    check failed ({e})")

    log_dir = Path.home() / "Library" / "Application Support" / "oMLX" / "logs"
    # menubar.log captures the visibility probe (frame + isVisible);
    # server.log may carry fallback warnings for older builds.
    log_candidates = [log_dir / "menubar.log", log_dir / "server.log"]
    print(f"Log dir:        {log_dir}")

    hits: list[tuple[str, str]] = []
    for path in log_candidates:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 131072))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError as e:
            print(f"Could not read {path.name}: {e}")
            continue
        for ln in tail.splitlines():
            if (
                "menubar visibility probe" in ln
                or "NSStatusItem" in ln
                or "ControlCenter" in ln
                or "Menu Bar" in ln
            ):
                hits.append((path.name, ln))

    if hits:
        print("\nRecent visibility log entries (last 10):")
        for src, ln in hits[-10:]:
            print(f"  [{src}] {ln}")
    else:
        print("\nNo visibility log entries found (app may not have probed yet).")

    print()
    print("If the icon is missing on macOS Tahoe (26.x):")
    print("  1. Open System Settings > Menu Bar")
    print("     open 'x-apple.systempreferences:com.apple.ControlCenter-Settings.extension?MenuBar'")
    print("  2. Find 'oMLX' and set it to 'Show in Menu Bar'")
    print("  3. If oMLX isn't in the list, quit the menubar app and relaunch oMLX.app")
    print()
    print("Note: Apple's sandbox policy prevents third-party apps from")
    print("programmatically re-enabling their own menubar visibility on Tahoe.")
    return 0


def diagnose_command(args) -> int:
    """Dispatch 'omlx diagnose <target>' to the appropriate subcommand."""
    target = getattr(args, "target", None)
    if target == "menubar":
        return diagnose_menubar()
    print(f"Unknown diagnose target: {target}")
    print("Available: menubar")
    return 1


def _parse_layer_spec(spec: str | None) -> list[int] | None:
    """Parse a ``--layers`` spec into a list of indices.

    Accepts an inclusive range ``"10-31"`` or a comma list ``"10,11,12"``.
    Returns None for an empty spec (meaning "all layers").
    """
    if not spec:
        return None
    spec = spec.strip()
    if "," not in spec and spec.count("-") == 1 and not spec.startswith("-"):
        start_str, end_str = spec.split("-")
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError(f"range start {start} exceeds end {end}")
        return list(range(start, end + 1))
    return [int(tok) for tok in spec.split(",") if tok.strip()]


@contextlib.contextmanager
def _drop_mtp_weights_on_load():
    """Scoped shim: drop ``mtp.*`` tensors during ``load_weights``.

    MTP (multi-token prediction) heads are a decoding accelerator and play
    no part in steering. Some checkpoints (e.g. Qwen3.6) ship MTP-head
    weights that mlx-vlm's loader does not sanitize away — it sanitizes
    against the ``LanguageModel`` class, never the ``Model.sanitize`` that
    drops them — so a plain load rejects them as unexpected parameters.
    Filtering them here loads the model as an ordinary non-MTP model,
    which is exactly what steering needs.
    """
    import mlx.nn as _nn

    original = _nn.Module.load_weights

    def _filtered(self, weights, *args, **kwargs):
        if isinstance(weights, list):
            weights = [
                (k, v)
                for k, v in weights
                if ".mtp." not in k and not k.startswith("mtp.")
            ]
        return original(self, weights, *args, **kwargs)

    _nn.Module.load_weights = _filtered
    try:
        yield
    finally:
        _nn.Module.load_weights = original


def _load_steering_model(model_path: str):
    """Load a model + tokenizer for steering work, VLM- and oQ-aware.

    Mirrors omlx's own engine load path: pre-load patches (oQ per-layer
    quant key expansion), custom-quant loaders (paroquant), and — for VLMs
    — the audio-config and nested-visual remap shims, since mlx-lm cannot
    load VLM checkpoints. MTP-head weights are dropped (see
    :func:`_drop_mtp_weights_on_load`).
    """
    import json
    from pathlib import Path

    from .utils.model_loading import (
        maybe_apply_pre_load_patches,
        maybe_load_custom_quantization,
    )

    is_vlm = False
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            is_vlm = "vision_config" in cfg or "text_config" in cfg
        except (OSError, json.JSONDecodeError):
            pass

    # Pre-load patches: expand oQ per-layer quant keys, and inject MTP
    # modules so an MTP-head checkpoint's mtp.* weights load into a real
    # submodule rather than being rejected as unexpected parameters.
    maybe_apply_pre_load_patches(model_path)

    custom = maybe_load_custom_quantization(model_path, is_vlm=is_vlm)
    if custom is not None:
        model, processor = custom
        return model, getattr(processor, "tokenizer", processor)

    if is_vlm:
        from mlx_vlm.utils import load as vlm_load

        from .engine.vlm import (
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
            _drop_mtp_weights_on_load(),
        ):
            model, processor = vlm_load(model_path)
        return model, getattr(processor, "tokenizer", processor)

    from mlx_lm import load as lm_load

    with _drop_mtp_weights_on_load():
        return lm_load(model_path)


def steering_generate_command(args) -> int:
    """Generate a steering vector from contrastive prompt pairs."""
    import json

    from .steering_generator import generate_steering_vector

    prompts_path = args.prompts
    try:
        with open(prompts_path, encoding="utf-8") as f:
            data = json.load(f)
        positive = list(data["positive"])
        negative = list(data["negative"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Failed to read prompt pairs from {prompts_path}: {e}")
        print('Expected JSON: {"positive": [...], "negative": [...]}')
        return 1

    if args.max_pairs is not None:
        positive = positive[: args.max_pairs]
        negative = negative[: args.max_pairs]

    try:
        layers = _parse_layer_spec(args.layers)
    except ValueError as e:
        print(f"Invalid --layers spec {args.layers!r}: {e}")
        return 1

    print(f"Loading model: {args.model}")
    model, tokenizer = _load_steering_model(args.model)

    vector = generate_steering_vector(
        model,
        tokenizer,
        positive,
        negative,
        method=args.method,
        model_name=args.model,
        layers=layers,
        scaling=args.scaling,
        orthogonalize=args.orthogonalize,
    )
    vector.save(args.output)
    print(
        f"Wrote steering vector ({len(vector.directions)} layers, "
        f"n_embd={vector.n_embd}, method={vector.method}) to {args.output}"
    )
    return 0


def steering_eval_command(args) -> int:
    """Generate from a model at a sweep of steering strengths."""
    from .steering import SteeringVector
    from .steering_eval import evaluate_steering

    try:
        scales = [float(s) for s in args.scales.split(",") if s.strip()]
    except ValueError as e:
        print(f"Invalid --scales {args.scales!r}: {e}")
        return 1
    if not scales:
        print("--scales is empty")
        return 1

    try:
        layers = _parse_layer_spec(args.layers)
    except ValueError as e:
        print(f"Invalid --layers spec {args.layers!r}: {e}")
        return 1
    layer_start = min(layers) if layers else None
    layer_end = max(layers) if layers else None

    try:
        vector = SteeringVector.load(args.vector)
    except (FileNotFoundError, ValueError) as e:
        print(f"Failed to load steering vector: {e}")
        return 1

    print(f"Loading model: {args.model}")
    model, tokenizer = _load_steering_model(args.model)

    results = evaluate_steering(
        model,
        tokenizer,
        vector,
        args.prompt,
        scales=scales,
        mode=args.mode,
        layer_start=layer_start,
        layer_end=layer_end,
        max_tokens=args.max_tokens,
    )

    print()
    for scale, text in results:
        label = (
            "baseline (no steering)"
            if scale == 0.0
            else f"strength {scale:+g}  mode={args.mode}"
        )
        bar = "=" * 70
        print(f"{bar}\n  {label}\n{bar}\n{text}\n")
    return 0


def steering_command(args) -> int:
    """Dispatch 'omlx steering <subcommand>'."""
    sub = getattr(args, "steering_command", None)
    if sub == "generate":
        return steering_generate_command(args)
    if sub == "eval":
        return steering_eval_command(args)
    print(f"Unknown steering subcommand: {sub}")
    print("Available: generate, eval")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="omlx: Production-ready LLM server for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
  omlx launch codex --model qwen3.5
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command (multi-model)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start multi-model OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a multi-model inference server with LRU-based memory management.

Models are discovered from subdirectories of --model-dir. Each subdirectory
should contain a valid model with config.json and *.safetensors files.

Example directory structure:
  /path/to/models/
  ├── llama-3b/           → model_id: "llama-3b"
  │   ├── config.json
  │   └── model.safetensors
  ├── qwen-7b/            → model_id: "qwen-7b"
  └── mistral-7b/         → model_id: "mistral-7b"
""",
    )

    # Required arguments
    serve_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories (default: ~/.omlx/models)",
    )
    serve_parser.add_argument(
        "--max-model-memory",
        type=str,
        default=None,
        help="Maximum memory for loaded models (e.g., 32GB, 'disabled'). Default: 80%% of system memory.",
    )
    serve_parser.add_argument(
        "--max-process-memory",
        type=str,
        default=None,
        help=(
            "Max total process memory as percentage of system RAM (10-99%%), "
            "'auto' (RAM - 8GB), or 'disabled'. Default: auto."
        ),
    )

    # Server options
    serve_parser.add_argument("--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000)")
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )
    serve_parser.add_argument(
        "--sse-keepalive-mode",
        type=str,
        choices=["chunk", "comment", "off"],
        default=None,
        help="SSE keepalive emission mode (default: chunk). 'chunk' emits "
        "protocol-aware no-op events compatible with strict clients like "
        "OpenClaw / WorkBuddy; 'comment' emits the legacy ': keep-alive' SSE "
        "comment; 'off' disables keepalive entirely",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Max requests processed simultaneously. Higher values increase throughput but use more memory. (default: 8)",
    )

    # paged SSD cache options
    serve_parser.add_argument(
        "--paged-ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for paged SSD cache storage (enables oMLX prefix cache)",
    )
    serve_parser.add_argument(
        "--paged-ssd-cache-max-size",
        type=str,
        default=None,
        help="Maximum paged SSD cache size (e.g., '100GB', '50GB'). Default: 100GB",
    )
    serve_parser.add_argument(
        "--hot-cache-max-size",
        type=str,
        default=None,
        help="Maximum in-memory hot cache size (e.g., '8GB', '4GB'). Default: 0 (disabled)",
    )
    serve_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oMLX paged SSD cache. mlx-lm BatchGenerator still manages KV states internally.",
    )
    serve_parser.add_argument(
        "--initial-cache-blocks",
        type=int,
        default=None,
        help="Number of cache blocks to pre-allocate at startup (default: 256). "
        "Higher values reduce dynamic allocation overhead for large contexts.",
    )

    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )

    # HuggingFace options
    serve_parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Custom HuggingFace Hub endpoint URL (e.g., https://hf-mirror.com)",
    )

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
    )

    # Network options
    serve_parser.add_argument(
        "--http-proxy",
        type=str,
        default=None,
        help="HTTP proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--https-proxy",
        type=str,
        default=None,
        help="HTTPS proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--no-proxy",
        type=str,
        default=None,
        help="Comma-separated hosts/IPs to bypass proxy (e.g., localhost,127.0.0.1)",
    )
    serve_parser.add_argument(
        "--ca-bundle",
        type=str,
        default=None,
        help="Path to CA bundle PEM file for TLS interception environments",
    )

    # Base path and auth
    serve_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory for oMLX data (default: ~/.omlx)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (optional)",
    )

    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch an external tool with oMLX integration",
        description="Configure and launch external coding tools (Claude Code, Copilot, Codex, OpenCode, OpenClaw, Hermes Agent, Pi) "
        "to use the running oMLX server.",
    )
    launch_parser.add_argument(
        "tool",
        type=str,
        help="Tool to launch: claude, copilot, codex, opencode, openclaw, hermes, pi, or 'list' to show available",
    )
    launch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (interactive selection if not specified)",
    )
    launch_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="oMLX server host (default: from settings or 127.0.0.1)",
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="oMLX server port (default: from settings or 8000)",
    )
    launch_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for oMLX server authentication",
    )
    launch_parser.add_argument(
        "--tools-profile",
        type=str,
        default="coding",
        choices=["minimal", "coding", "messaging", "full"],
        help="OpenClaw tools profile (default: coding)",
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose installation or runtime issues",
        description="Run diagnostic checks and print recovery steps.",
    )
    diagnose_parser.add_argument(
        "target",
        type=str,
        choices=["menubar"],
        help="What to diagnose. 'menubar' checks Tahoe ControlCenter visibility.",
    )

    # Steering command
    steering_parser = subparsers.add_parser(
        "steering",
        help="Steering (control) vector tools",
        description="Generate and manage steering vectors — per-layer additive "
        "biases on the residual stream that nudge model behaviour.",
    )
    steering_sub = steering_parser.add_subparsers(
        dest="steering_command", help="Steering subcommands"
    )
    steering_gen = steering_sub.add_parser(
        "generate",
        help="Generate a steering vector from contrastive prompt pairs",
        description="Run a model on (positive, negative) prompt pairs, capture "
        "per-layer hidden states, and reduce their differences to a per-layer "
        "steering direction.",
    )
    steering_gen.add_argument(
        "--model", type=str, required=True, help="Model path or HF repo id"
    )
    steering_gen.add_argument(
        "--prompts",
        type=str,
        required=True,
        help='JSON file: {"positive": [...], "negative": [...]} (equal-length lists)',
    )
    steering_gen.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Destination .safetensors file for the steering vector",
    )
    steering_gen.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "mean", "crosscov"],
        help=(
            "Reduction method: 'mean', 'pca', or 'crosscov' (cross-covariance "
            "contrastive axis — cleaner, wants many prompt pairs). Default: pca"
        ),
    )
    steering_gen.add_argument(
        "--scaling",
        type=str,
        default="magnitude",
        choices=["unit", "magnitude"],
        help=(
            "Per-layer scaling: 'magnitude' (scale by mean projection so one "
            "strength works across layers — recommended) or 'unit' (unit-norm "
            "directions). Default: magnitude"
        ),
    )
    steering_gen.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Use at most this many prompt pairs (default: all)",
    )
    steering_gen.add_argument(
        "--layers",
        type=str,
        default=None,
        help='Layers to generate, e.g. "10-31" or "10,11,12" (default: all)',
    )
    steering_gen.add_argument(
        "--orthogonalize",
        action="store_true",
        help=(
            "Project each direction orthogonal to the control-class mean "
            "(ds4-style) — strips general activation drift from the trait axis"
        ),
    )

    steering_eval = steering_sub.add_parser(
        "eval",
        help="Generate at a sweep of steering strengths to compare behaviour",
        description="Apply a steering vector at several strengths and print "
        "the generated text for each, including a no-steering baseline. "
        "Generation is greedy so differences are attributable to steering.",
    )
    steering_eval.add_argument(
        "--model", type=str, required=True, help="Model path or HF repo id"
    )
    steering_eval.add_argument(
        "--vector",
        type=str,
        required=True,
        help="Steering vector .safetensors file to evaluate",
    )
    steering_eval.add_argument(
        "--prompt", type=str, required=True, help="User prompt to generate from"
    )
    steering_eval.add_argument(
        "--scales",
        type=str,
        default="-1,0,0.5,1,1.5",
        help="Comma-separated strengths; 0 = baseline (default: -1,0,0.5,1,1.5)",
    )
    steering_eval.add_argument(
        "--mode",
        type=str,
        default="add",
        choices=["add", "project"],
        help="Steering mode (default: add)",
    )
    steering_eval.add_argument(
        "--layers",
        type=str,
        default=None,
        help='Layer range to steer, e.g. "10-31" (default: all)',
    )
    steering_eval.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Tokens to generate per scale (default: 200)",
    )

    # Use parse_known_args so `omlx launch <tool> -- ...` can forward unknown
    # tokens (e.g. `-r`, `--resume <id>`) to the underlying tool binary.
    # Non-launch commands keep the previous strictness by rejecting unknowns.
    args, extra_args = parser.parse_known_args()

    if args.command == "launch":
        launch_command(args, extra_args=extra_args)
    else:
        if extra_args:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
        if args.command == "serve":
            serve_command(args)
        elif args.command == "diagnose":
            sys.exit(diagnose_command(args))
        elif args.command == "steering":
            sys.exit(steering_command(args))
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
