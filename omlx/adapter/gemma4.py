# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 reasoning-channel output parsing and message extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..api.utils import _PRESERVE_BOUNDARY_KEY
from ..utils.tokenizer import create_streaming_detokenizer
from .output_parser import OutputParserFinalizeResult, OutputParserTokenResult

logger = logging.getLogger(__name__)

_OPEN_MARKER = "<|channel>thought\n"
_OPEN_MARKER_BARE = "<|channel>"
_CLOSE_MARKER = "<channel|>"
_TURN_END_MARKER = "<turn|>"
_TOOL_RESPONSE_OPEN = "<|tool_response>"
_TOOL_RESPONSE_CLOSE = "<tool_response|>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_LEADING_THOUGHT_RE = re.compile(
    r"\A\s*(?:(?:<think>.*?</think>|<\|channel>.*?<channel\|>)\s*)+",
    re.DOTALL,
)

# Matches the STRAY bare-token spellings (<|tool_call> and <tool_call|>),
# not the template's well-formed closing form (</tool_call|> with slash).
_PROTOCOL_MARKER_RE = re.compile(r"<\|tool_call>|<tool_call\|>")


def _strip_protocol_markers(text: Any) -> Any:
    """Remove stray <|tool_call> / <tool_call|> tokens from assistant content."""
    if not isinstance(text, str) or not text:
        return text
    return _PROTOCOL_MARKER_RE.sub("", text)


def _try_parse_json(s: str) -> Any:
    """Parse string as JSON if possible, otherwise return as-is."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s or not (s.startswith("{") or s.startswith("[")):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _strip_thinking(text: Any) -> Any:
    """Remove leading ``<think>...</think>`` or raw ``<|channel>...<channel|>`` spans.

    Gemma 4's multi-turn rule requires that only the final visible answer
    is kept in chat history. Clients such as Open WebUI replay the full
    assistant content (including the rendered ``<think>`` block, or the
    raw protocol form when a client preserves it). Feeding prior thought
    blocks back primes the model to emit malformed channel markers on the
    next turn, which then leak into user-facing output.

    The match is anchored to the start of the message: the rendered thought
    block always precedes the visible answer, so this catches every
    legitimate occurrence while leaving inline mentions (e.g. an assistant
    explaining how ``<think>`` tags work) untouched.
    """
    if not isinstance(text, str) or not text:
        return text
    return _LEADING_THOUGHT_RE.sub("", text, count=1)


def extract_gemma4_messages(
    messages: list[Any],
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
    consolidate_system_messages: bool = True,
) -> list[dict]:
    """Convert OpenAI-format messages to Gemma 4 chat-template format.

    The Gemma 4 chat template does not handle ``role=tool`` messages.
    Tool results must instead appear on a model-role turn as a
    ``tool_responses`` list, where each entry is::

        {"name": "<function_name>", "response": <dict_or_scalar>}

    This function:
    - Passes non-tool messages through unchanged.
    - Preserves ``tool_calls`` on assistant turns (template renders them
      as ``<|tool_call>...</tool_call|>``).
    - Folds consecutive ``role=tool`` messages that follow an assistant
      turn into a single ``{"role": "assistant", "tool_responses": [...]}``
      message, resolving function names from the preceding tool_calls by
      ``tool_call_id``.  Falls back to the raw ``tool_call_id`` as the
      name when no match is found.
    - JSON-parses tool result content into a dict/list where possible so
      the template renders structured responses correctly.

    Args:
        messages: OpenAI-format Message objects or dicts.
        max_tool_result_tokens: Maximum token count for tool results
            (truncation applied when tokenizer is provided).
        tokenizer: Tokenizer for optional truncation.
        consolidate_system_messages: When True, preserve the legacy behavior
            of moving all system/developer messages to the leading system
            prompt. Server routes pass False and defer that decision until the
            model chat template can be probed.

    Returns:
        List of dicts ready for ``tokenizer.apply_chat_template``.
    """
    from ..api.utils import (
        _extract_text_from_content_list,
    )  # avoid circular at module level

    processed: list[dict] = []

    # Build index of message objects as plain dicts
    raw: list[dict] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            raw.append(msg.model_dump())
        elif isinstance(msg, dict):
            raw.append(dict(msg))
        else:
            raw.append(
                {
                    "role": getattr(msg, "role", "user"),
                    "content": getattr(msg, "content", ""),
                }
            )

    i = 0
    while i < len(raw):
        msg = raw[i]
        role = msg.get("role", "user")

        if role == "developer":
            role = "system"

        if role == "tool":
            # Orphaned tool result with no preceding assistant turn — attach
            # to a synthetic assistant turn with no content.
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            if max_tool_result_tokens and tokenizer and content:
                from ..api.anthropic_utils import truncate_tool_result

                content = truncate_tool_result(
                    content, max_tool_result_tokens, tokenizer
                )
            response = _try_parse_json(content)
            # Fallback name: use tool_call_id
            processed.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_responses": [
                        {"name": tool_call_id or "unknown", "response": response}
                    ],
                    _PRESERVE_BOUNDARY_KEY: True,
                }
            )
            i += 1
            continue

        if role == "assistant":
            # Build a tool_call_id → function_name lookup from this turn's calls.
            tc_id_to_name: dict[str, str] = {}
            tool_calls_raw = msg.get("tool_calls") or []
            for tc in tool_calls_raw:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    func_name = (tc.get("function") or {}).get("name", "")
                else:
                    tc_id = getattr(tc, "id", "")
                    func = getattr(tc, "function", None)
                    func_name = getattr(func, "name", "") if func else ""
                if tc_id:
                    tc_id_to_name[tc_id] = func_name

            # Extract content
            content = msg.get("content", "")
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            # Per Gemma 4's multi-turn rule, prior thought blocks must not
            # be fed back into the next turn. Strip them before rendering.
            content = _strip_thinking(content)
            content = _strip_protocol_markers(content)

            out_msg: dict = {"role": "assistant", "content": content or ""}

            # Preserve tool_calls for template rendering
            if tool_calls_raw:
                out_calls = []
                for tc in tool_calls_raw:
                    if isinstance(tc, dict):
                        func = tc.get("function") or {}
                        out_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": _try_parse_json(
                                        func.get("arguments", "{}")
                                    ),
                                },
                            }
                        )
                    else:
                        func = getattr(tc, "function", None)
                        args_str = getattr(func, "arguments", "{}") if func else "{}"
                        out_calls.append(
                            {
                                "id": getattr(tc, "id", ""),
                                "function": {
                                    "name": getattr(func, "name", "") if func else "",
                                    "arguments": _try_parse_json(args_str),
                                },
                            }
                        )
                out_msg["tool_calls"] = out_calls
                out_msg[_PRESERVE_BOUNDARY_KEY] = True

            processed.append(out_msg)
            i += 1

            # Consume any immediately following tool results into a
            # single model turn with tool_responses.
            tool_responses = []
            while i < len(raw) and raw[i].get("role") == "tool":
                tr = raw[i]
                tc_id = tr.get("tool_call_id", "")
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = _extract_text_from_content_list(tr_content)
                if max_tool_result_tokens and tokenizer and tr_content:
                    from ..api.anthropic_utils import truncate_tool_result

                    tr_content = truncate_tool_result(
                        tr_content, max_tool_result_tokens, tokenizer
                    )
                response = _try_parse_json(tr_content)
                name = tc_id_to_name.get(tc_id) or tc_id or "unknown"
                tool_responses.append({"name": name, "response": response})
                i += 1

            if tool_responses:
                # Attach tool_responses to the SAME assistant message that
                # has tool_calls.  The Gemma 4 chat template checks for
                # tool_responses on the current message (lines 261-267)
                # BEFORE falling back to a forward-scan for role='tool'
                # messages (lines 268-302).  Putting them on a separate
                # assistant message causes both paths to miss, producing a
                # corrupt bare <|tool_response> tag and making the model
                # loop on the same tool call.
                out_msg["tool_responses"] = tool_responses
            continue

        # All other roles (user, system)
        # Preserve image_url and input_audio parts for VLM processing
        content = msg.get("content", "")
        if isinstance(content, list):
            from ..api.utils import _extract_multimodal_content_list

            multimodal_parts = _extract_multimodal_content_list(content)
            multimodal_types = {"image_url", "input_audio"}
            has_multimodal = any(
                p.get("type") in multimodal_types for p in multimodal_parts
            )
            if has_multimodal:
                content = multimodal_parts
            else:
                content = _extract_text_from_content_list(content)
        out: dict = {"role": role, "content": content if content is not None else ""}
        processed.append(out)
        i += 1

    # Standard cleanup passes shared with other extractors
    from ..api.utils import (
        _consolidate_system_messages,
        _drop_void_assistant_messages,
        _merge_consecutive_roles,
    )

    cleaned = processed
    if consolidate_system_messages:
        cleaned = _consolidate_system_messages(cleaned)
    cleaned = _drop_void_assistant_messages(cleaned)
    return _merge_consecutive_roles(cleaned)


def _matching_prefix_len(text: str, marker: str) -> int:
    """Return longest suffix of ``text`` that is a prefix of ``marker``."""
    max_len = min(len(text), len(marker) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0


class _Gemma4LegacyOutputParserSession:
    """Legacy text-based parser. Kept behind a flag for A/B comparison.

    Pattern-matches decoded marker strings. Has a structural limitation:
    reasoning content that happens to contain the literal marker strings
    (e.g. the model writing about ``<channel|>``) is misread as a state
    transition. Prefer ``Gemma4OutputParserSession`` which keys off special
    token IDs and can't be fooled by regular-token text."""

    def __init__(self, tokenizer: Any, model_path: str | None = None):
        self._tokenizer = tokenizer
        self._buffer = ""
        self._in_thought = False
        self._text_mode = False

        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

    def notify_prefilled_thought(self) -> None:
        """Start the session inside a thought block opened by the prompt.

        Gemma 4's chat template opens the thought channel in the prompt when
        generation continues after a tool response. The model therefore
        starts by emitting the thought body and never generates the opening
        marker that normally sets ``_in_thought``.
        """
        self._in_thought = True

    def _append_text(
        self,
        stream_parts: list[str],
        visible_parts: list[str],
        text: str,
    ) -> None:
        if not text:
            return
        stream_parts.append(text)
        visible_parts.append(text)

    def _active_markers(self) -> list[str]:
        # Channel open/close are tracked unconditionally so a stray
        # ``<channel|>`` outside a thought block (occasionally emitted in long
        # multi-turn contexts) is absorbed instead of leaking into visible
        # text. ``_OPEN_MARKER_BARE`` is a defensive fallback for malformed
        # opens (e.g. ``<|channel>thought<channel|>`` with no newline, or a
        # bare ``<|channel>`` emitted when the model is confused by polluted
        # history). Tool-call markup is intentionally not tracked here — the
        # downstream ``ToolCallStreamFilter`` removes it from stream deltas
        # while ``parse_tool_calls`` still sees the raw markers in
        # ``output_text`` for extraction.
        return [
            _OPEN_MARKER,
            _OPEN_MARKER_BARE,
            _CLOSE_MARKER,
            _TURN_END_MARKER,
            _TOOL_RESPONSE_OPEN,
            _TOOL_RESPONSE_CLOSE,
        ]

    @staticmethod
    def _find_next_marker(
        source: str, pos: int, markers: list[str]
    ) -> tuple[int, str] | tuple[None, None]:
        next_idx: int | None = None
        next_marker: str | None = None
        for marker in markers:
            idx = source.find(marker, pos)
            if idx == -1:
                continue
            if next_idx is None or idx < next_idx:
                next_idx = idx
                next_marker = marker
        return next_idx, next_marker

    def _consume_text(
        self, text: str, *, final: bool = False
    ) -> OutputParserTokenResult:
        source = self._buffer + text
        self._buffer = ""

        stream_parts: list[str] = []
        visible_parts: list[str] = []
        pos = 0

        while pos < len(source):
            markers = self._active_markers()
            idx, marker = self._find_next_marker(source, pos, markers)

            if idx is None or marker is None:
                remainder = source[pos:]
                if not final:
                    keep = max(
                        _matching_prefix_len(remainder, marker_text)
                        for marker_text in markers
                    )
                    if keep:
                        emit = remainder[:-keep]
                        self._buffer = remainder[-keep:]
                    else:
                        emit = remainder
                else:
                    emit = remainder

                self._append_text(stream_parts, visible_parts, emit)
                break

            # Streaming defer: a bare ``<|channel>`` (or ``<|channel>thought``
            # without trailing newline) at the end of the source could still
            # extend to the canonical ``<|channel>thought\n`` once more
            # tokens arrive. Buffer and wait so the canonical match wins.
            if not final and marker == _OPEN_MARKER_BARE:
                suffix = source[idx:]
                if len(suffix) < len(_OPEN_MARKER) and _OPEN_MARKER.startswith(suffix):
                    self._append_text(stream_parts, visible_parts, source[pos:idx])
                    self._buffer = suffix
                    return OutputParserTokenResult(
                        stream_text="".join(stream_parts),
                        visible_text="".join(visible_parts),
                    )

            self._append_text(stream_parts, visible_parts, source[pos:idx])

            advance = len(marker)

            if marker == _OPEN_MARKER:
                # Nested open while already in a thought block: drop the stray
                # marker without re-emitting ``<think>`` to keep the structure
                # well-formed.
                if not self._in_thought:
                    stream_parts.append(_THINK_OPEN)
                    visible_parts.append(_THINK_OPEN)
                    self._in_thought = True
            elif marker == _OPEN_MARKER_BARE:
                # Defensive fallback for malformed opens: ``<|channel>thought``
                # without the trailing newline, or a bare ``<|channel>`` with
                # an unrecognised channel name. Treat as a thought open and
                # absorb the optional ``thought`` keyword and newline so they
                # don't leak as visible text.
                if not self._in_thought:
                    stream_parts.append(_THINK_OPEN)
                    visible_parts.append(_THINK_OPEN)
                    self._in_thought = True
                after = idx + advance
                if source.startswith("thought\n", after):
                    advance += len("thought\n")
                elif source.startswith("thought", after):
                    advance += len("thought")
            elif marker == _CLOSE_MARKER:
                # Stray close outside a thought block: drop silently to keep
                # the marker out of visible content.
                if self._in_thought:
                    stream_parts.append(_THINK_CLOSE)
                    visible_parts.append(_THINK_CLOSE)
                    self._in_thought = False
            # _TURN_END_MARKER, _TOOL_RESPONSE_OPEN / _CLOSE: silent drop.

            pos = idx + advance

        return OutputParserTokenResult(
            stream_text="".join(stream_parts),
            visible_text="".join(visible_parts),
        )

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            text = self._detokenizer.last_segment
        else:
            text = self._tokenizer.decode([token_id])
        return self._consume_text(text)

    def process_text(self, text: str) -> OutputParserTokenResult:
        """Process an already-detokenized text segment.

        Engines that emit text segments instead of token ids (the serial
        diffusion lane detokenizes inside ``stream_diffusion_generate``)
        feed their output through this entry point so protocol markers
        are handled identically to the token-id path.  Switches the
        session to text mode so ``finalize`` does not flush the unused
        token detokenizer.
        """
        self._text_mode = True
        if not text:
            return OutputParserTokenResult(stream_text="", visible_text="")
        return self._consume_text(text)

    def finalize(self) -> OutputParserFinalizeResult:
        text = ""
        if self._detokenizer is not None and not self._text_mode:
            self._detokenizer.finalize()
            text = self._detokenizer.last_segment

        token_result = self._consume_text(text, final=True)

        stream_text = token_result.stream_text
        visible_text = token_result.visible_text

        if self._buffer:
            stream_text += self._buffer
            visible_text += self._buffer
            self._buffer = ""

        if self._in_thought:
            stream_text += _THINK_CLOSE
            visible_text += _THINK_CLOSE
            self._in_thought = False

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
        )


_CHANNEL_OPEN_TOKEN = "<|channel>"
_CHANNEL_CLOSE_TOKEN = "<channel|>"
_TURN_END_TOKEN = "<turn|>"
_TOOL_CALL_OPEN_TOKEN = "<|tool_call>"
_TOOL_CALL_CLOSE_TOKEN = "<tool_call|>"


def _resolve_token_id(tokenizer: Any, token: str) -> int | None:
    """Look up a special-token ID by string, rejecting UNK collisions.

    HF fast tokenizers return ``unk_token_id`` (a positive int) when a token
    isn't in the vocabulary. Without filtering that out, a model whose
    tokenizer is missing the Gemma 4 marker strings would collapse every
    marker onto the same UNK id and route every generated UNK token into the
    state machine. Return ``None`` in that case so the parser treats the
    marker as absent and degrades to a plain passthrough.
    """
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if not isinstance(tid, int) or tid < 0:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and tid == unk:
        return None
    return tid


def _normalize_parse_response_tool_calls(parsed: dict) -> list[dict[str, str]]:
    """Convert ``tokenizer.parse_response`` tool_calls to scheduler shape.

    parse_response (Gemma 4 schema) returns::

        [{"type": "function",
          "function": {"name": str, "arguments": dict}}]

    The scheduler and OpenAI/Anthropic response builders expect::

        [{"name": str, "arguments": <json-string>}]
    """
    raw = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    if not raw:
        return []
    normalized: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        func = entry.get("function")
        if not isinstance(func, dict):
            continue
        name = func.get("name") or ""
        arguments = func.get("arguments", {})
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments)
            except (TypeError, ValueError):
                arguments = "{}"
        normalized.append({"name": str(name), "arguments": arguments})
    return normalized


class Gemma4OutputParserSession:
    """Token-ID streaming parser for Gemma 4.

    Keys off the special-token IDs for ``<|channel>``, ``<channel|>``,
    ``<turn|>``, ``<|tool_call>``, ``<tool_call|>`` rather than the decoded
    text. That makes it immune to collisions where reasoning content contains
    the literal marker strings — special and regular token IDs live in disjoint
    spaces, so a state transition can't be forged by regular text.

    After ``<|channel>`` the model emits ``thought\\n`` as regular tokens;
    those are buffered in a header state until the first newline, then
    ``<think>`` is emitted and subsequent thought content streams through.
    If generation stops before the newline arrives (max-tokens mid-header),
    finalize emits the buffered content wrapped in think tags so nothing
    is silently dropped.

    At finalize, the full raw token stream is decoded and handed to
    ``tokenizer.parse_response`` so tool calls extracted from the
    ``response_schema`` ``x-regex-iterator`` / ``x-parser`` chain flow through
    as ``OutputParserFinalizeResult.tool_calls``. Works only when the tokenizer
    ships a ``response_schema``; otherwise tool-call extraction is a no-op and
    streaming behavior is unchanged.
    """

    _STATE_NORMAL = 0
    _STATE_HEADER = 1
    _STATE_THOUGHT = 2

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

        self._channel_open_id = _resolve_token_id(tokenizer, _CHANNEL_OPEN_TOKEN)
        self._channel_close_id = _resolve_token_id(tokenizer, _CHANNEL_CLOSE_TOKEN)
        self._turn_end_id = _resolve_token_id(tokenizer, _TURN_END_TOKEN)
        self._tool_call_open_id = _resolve_token_id(tokenizer, _TOOL_CALL_OPEN_TOKEN)
        self._tool_call_close_id = _resolve_token_id(tokenizer, _TOOL_CALL_CLOSE_TOKEN)

        self._marker_ids = {
            tid
            for tid in (
                self._channel_open_id,
                self._channel_close_id,
                self._turn_end_id,
                self._tool_call_open_id,
                self._tool_call_close_id,
            )
            if tid is not None
        }

        missing = [
            name
            for name, tid in (
                (_CHANNEL_OPEN_TOKEN, self._channel_open_id),
                (_CHANNEL_CLOSE_TOKEN, self._channel_close_id),
                (_TURN_END_TOKEN, self._turn_end_id),
            )
            if tid is None
        ]
        if missing:
            logger.warning(
                "gemma4 parser: tokenizer missing marker tokens %s; streaming "
                "marker suppression will degrade to passthrough",
                ", ".join(missing),
            )

        self._state = self._STATE_NORMAL
        self._in_tool_call = False
        self._header_buffer = ""
        self._raw_token_ids: list[int] = []

        self._detokenizer = create_streaming_detokenizer(tokenizer)

        if self._detokenizer is not None:
            self._detokenizer.reset()

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        return self._tokenizer.decode([token_id])

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        self._raw_token_ids.append(token_id)

        if token_id in self._marker_ids:
            return self._handle_marker(token_id)

        text = self._decode_token(token_id)
        if not text:
            return OutputParserTokenResult()

        if self._in_tool_call:
            return OutputParserTokenResult()

        if self._state == self._STATE_HEADER:
            self._header_buffer += text
            newline_idx = self._header_buffer.find("\n")
            if newline_idx < 0:
                return OutputParserTokenResult()
            after = self._header_buffer[newline_idx + 1 :]
            self._header_buffer = ""
            self._state = self._STATE_THOUGHT
            emitted = _THINK_OPEN + after
            return OutputParserTokenResult(stream_text=emitted, visible_text=emitted)

        return OutputParserTokenResult(stream_text=text, visible_text=text)

    def _handle_marker(self, token_id: int) -> OutputParserTokenResult:
        # Tool-call bracketing trumps channel state. Entering a tool call
        # while we have a pending think block closes it cleanly; channel
        # markers are ignored entirely while inside a tool call so weird
        # model interleavings can't corrupt the state machine.
        if token_id == self._tool_call_open_id:
            stream = visible = ""
            if self._state == self._STATE_THOUGHT:
                stream = visible = _THINK_CLOSE
            self._state = self._STATE_NORMAL
            self._header_buffer = ""
            self._in_tool_call = True
            return OutputParserTokenResult(stream_text=stream, visible_text=visible)

        if token_id == self._tool_call_close_id:
            self._in_tool_call = False
            return OutputParserTokenResult()

        if self._in_tool_call:
            return OutputParserTokenResult()

        if token_id == self._channel_open_id:
            # Consecutive ``<|channel>`` while already in THOUGHT: close the
            # prior block first so the emitted stream stays well-formed.
            stream = visible = ""
            if self._state == self._STATE_THOUGHT:
                stream = visible = _THINK_CLOSE
            self._state = self._STATE_HEADER
            self._header_buffer = ""
            return OutputParserTokenResult(stream_text=stream, visible_text=visible)

        if token_id == self._channel_close_id:
            if self._state == self._STATE_THOUGHT:
                self._state = self._STATE_NORMAL
                return OutputParserTokenResult(
                    stream_text=_THINK_CLOSE,
                    visible_text=_THINK_CLOSE,
                )
            if self._state == self._STATE_HEADER:
                # Header never saw its newline terminator. Emit whatever
                # was buffered as thought content rather than dropping it.
                body = _THINK_OPEN + self._header_buffer + _THINK_CLOSE
                self._state = self._STATE_NORMAL
                self._header_buffer = ""
                return OutputParserTokenResult(stream_text=body, visible_text=body)
            return OutputParserTokenResult()

        if token_id == self._turn_end_id:
            return OutputParserTokenResult()

        # Unreachable: _marker_ids only contains the five ids resolved above.
        return OutputParserTokenResult()

    def _finalize_detokenizer(self) -> str:
        if self._detokenizer is None:
            return ""
        try:
            self._detokenizer.finalize()
        except Exception:
            return ""
        return self._detokenizer.last_segment or ""

    def _extract_tool_calls(self) -> list[dict[str, str]]:
        parse_response = getattr(self._tokenizer, "parse_response", None)
        if parse_response is None:
            return []
        if not getattr(self._tokenizer, "response_schema", None):
            return []
        if not self._raw_token_ids:
            return []
        # Must keep special tokens in the decoded string so parse_response's
        # x-regex can find the <|tool_call>...<tool_call|> brackets. If the
        # tokenizer wrapper doesn't accept the kwarg, skipping specials would
        # silently produce empty tool_calls — surface it instead of guessing.
        try:
            full_text = self._tokenizer.decode(
                self._raw_token_ids, skip_special_tokens=False
            )
        except Exception:
            logger.warning(
                "gemma4: tokenizer.decode(skip_special_tokens=False) failed; "
                "tool-call extraction skipped",
                exc_info=True,
            )
            return []
        try:
            parsed = parse_response(full_text)
        except Exception:
            logger.warning(
                "gemma4: tokenizer.parse_response() raised; "
                "tool-call extraction skipped",
                exc_info=True,
            )
            return []
        if not isinstance(parsed, dict):
            return []
        return _normalize_parse_response_tool_calls(parsed)

    def finalize(self) -> OutputParserFinalizeResult:
        trailing = self._finalize_detokenizer()
        stream_text = ""
        visible_text = ""

        if trailing:
            if self._in_tool_call:
                pass  # suppressed
            elif self._state == self._STATE_HEADER:
                self._header_buffer += trailing
                newline_idx = self._header_buffer.find("\n")
                if newline_idx >= 0:
                    after = self._header_buffer[newline_idx + 1 :]
                    self._header_buffer = ""
                    self._state = self._STATE_THOUGHT
                    stream_text += _THINK_OPEN + after
                    visible_text += _THINK_OPEN + after
            else:
                stream_text += trailing
                visible_text += trailing

        # Unterminated header (e.g. max-tokens hit before the header newline
        # arrived): emit whatever is buffered wrapped in think tags rather
        # than dropping it silently. Better to show partial reasoning than
        # lose it.
        if self._state == self._STATE_HEADER:
            if self._header_buffer:
                body = _THINK_OPEN + self._header_buffer + _THINK_CLOSE
                stream_text += body
                visible_text += body
            self._header_buffer = ""
            self._state = self._STATE_NORMAL

        if self._state == self._STATE_THOUGHT:
            stream_text += _THINK_CLOSE
            visible_text += _THINK_CLOSE
            self._state = self._STATE_NORMAL

        tool_calls = self._extract_tool_calls()
        finish_reason = "tool_calls" if tool_calls else None

        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
