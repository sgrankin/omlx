# SPDX-License-Identifier: Apache-2.0
"""Tests for protocol-specific output parser sessions."""

from __future__ import annotations

from openai_harmony import load_harmony_encoding

from omlx.adapter.gemma4 import (
    Gemma4OutputParserSession,
    _Gemma4LegacyOutputParserSession,
)
from omlx.adapter.output_parser import detect_output_parser


class FakeDetokenizer:
    def __init__(self, decode_one):
        self._decode_one = decode_one
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token_id: int):
        self.last_segment = self._decode_one(token_id)

    def finalize(self):
        self.last_segment = ""


class GemmaTokenizer:
    """Simple tokenizer used by the legacy text-based parser tests.

    Does not implement ``convert_tokens_to_ids`` — the legacy parser works on
    decoded text, so marker-id resolution is not needed.
    """

    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._token_map[token_id])

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(self._token_map[token_id] for token_id in token_ids)


class TokenIdGemmaTokenizer:
    """Tokenizer that treats marker strings as distinct special token IDs.

    Models the real Gemma 4 tokenizer's behavior: marker strings like
    ``<|channel>`` correspond to dedicated single-token IDs, and the
    decoded text for a regular token is independent of whether it happens
    to spell out a marker.

    Args:
        token_map: regular-token id -> decoded text.
        marker_ids: marker string -> token id (must not overlap with token_map).
        response_schema: Optional object to expose as ``response_schema``
            attribute (truthy value enables the finalize tool-call path).
        parse_response_fn: Optional callable(full_text) -> dict mimicking
            transformers' ``parse_response``.
    """

    def __init__(
        self,
        token_map: dict[int, str],
        marker_ids: dict[str, int],
        *,
        response_schema=None,
        parse_response_fn=None,
    ):
        overlap = set(token_map) & set(marker_ids.values())
        if overlap:
            raise ValueError(f"token_map / marker_ids overlap on ids {overlap}")
        self._token_map = token_map
        self._marker_ids = marker_ids
        self._id_to_marker = {tid: name for name, tid in marker_ids.items()}
        self.response_schema = response_schema
        self._parse_response_fn = parse_response_fn

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._marker_ids.get(token, -1)

    def _decode_one(self, token_id: int) -> str:
        if token_id in self._id_to_marker:
            return self._id_to_marker[token_id]
        return self._token_map[token_id]

    @property
    def detokenizer(self):
        return FakeDetokenizer(self._decode_one)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        parts = []
        for tid in token_ids:
            if tid in self._id_to_marker:
                if not skip_special_tokens:
                    parts.append(self._id_to_marker[tid])
                continue
            parts.append(self._token_map[tid])
        return "".join(parts)

    def parse_response(self, text: str):
        if self._parse_response_fn is None:
            raise AttributeError("parse_response not configured on test tokenizer")
        return self._parse_response_fn(text)


class HarmonyTokenizer:
    def __init__(self, encoding):
        self._encoding = encoding

    def convert_tokens_to_ids(self, token: str) -> int:
        ids = self._encoding.encode(token, allowed_special="all")
        return ids[0] if ids else -1

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self._encoding.decode(token_ids)

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._encoding.decode([token_id]))


# Canonical Gemma 4 marker ids used by the token-ID parser tests.
_GEMMA4_MARKER_IDS = {
    "<|channel>": 100,
    "<channel|>": 101,
    "<turn|>": 106,
    "<|tool_call>": 48,
    "<tool_call|>": 49,
}


class TestGemma4OutputParserSession:
    """Tests for the new token-ID based streaming parser."""

    def _run(self, tokenizer, token_ids):
        session = Gemma4OutputParserSession(tokenizer)
        stream = []
        visible = []
        for tid in token_ids:
            r = session.process_token(tid)
            stream.append(r.stream_text)
            visible.append(r.visible_text)
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)
        return "".join(stream), "".join(visible), final

    def test_normal_reasoning_block(self):
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "\n",
                202: "step 1",
                203: "answer",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, visible, _ = self._run(tok, [100, 200, 201, 202, 101, 203, 106])
        assert stream == "<think>step 1</think>answer"
        assert visible == stream
        assert "<|channel>" not in stream
        assert "<channel|>" not in stream
        assert "<turn|>" not in stream

    def test_empty_thought_block(self):
        tok = TokenIdGemmaTokenizer(
            token_map={200: "thought", 201: "\n", 203: "answer"},
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(tok, [100, 200, 201, 101, 203])
        assert stream == "<think></think>answer"

    def test_collision_literal_marker_in_thought_content(self):
        """Regular-token text that spells out ``<channel|>`` must not flip state.

        This is the bug the new parser is designed to fix. Token 204 decodes
        to the literal string ``<channel|>`` but its ID (204) is distinct
        from the special close-marker ID (101), so it must flow through as
        thought content.
        """
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "\n",
                202: "I should note that ",
                204: "<channel|>",
                205: " is a close marker",
                206: "the real answer",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, visible, _ = self._run(
            tok, [100, 200, 201, 202, 204, 205, 101, 206, 106]
        )
        assert stream == (
            "<think>I should note that <channel|> is a close marker</think>"
            "the real answer"
        )
        assert visible == stream

    def test_suppresses_turn_end_marker(self):
        tok = TokenIdGemmaTokenizer(
            token_map={200: "thought", 201: "\n", 202: "r", 203: "answer"},
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(tok, [100, 200, 201, 202, 101, 203, 106])
        assert "<turn|>" not in stream
        assert stream == "<think>r</think>answer"

    def test_swallows_tool_call_markup_during_stream(self):
        """Between ``<|tool_call>`` and ``<tool_call|>``, all tokens are hidden."""
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "\n",
                202: "r",
                203: "done",
                300: "call:foo{\"x\":1}",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(
            tok, [100, 200, 201, 202, 101, 48, 300, 49, 203, 106]
        )
        assert stream == "<think>r</think>done"
        assert "call:foo" not in stream

    def test_tool_calls_extracted_at_finalize(self):
        def fake_parse(text):
            assert "<|tool_call>" in text
            return {
                "role": "assistant",
                "thinking": "r",
                "content": "done",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "foo",
                            "arguments": {"x": 1},
                        },
                    }
                ],
            }

        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "\n",
                202: "r",
                203: "done",
                300: "call:foo{\"x\":1}",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
            response_schema={"type": "object"},  # truthy
            parse_response_fn=fake_parse,
        )
        _, _, final = self._run(
            tok, [100, 200, 201, 202, 101, 48, 300, 49, 203, 106]
        )
        assert final.tool_calls == [{"name": "foo", "arguments": '{"x": 1}'}]
        assert final.finish_reason == "tool_calls"

    def test_tool_calls_noop_without_response_schema(self):
        tok = TokenIdGemmaTokenizer(
            token_map={200: "thought", 201: "\n", 202: "r", 203: "done"},
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        _, _, final = self._run(tok, [100, 200, 201, 202, 101, 203, 106])
        assert final.tool_calls == []
        assert final.finish_reason is None

    def test_missing_marker_ids_degrade_gracefully(self):
        """Tokenizer without the marker tokens should not crash.

        The parser returns regular text; reasoning extraction is off.
        """
        tok = TokenIdGemmaTokenizer(
            token_map={1: "plain answer"},
            marker_ids={},  # no marker IDs resolved
        )
        stream, _, _ = self._run(tok, [1])
        assert stream == "plain answer"

    def test_unk_collision_does_not_activate_markers(self):
        """HF fast tokenizers return ``unk_token_id`` (a positive int) when
        a token isn't in the vocabulary. Without filtering, every marker
        would collapse to the same UNK id and every UNK token generated by
        the model would be misread as a state transition. Verify the resolver
        rejects UNK so the parser degrades to passthrough instead.
        """

        class UnkTokenizer(TokenIdGemmaTokenizer):
            unk_token_id = 3

            def convert_tokens_to_ids(self, token: str) -> int:
                return self.unk_token_id  # every lookup returns UNK

        tok = UnkTokenizer(
            token_map={1: "hello ", 2: "world", 3: "<unk>"},
            marker_ids={},
        )
        stream, _, _ = self._run(tok, [1, 3, 2])
        assert stream == "hello <unk>world"

    def test_tool_call_suppresses_interleaved_channel_markers(self):
        """Channel markers inside a tool-call block are ignored so the state
        machine stays consistent even if the model emits weird interleavings.
        """
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "\n",
                202: "r",
                203: "done",
                300: "tool body",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(
            tok,
            [
                100, 200, 201, 202,  # <|channel>thought\nr
                101,                  # <channel|>  → </think>
                48,                   # <|tool_call>
                100, 300, 101,        # interleaved <|channel>..<channel|>  (ignored)
                49,                   # <tool_call|>
                203, 106,
            ],
        )
        assert stream == "<think>r</think>done"
        assert "tool body" not in stream

    def test_tool_call_open_closes_pending_thought(self):
        """``<|tool_call>`` mid-thought closes the open ``<think>`` block."""
        tok = TokenIdGemmaTokenizer(
            token_map={200: "thought", 201: "\n", 202: "r", 203: "done"},
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(
            tok,
            [100, 200, 201, 202, 48, 49, 203, 106],
        )
        assert stream == "<think>r</think>done"

    def test_consecutive_channel_markers_close_prior_think(self):
        """Two ``<|channel>`` in a row emit ``</think>`` before starting
        the second block."""
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "\n",
                202: "first",
                203: "second",
                204: "answer",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(
            tok,
            [
                100, 200, 201, 202,
                100, 200, 201, 203,
                101, 204, 106,
            ],
        )
        assert stream == "<think>first</think><think>second</think>answer"
        assert stream.count("<think>") == stream.count("</think>")

    def test_unterminated_header_at_finalize_emits_buffered_content(self):
        """Generation cut off mid-header does not silently lose content."""
        tok = TokenIdGemmaTokenizer(
            token_map={200: "thought", 202: "partial"},
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(tok, [100, 200, 202])
        assert "<think>" in stream
        assert "</think>" in stream
        assert "partial" in stream

    def test_extract_tool_calls_logs_on_parse_response_failure(self, caplog):
        """parse_response raising surfaces as a WARNING, not a silent empty."""
        import logging

        def boom(_text):
            raise RuntimeError("schema regex exploded")

        tok = TokenIdGemmaTokenizer(
            token_map={200: "thought", 201: "\n", 202: "r", 203: "done"},
            marker_ids=_GEMMA4_MARKER_IDS,
            response_schema={"type": "object"},
            parse_response_fn=boom,
        )
        with caplog.at_level(logging.WARNING, logger="omlx.adapter.gemma4"):
            _, _, final = self._run(
                tok, [100, 200, 201, 202, 101, 203, 106]
            )
        assert final.tool_calls == []
        assert any(
            "parse_response" in rec.message for rec in caplog.records
        )


class TestGemma4LegacyOutputParserSession:
    """Tests for the original text-based parser, kept behind the legacy flag."""

    def test_normal_reasoning_block(self):
        token_map = {
            1: "<|channel>",
            2: "thought\n",
            3: "reasoning",
            4: "<channel|>",
            5: "final answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = _Gemma4LegacyOutputParserSession(tokenizer)

        stream = []
        visible = []
        for token_id in [1, 2, 3, 4, 5]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        full_stream = "".join(stream)
        full_visible = "".join(visible)

        assert full_stream == "<think>reasoning</think>final answer"
        assert full_visible == full_stream
        assert "<|channel>" not in full_stream
        assert "<channel|>" not in full_stream

    def test_empty_thought_block(self):
        token_map = {
            1: "<|channel>thought\n",
            2: "<channel|>",
            3: "answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = _Gemma4LegacyOutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        assert "".join(parts) == "<think></think>answer"

    def test_partial_marker_across_tokens(self):
        token_map = {
            1: "<|chan",
            2: "nel>thought\nstep 1",
            3: " and step 2<chan",
            4: "nel|>",
            5: "done",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = _Gemma4LegacyOutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>step 1 and step 2</think>done"
        assert "<|channel>thought" not in text
        assert "<channel|>" not in text

    def test_malformed_channel_header_still_strips_from_visible(self):
        """Legacy loose-marker fallback: when the strict marker doesn't match,
        fall back to matching just ``<|channel>`` so visible output stays clean
        even though the reasoning block contains garbage metadata."""
        token_map = {
            1: "<|channel>thought|thought\n ",
            2: "<channel|>",
            3: "In the OpenAI Chat Completions API, images are handled correctly.",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = _Gemma4LegacyOutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert "In the OpenAI Chat Completions API" in text
        assert "<|channel>" not in text
        assert "<channel|>" not in text
        assert text == (
            "<think>|thought\n </think>"
            "In the OpenAI Chat Completions API, images are handled correctly."
        )

    def test_suppresses_turn_end_marker(self):
        token_map = {
            1: "<|channel>thought\n",
            2: "reasoning",
            3: "<channel|>",
            4: "answer",
            5: "<turn|>",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = _Gemma4LegacyOutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5]:
            result = session.process_token(token_id)
            parts.append(result.stream_text)
            assert "<turn|>" not in result.stream_text
            assert "<turn|>" not in result.visible_text
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>reasoning</think>answer"
        assert "<turn|>" not in text

    def test_stray_close_marker_outside_thought_dropped(self):
        """A bare ``<channel|>`` after the thought block already closed must
        not leak into visible content. Models occasionally emit one in long
        multi-turn contexts and the SDK rejects it as raw markup."""
        token_map = {
            1: "<|channel>thought\n",
            2: "reasoning",
            3: "<channel|>",
            4: "answer",
            5: "<channel|>",
            6: "more",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>reasoning</think>answermore"
        assert "<channel|>" not in text

    def test_stray_open_marker_inside_thought_dropped(self):
        """A nested ``<|channel>thought\\n`` while already inside a thought
        block must not re-emit ``<think>``. The block stays open until the
        first matching close marker."""
        token_map = {
            1: "<|channel>thought\n",
            2: "step 1",
            3: "<|channel>thought\n",
            4: "step 2",
            5: "<channel|>",
            6: "answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>step 1step 2</think>answer"
        assert text.count("<think>") == 1
        assert text.count("</think>") == 1

    def test_tool_call_markers_pass_through(self):
        """Tool-call markup must reach the buffered output text untouched so
        ``parse_tool_calls`` can extract the call. ``ToolCallStreamFilter``
        downstream is responsible for removing it from stream deltas."""
        token_map = {
            1: "<|channel>thought\n",
            2: "calling",
            3: "<channel|>",
            4: "<|tool_call>",
            5: "call:bash{cmd:ls}",
            6: "<tool_call|>",
            7: "done",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        stream_parts = []
        visible_parts = []
        for token_id in [1, 2, 3, 4, 5, 6, 7]:
            result = session.process_token(token_id)
            stream_parts.append(result.stream_text)
            visible_parts.append(result.visible_text)
        final = session.finalize()
        stream_parts.append(final.stream_text)
        visible_parts.append(final.visible_text)

        stream_text = "".join(stream_parts)
        visible_text = "".join(visible_parts)
        assert stream_text == visible_text
        assert "<|tool_call>" in stream_text
        assert "<tool_call|>" in stream_text
        assert "call:bash{cmd:ls}" in stream_text


class TestOutputParserFactory:
    def test_detects_gemma4(self):
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/gemma-4b",
            tokenizer,
            {"model_type": "gemma4"},
        )

        assert factory is not None
        assert factory.kind == "gemma4"

    def test_gemma4_flag_selects_legacy_parser(self, monkeypatch):
        monkeypatch.setenv("OMLX_GEMMA4_PARSER", "legacy")
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/gemma-4b",
            tokenizer,
            {"model_type": "gemma4"},
        )
        assert factory is not None
        session = factory.create_session(tokenizer)
        assert isinstance(session, _Gemma4LegacyOutputParserSession)

    def test_gemma4_default_selects_new_parser(self, monkeypatch):
        monkeypatch.delenv("OMLX_GEMMA4_PARSER", raising=False)
        tokenizer = TokenIdGemmaTokenizer(token_map={}, marker_ids=_GEMMA4_MARKER_IDS)
        factory = detect_output_parser(
            "google/gemma-4b",
            tokenizer,
            {"model_type": "gemma4"},
        )
        assert factory is not None
        session = factory.create_session(tokenizer)
        assert isinstance(session, Gemma4OutputParserSession)

    def test_harmony_wrapper_regression(self):
        encoding = load_harmony_encoding("HarmonyGptOss")
        tokenizer = HarmonyTokenizer(encoding)
        factory = detect_output_parser(
            "gpt-oss-20b",
            tokenizer,
            {"model_type": "gpt_oss"},
        )

        assert factory is not None
        assert factory.kind == "harmony"

        session = factory.create_session(tokenizer)
        tokens = encoding.encode(
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer<|return|>",
            allowed_special="all",
        )

        stream = []
        visible = []
        saw_stop = False
        for token in tokens:
            result = session.process_token(token)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            saw_stop = saw_stop or result.is_stop
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        assert saw_stop is True
        assert "<think>" in "".join(stream)
        assert "</think>" in "".join(stream)
        assert "".join(visible) == "Answer"

    def test_harmony_non_streaming_preserves_reasoning(self):
        """Non-streaming output_text retains analysis-channel reasoning."""
        from omlx.api.thinking import extract_thinking

        encoding = load_harmony_encoding("HarmonyGptOss")
        tokenizer = HarmonyTokenizer(encoding)
        factory = detect_output_parser(
            "gpt-oss-20b",
            tokenizer,
            {"model_type": "gpt_oss"},
        )
        session = factory.create_session(tokenizer)

        tokens = encoding.encode(
            "<|channel|>analysis<|message|>Let me think about this<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Four<|return|>",
            allowed_special="all",
        )

        visible_parts = []
        for token in tokens:
            result = session.process_token(token)
            visible_parts.append(result.visible_text)

        final = session.finalize()
        visible_parts.append(final.visible_text)

        # Mirror scheduler aggregation: prepend any parser-provided prefix
        # to the accumulated visible_text before exposing as output_text.
        prefix = getattr(final, "output_text_prefix", "")
        output_text = prefix + "".join(visible_parts)

        thinking, content = extract_thinking(output_text)
        assert thinking == "Let me think about this"
        assert content == "Four"
