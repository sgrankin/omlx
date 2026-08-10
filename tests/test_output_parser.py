# SPDX-License-Identifier: Apache-2.0
"""Tests for protocol-specific output parser sessions."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from omlx.adapter.gemma4 import (
    Gemma4OutputParserSession,
    _Gemma4LegacyOutputParserSession,
)
from omlx.adapter.harmony import load_harmony_gpt_oss_encoding
from omlx.adapter.output_parser import (
    OutputParserFactory,
    detect_output_parser,
    select_text_parser_session,
)


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


class CohereTokenizer:
    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._token_map[token_id])

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(self._token_map[token_id] for token_id in token_ids)


class DeepSeekV4Tokenizer(CohereTokenizer):
    has_tool_calling = True
    tool_call_start = "<｜DSML｜tool_calls>"
    tool_call_end = "</｜DSML｜tool_calls>"

    def tool_parser(self, text: str, tools=None):
        from omlx.patches.deepseek_v4.tool_parser_v4 import parse_tool_call

        return parse_tool_call(text, tools)


class BailingHybridTokenizer(CohereTokenizer):
    _token_ids = {
        "<role>": 157151,
        "</role>": 157152,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_ids.get(token, -1)

    def encode(self, text: str, add_special_tokens: bool = False):
        token_id = self._token_ids.get(text)
        return [token_id] if token_id is not None else []


class _FakeMelodyOptions:
    def cmd4(self):
        return self

    def stream_tool_actions(self):
        return self


class _FakeMelodyFilter:
    def __init__(self, options):
        self.options = options

    def write_decoded(self, decoded_text: str):
        if decoded_text.startswith("R:"):
            return SimpleNamespace(
                content=None,
                reasoning=decoded_text[2:],
                tool_calls=[],
            )
        if decoded_text.startswith("C:"):
            return SimpleNamespace(
                content=decoded_text[2:],
                reasoning=None,
                tool_calls=[],
            )
        if decoded_text.startswith("T1"):
            tool_call = SimpleNamespace(
                index=0,
                id="call_",
                name="look",
                arguments='{"q"',
            )
            return SimpleNamespace(content=None, reasoning=None, tool_calls=[tool_call])
        if decoded_text.startswith("T2"):
            tool_call = SimpleNamespace(
                index=0,
                id="1",
                name="up",
                arguments=':"x"}',
            )
            return SimpleNamespace(content=None, reasoning=None, tool_calls=[tool_call])
        return SimpleNamespace(content=None, reasoning=None, tool_calls=[])

    def flush_partials(self):
        return SimpleNamespace(content=None, reasoning=None, tool_calls=[])


def _install_fake_melody(monkeypatch):
    module = types.ModuleType("cohere_melody")
    module.PyFilter = _FakeMelodyFilter
    module.PyFilterOptions = _FakeMelodyOptions
    monkeypatch.setitem(sys.modules, "cohere_melody", module)


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _spm_decoder():
    return {
        "type": "Sequence",
        "decoders": [
            {
                "type": "Replace",
                "pattern": {"String": "\u2581"},
                "content": " ",
            },
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }


class ByteFallbackTokenizer:
    clean_up_tokenization_spaces = False
    vocab = {
        "<pad>": 0,
        "<0xEC>": 1,
        "<0x9E>": 2,
        "<0xA0>": 3,
    }

    def decode(self, token_ids, skip_special_tokens: bool = True):
        table = {
            0: b"",
            1: bytes([0xEC]),
            2: bytes([0x9E]),
            3: bytes([0xA0]),
        }
        raw = b"".join(table[token_id] for token_id in token_ids)
        if not raw:
            return ""
        if raw == bytes([0xEC, 0x9E, 0xA0]):
            return "\uc7a0"
        return "\ufffd" * sum(1 for token_id in token_ids if token_id != 0)


class TestBailingHybridOutputParserSession:
    def test_role_boundary_tokens_are_suppressed(self):
        tokenizer = BailingHybridTokenizer(
            {
                1: "Now I have ",
                2: "fixed it.",
                157151: "<role>",
                157152: "</role>",
            }
        )
        factory = detect_output_parser(
            "Ling-3.0-flash-mxfp4",
            tokenizer,
            {"model_type": "bailing_hybrid"},
        )

        assert factory is not None
        assert factory.kind == "bailing_hybrid"
        session = factory.create_session(tokenizer)
        results = [
            session.process_token(token_id)
            for token_id in (1, 157152, 2, 157151)
        ]
        final = session.finalize()

        assert "".join(result.stream_text for result in results) == (
            "Now I have fixed it."
        )
        assert "".join(result.visible_text for result in results) == (
            "Now I have fixed it."
        )
        assert results[1].record_token is True
        assert results[3].record_token is True
        assert final.stream_text == ""
        assert final.visible_text == ""

    def test_fragmented_tool_protocol_is_hidden_and_parsed(self):
        tokenizer = BailingHybridTokenizer(
            {
                1: "Before ",
                2: "<to",
                3: "ol_call>weather<arg_",
                4: "key>city</arg_key><arg_value>Paris",
                5: "</arg_value></tool_call>",
                6: " after.",
            }
        )
        factory = detect_output_parser(
            "Ling-3.0-flash-mxfp4",
            tokenizer,
            {"model_type": "bailing_hybrid"},
        )

        assert factory is not None
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        assert factory.create_session_with_tools is not None
        session = factory.create_session_with_tools(tokenizer, tools)
        results = [session.process_token(token_id) for token_id in range(1, 7)]
        final = session.finalize()
        streamed = "".join(result.stream_text for result in results)
        visible = "".join(result.visible_text for result in results)

        assert streamed + final.stream_text == "Before  after."
        assert visible + final.visible_text == "Before  after."
        assert all(
            marker not in streamed + final.stream_text
            for marker in ("<tool_call>", "<arg_key>", "<arg_value>")
        )
        assert final.finish_reason == "tool_calls"
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["name"] == "weather"
        assert json.loads(final.tool_calls[0]["arguments"]) == {"city": "Paris"}

    def test_tool_calls_use_request_schema_and_registered_names(self):
        tokenizer = BailingHybridTokenizer(
            {
                1: (
                    "<tool_call>weather<arg_key>code</arg_key>"
                    "<arg_value>123</arg_value></tool_call>"
                    "<tool_call>unknown<arg_key>x</arg_key>"
                    "<arg_value>1</arg_value></tool_call>"
                )
            }
        )
        factory = detect_output_parser(
            "Ling-3.0-flash-mxfp4",
            tokenizer,
            {"model_type": "bailing_hybrid"},
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                },
            }
        ]

        assert factory is not None
        assert factory.create_session_with_tools is not None
        session = factory.create_session_with_tools(tokenizer, tools)
        session.process_token(1)
        final = session.finalize()

        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["name"] == "weather"
        assert json.loads(final.tool_calls[0]["arguments"]) == {"code": "123"}

    def test_tool_protocol_without_request_tools_is_not_a_tool_call(self):
        tokenizer = BailingHybridTokenizer(
            {
                1: (
                    "<tool_call>weather<arg_key>city</arg_key>"
                    "<arg_value>Paris</arg_value></tool_call>"
                )
            }
        )
        factory = detect_output_parser(
            "Ling-3.0-flash-mxfp4",
            tokenizer,
            {"model_type": "bailing_hybrid"},
        )

        assert factory is not None
        session = factory.create_session(tokenizer)
        session.process_token(1)
        final = session.finalize()

        assert final.tool_calls == []
        assert final.finish_reason is None

class TestCohere2MoeOutputParserSession:
    def test_detects_cohere2_moe_from_model_config(self, monkeypatch):
        _install_fake_melody(monkeypatch)
        tokenizer = CohereTokenizer({1: "C:hello"})

        factory = detect_output_parser(
            "North-Mini-Code",
            tokenizer,
            {"model_type": "cohere2_moe"},
        )

        assert factory is not None
        assert factory.kind == "cohere2_moe"

    def test_streams_reasoning_as_think_block_and_visible_content(self, monkeypatch):
        _install_fake_melody(monkeypatch)
        tokenizer = CohereTokenizer(
            {
                1: "R:reasoning",
                2: "C:answer",
            }
        )
        factory = detect_output_parser(
            "North-Mini-Code",
            tokenizer,
            {"model_type": "cohere2_moe"},
        )
        session = factory.create_session(tokenizer)

        parts = []
        visible = []
        for token_id in [1, 2]:
            result = session.process_token(token_id)
            parts.append(result.stream_text)
            visible.append(result.visible_text)
        final = session.finalize()
        parts.append(final.stream_text)
        visible.append(final.visible_text)

        assert "".join(parts) == "<think>\nreasoning</think>\nanswer"
        assert "".join(visible) == "<think>\nreasoning</think>\nanswer"
        assert final.tool_calls == []
        assert final.finish_reason is None

    def test_accumulates_streamed_tool_call_deltas(self, monkeypatch):
        _install_fake_melody(monkeypatch)
        tokenizer = CohereTokenizer({1: "T1", 2: "T2"})
        factory = detect_output_parser(
            "North-Mini-Code",
            tokenizer,
            {"model_type": "cohere2_moe"},
        )
        session = factory.create_session(tokenizer)

        assert session.process_token(1).stream_text == ""
        assert session.process_token(2).stream_text == ""
        final = session.finalize()

        assert final.tool_calls == [
            {
                "id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            }
        ]
        assert final.finish_reason == "tool_calls"

    def test_literal_newline_in_arguments_is_reescaped(self, monkeypatch):
        """Melody may stream literal control chars when the model emits them inside
        JSON string values (e.g. newlines inside code arguments).  finalize() must
        re-serialize the accumulated arguments so they are valid JSON."""
        # Build a fake Melody that returns arguments containing a literal newline
        # (U+000A) inside the JSON string value, as the real model sometimes does.
        literal_newline_args = '{"path":"f.py","code":"line1\nline2"}'  # literal \n

        class _FakeMelodyFilterLiteralNewline:
            def __init__(self, options):
                pass

            def write_decoded(self, decoded_text: str):
                if decoded_text == "TC":
                    tc = SimpleNamespace(
                        index=0,
                        id="call_1",
                        name="edit",
                        arguments=literal_newline_args,
                    )
                    return SimpleNamespace(
                        content=None, reasoning=None, tool_calls=[tc]
                    )
                return SimpleNamespace(content=None, reasoning=None, tool_calls=[])

            def flush_partials(self):
                return SimpleNamespace(content=None, reasoning=None, tool_calls=[])

        import types, json as _json

        module = types.ModuleType("cohere_melody")
        module.PyFilter = _FakeMelodyFilterLiteralNewline
        module.PyFilterOptions = _FakeMelodyOptions
        monkeypatch.setitem(__import__("sys").modules, "cohere_melody", module)

        tokenizer = CohereTokenizer({"TC": "TC"})
        from omlx.adapter.output_parser import Cohere2MoeOutputParserSession

        session = Cohere2MoeOutputParserSession.__new__(Cohere2MoeOutputParserSession)
        session._tokenizer = tokenizer
        session._melody = _FakeMelodyFilterLiteralNewline(None)
        session._detokenizer = None
        session._thinking_started = False
        session._thinking_closed = False
        session._tool_calls = {}

        session.process_token("TC")
        final = session.finalize()

        assert len(final.tool_calls) == 1
        args_str = final.tool_calls[0]["arguments"]
        # Must be valid strict JSON (no literal control characters)
        parsed = _json.loads(args_str)
        assert parsed["code"] == "line1\nline2"
        # The literal newline must have been escaped
        assert "\n" not in args_str or "\\n" in args_str


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

    def test_prefilled_thought_closes_before_visible_content(self):
        """A prompt-side opener must seed the parser before generation.

        Gemma 4 tool continuations start generation inside the thought
        channel, so the generated stream carries only the body, the close
        marker, and the visible answer — no ``<|channel>`` token ever
        arrives to move the state machine into the thought state.
        """
        tok = TokenIdGemmaTokenizer(
            token_map={202: "reasoning", 203: "answer"},
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        session = Gemma4OutputParserSession(tok)
        session.notify_prefilled_thought()

        parts = []
        for token_id in [202, 101, 203]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        assert "".join(parts) == "reasoning</think>answer"

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

    def test_malformed_header_metadata_discarded(self):
        """Header tokens containing garbage before the newline are discarded.

        Regression: Gemma 4 was observed emitting
        ``<|channel>thought|thought\\n <channel|>`` instead of the canonical
        ``<|channel>thought\\n ... <channel|>``. The HEADER state buffers
        all regular tokens until the first newline and discards them as
        channel-name metadata, so garbled metadata never reaches the visible
        stream. Content after the newline (even a lone space) becomes the
        thought body.
        """
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought|thought\n ",  # whole header+leading space in one token
                201: "In the OpenAI Chat Completions API.",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, visible, _ = self._run(tok, [100, 200, 101, 201, 106])
        # Visible answer preserved verbatim.
        assert "In the OpenAI Chat Completions API." in visible
        # No raw channel markers leak through.
        assert "<|channel>" not in stream
        assert "<channel|>" not in stream
        # Garbage metadata up to \n is stripped; only the trailing space
        # survives as thought content.
        assert "thought|thought" not in stream
        assert stream == "<think> </think>In the OpenAI Chat Completions API."

    def test_malformed_header_metadata_split_across_tokens(self):
        """Garbled header split across multiple tokens still discards pre-\\n content."""
        tok = TokenIdGemmaTokenizer(
            token_map={
                200: "thought",
                201: "|thought",
                202: "\n",
                203: "answer",
            },
            marker_ids=_GEMMA4_MARKER_IDS,
        )
        stream, _, _ = self._run(tok, [100, 200, 201, 202, 101, 203, 106])
        # Everything before \n discarded as metadata; thought body empty.
        assert stream == "<think></think>answer"
        assert "thought|thought" not in stream

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
        session = _Gemma4LegacyOutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>reasoning</think>answermore"
        assert "<channel|>" not in text

    def test_prefilled_thought_closes_before_visible_content(self):
        """A prompt-side opener must seed the parser before generation.

        Gemma 4 tool continuations start generation inside the thought
        channel, so the generated stream contains only the body, close marker,
        and visible answer.
        """
        token_map = {
            1: "reasoning",
            2: "<channel|>",
            3: "answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = _Gemma4LegacyOutputParserSession(tokenizer)
        session.notify_prefilled_thought()

        parts = []
        for token_id in [1, 2, 3]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        assert "".join(parts) == "reasoning</think>answer"

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
        session = _Gemma4LegacyOutputParserSession(tokenizer)

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

    def test_spm_fallback_buffers_split_utf8(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": _spm_decoder()})
        session = _Gemma4LegacyOutputParserSession(
            ByteFallbackTokenizer(),
            model_path=tmp_path,
        )

        parts = []
        for token_id in [1, 2, 3]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "\uc7a0"
        assert "\ufffd" not in text


class TestOutputParserFactory:
    def test_detects_deepseek_v4_by_config(self):
        tokenizer = DeepSeekV4Tokenizer({1: "x"})
        factory = detect_output_parser(
            "DeepSeek-V4-Flash-oQ4e",
            tokenizer,
            {"model_type": "deepseek_v4"},
        )

        assert factory is not None
        assert factory.kind == "deepseek_v4"

    def test_deepseek_v4_stops_at_first_dsml_tool_block(self):
        tokenizer = DeepSeekV4Tokenizer(
            {
                1: "Before ",
                2: "<｜DSML｜tool",
                3: '_calls>\n<｜DSML｜invoke name="Bash">\n',
                4: '<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>\n'
                "</｜DSML｜invoke>\n",
                5: "</｜DSML｜tool_calls>",
            }
        )
        factory = detect_output_parser(
            "DeepSeek-V4-Flash-oQ4e",
            tokenizer,
            {"model_type": "deepseek_v4"},
        )
        session = factory.create_session(tokenizer)

        stream = []
        visible = []
        stop_seen = False
        for token_id in [1, 2, 3, 4, 5]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            stop_seen = stop_seen or result.is_stop
            assert result.record_token is True

        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        assert stop_seen is True
        assert "".join(stream) == "Before "
        assert "".join(visible) == "Before "
        assert final.finish_reason == "tool_calls"
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["name"] == "Bash"
        assert json.loads(final.tool_calls[0]["arguments"]) == {"command": "ls"}

    def test_deepseek_v4_drops_text_after_tool_end_in_same_token(self):
        tokenizer = DeepSeekV4Tokenizer(
            {
                1: '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="Bash">\n',
                2: '<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>\n'
                "</｜DSML｜invoke>\n",
                3: "</｜DSML｜tool_calls>\n"
                '<｜DSML｜parameter name="command" string="true">pwd</｜DSML｜parameter>',
            }
        )
        factory = detect_output_parser(
            "DeepSeek-V4-Flash-oQ4e",
            tokenizer,
            {"model_type": "deepseek_v4"},
        )
        session = factory.create_session(tokenizer)

        stream = []
        stop_seen = False
        for token_id in [1, 2, 3]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            stop_seen = stop_seen or result.is_stop

        final = session.finalize()
        stream.append(final.stream_text)

        assert stop_seen is True
        assert "".join(stream) == ""
        assert final.finish_reason == "tool_calls"
        assert len(final.tool_calls) == 1
        assert json.loads(final.tool_calls[0]["arguments"]) == {"command": "ls"}

    def test_detects_minimax_m3_by_config(self):
        tokenizer = CohereTokenizer({1: "x"})
        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )

        assert factory is not None
        assert factory.kind == "minimax_m3"

    def test_minimax_m3_parser_extracts_tool_calls(self, monkeypatch):
        module = types.ModuleType("mlx_vlm.tool_parsers.minimax_m3")

        def parse_tool_call(text):
            assert "lookup" in text
            return {"name": "lookup", "arguments": {"query": "mlx"}}

        module.parse_tool_call = parse_tool_call
        monkeypatch.setitem(sys.modules, "mlx_vlm.tool_parsers.minimax_m3", module)

        start = "]<]minimax[>[<tool_call>"
        end = "]<]minimax[>[</tool_call>"
        tokenizer = CohereTokenizer(
            {
                1: "before ",
                2: start,
                3: ']<]minimax[>[<invoke name="lookup">',
                4: "]<]minimax[>[</invoke>",
                5: end,
                6: " after",
            }
        )
        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )
        session = factory.create_session(tokenizer)

        visible = []
        stream = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
        final = session.finalize()

        assert "".join(stream) == "before  after"
        assert start not in "".join(stream)
        assert "".join(visible) + final.visible_text == "before  after"
        assert final.tool_calls == [{"name": "lookup", "arguments": '{"query":"mlx"}'}]
        assert final.finish_reason == "tool_calls"

    def test_minimax_m3_parser_normalizes_thinking_and_strips_eos(self):
        tokenizer = CohereTokenizer(
            {
                1: "<mm:think>",
                2: "reasoning",
                3: "</mm:think>",
                4: "Answer",
                5: "[e~[",
                6: "]!d~[",
            }
        )
        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )
        session = factory.create_session(tokenizer)

        stream = []
        visible = []
        stop_seen = False
        record_flags = []
        for token_id in [1, 2, 3, 4, 6, 5]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            stop_seen = stop_seen or result.is_stop
            record_flags.append(result.record_token)
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        assert "".join(stream) == "<think>reasoning</think>Answer"
        assert "".join(visible) == "<think>reasoning</think>Answer"
        assert stop_seen is True
        assert record_flags[-1] is False

    def test_minimax_m3_factory_exposes_native_thinking_markers(self):
        tokenizer = CohereTokenizer({})
        tokenizer.convert_tokens_to_ids = lambda text: {
            "[e~[": 200020,
            "<mm:think>": 200059,
            "</mm:think>": 200060,
        }.get(text, -1)
        tokenizer.unk_token_id = -1

        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )

        assert factory.thinking_start_text == "<mm:think>"
        assert factory.thinking_start_output_text == "<think>\n"
        assert factory.thinking_end_text == "</mm:think>"
        assert factory.stop_token_ids == {200020}

    def test_detects_gemma4(self):
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/gemma-4b",
            tokenizer,
            {"model_type": "gemma4"},
        )

        assert factory is not None
        assert factory.kind == "gemma4"
        assert factory.thinking_start_text == "<|channel>thought"
        assert factory.thinking_start_output_text == "<think>\n"
        assert factory.thinking_end_text == "<channel|>"

    def test_session_receives_model_path_when_provided(self, monkeypatch):
        """Since #2178 the scheduler's model_name is a display id, so the
        filesystem path must reach parser sessions via model_path."""
        import omlx.adapter.output_parser as output_parser_module

        seen = {}

        class RecordingSession:
            def __init__(self, tokenizer, model_path=None):
                seen["model_path"] = model_path

        monkeypatch.setattr(
            output_parser_module, "MiniMaxM3OutputParserSession", RecordingSession
        )
        tokenizer = CohereTokenizer({})
        tokenizer.convert_tokens_to_ids = lambda text: -1
        tokenizer.unk_token_id = -1

        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
            model_path="/models/minimax-m3",
        )
        factory.create_session(tokenizer)
        assert seen["model_path"] == "/models/minimax-m3"

    def test_session_falls_back_to_model_name_without_model_path(self, monkeypatch):
        """dflash/vlm engines pass their filesystem path as model_name and no
        model_path, so the session fallback must keep using model_name."""
        import omlx.adapter.output_parser as output_parser_module

        seen = {}

        class RecordingSession:
            def __init__(self, tokenizer, model_path=None):
                seen["model_path"] = model_path

        monkeypatch.setattr(
            output_parser_module, "MiniMaxM3OutputParserSession", RecordingSession
        )
        tokenizer = CohereTokenizer({})
        tokenizer.convert_tokens_to_ids = lambda text: -1
        tokenizer.unk_token_id = -1

        factory = detect_output_parser(
            "/models/MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )
        factory.create_session(tokenizer)
        assert seen["model_path"] == "/models/MiniMax-M3-4bit"

    def test_detects_gemma4_unified_by_config(self):
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "some-model",
            tokenizer,
            {"model_type": "gemma4_unified"},
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

    def test_gemma4_factory_provides_text_session(self, monkeypatch):
        """The diffusion text lane needs ``process_text``.

        The default token-ID session lacks it, so the factory must expose
        a text-capable session for engines that emit detokenized text.
        """
        monkeypatch.delenv("OMLX_GEMMA4_PARSER", raising=False)
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/diffusiongemma-26B-A4B-it",
            tokenizer,
            {"model_type": "diffusion_gemma"},
        )
        assert factory is not None
        assert factory.kind == "gemma4"
        assert factory.create_text_session is not None
        session = factory.create_text_session(tokenizer)
        assert isinstance(session, _Gemma4LegacyOutputParserSession)

    def test_select_text_parser_session_falls_back_for_gemma4(self, monkeypatch):
        """Default gemma4 session is token-ID only; selection must fall
        back to the legacy text session and still parse thought channels."""
        monkeypatch.delenv("OMLX_GEMMA4_PARSER", raising=False)
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/diffusiongemma-26B-A4B-it",
            tokenizer,
            {"model_type": "diffusion_gemma"},
        )
        session = select_text_parser_session(factory, tokenizer)
        assert isinstance(session, _Gemma4LegacyOutputParserSession)

        result = session.process_text("<|channel>thought\nreasoning<channel|>answer")
        text = result.visible_text + session.finalize().visible_text
        assert text == "<think>reasoning</think>answer"

    def test_select_text_parser_session_prefers_default(self):
        """When the default session is text-capable, the fallback builder
        must not be consulted at all."""

        class TextCapableSession:
            def process_token(self, token_id):
                raise NotImplementedError

            def process_text(self, text):
                raise NotImplementedError

            def finalize(self):
                raise NotImplementedError

        default_session = TextCapableSession()
        factory = OutputParserFactory(
            kind="text-default",
            create_session=lambda tokenizer: default_session,
            create_text_session=lambda tokenizer: pytest.fail(
                "fallback text session must not be used"
            ),
        )
        assert select_text_parser_session(factory, object()) is default_session

    def test_select_text_parser_session_none_without_text_session(self):
        class TokenOnlySession:
            def process_token(self, token_id):
                raise NotImplementedError

            def finalize(self):
                raise NotImplementedError

        factory = OutputParserFactory(
            kind="token-only",
            create_session=lambda tokenizer: TokenOnlySession(),
        )
        assert select_text_parser_session(factory, object()) is None

    def test_harmony_wrapper_regression(self):
        encoding = load_harmony_gpt_oss_encoding()
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

        encoding = load_harmony_gpt_oss_encoding()
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


class InklingTokenizer:
    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map
        self._reverse = {v: k for k, v in token_map.items()}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._reverse.get(token)

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self._reverse[text]] if text in self._reverse else [0, 1]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(self._token_map[token_id] for token_id in token_ids)

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._token_map[token_id])


class TestInklingOutputParserSession:
    def _factory(self, token_map):
        tokenizer = InklingTokenizer(token_map)
        factory = detect_output_parser(
            "inkling-small",
            tokenizer,
            {"model_type": "inkling_mm_model"},
        )
        assert factory is not None
        assert factory.kind == "inkling"
        return tokenizer, factory

    def _run(self, session, token_ids):
        stream, visible, stopped = [], [], False
        for token_id in token_ids:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            if result.is_stop:
                stopped = True
                break
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)
        return "".join(stream), "".join(visible), stopped, final

    def _parse_tool_call_payload(self, payload):
        token_map = {
            1: "<|content_invoke_tool_json|>",
            2: payload,
            3: "<|end_message|>",
            4: "<|content_model_end_sampling|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        _, _, stopped, final = self._run(session, [1, 2, 3, 4])
        assert stopped
        return final

    def test_thinking_then_text(self):
        token_map = {
            1: "<|content_thinking|>",
            2: "let me ",
            3: "reason",
            4: "<|end_message|>",
            5: "<|message_model|>",
            6: "<|content_text|>",
            7: "Answer",
            8: "<|content_model_end_sampling|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2, 3, 4, 5, 6, 7, 8])

        assert stream == "<think>let me reason</think>Answer"
        assert visible == stream
        assert stopped
        assert final.tool_calls == []
        assert 8 in factory.stop_token_ids

    def test_tool_call_suppressed_and_parsed(self):
        token_map = {
            1: "<|content_thinking|>",
            2: "need weather",
            3: "<|end_message|>",
            4: "<|message_model|>",
            5: "get_weather",
            6: "<|content_invoke_tool_json|>",
            7: '{"name":"get_weather","args":{"city":"Seoul"}}',
            8: "<|end_message|>",
            9: "<|content_model_end_sampling|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(
            session, [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

        assert stream == "<think>need weather</think>"
        assert visible == stream
        assert stopped
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["name"] == "get_weather"
        assert json.loads(final.tool_calls[0]["arguments"]) == {"city": "Seoul"}
        assert final.finish_reason == "tool_calls"

    def test_tool_call_accepts_json_encoded_arguments(self):
        arguments = {
            "city": "Chicago",
            "guests": {"adults": 2, "children": 1},
        }
        payload = json.dumps(
            {
                "name": "book_hotel",
                "arguments": json.dumps(arguments, separators=(",", ":")),
            },
            separators=(",", ":"),
        )

        final = self._parse_tool_call_payload(payload)

        assert final.tool_calls[0]["name"] == "book_hotel"
        assert json.loads(final.tool_calls[0]["arguments"]) == arguments
        assert final.finish_reason == "tool_calls"

    def test_tool_call_repairs_missing_outer_brace(self):
        arguments = {
            "city": "Chicago",
            "guests": {"adults": 2, "children": 1},
        }
        payload = json.dumps(
            {"name": "book_hotel", "args": arguments},
            separators=(",", ":"),
        )[:-1]

        final = self._parse_tool_call_payload(payload)

        assert final.tool_calls[0]["name"] == "book_hotel"
        assert json.loads(final.tool_calls[0]["arguments"]) == arguments
        assert final.finish_reason == "tool_calls"

    def test_truncated_tool_call_ignores_braces_inside_strings(self):
        for text in ("open { brace", "close } brace", 'quoted "} brace'):
            payload = json.dumps(
                {"name": "write", "args": {"text": text}},
                separators=(",", ":"),
            )[:-1]

            final = self._parse_tool_call_payload(payload)

            assert len(final.tool_calls) == 1, text
            assert json.loads(final.tool_calls[0]["arguments"]) == {"text": text}
            assert final.finish_reason == "tool_calls"

    def test_partial_marker_across_tokens(self):
        token_map = {
            1: "<|content_",
            2: "text|>an",
            3: "swer<|end_",
            4: "message|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2, 3, 4])

        assert visible == "answer"
        assert "<|content_text|>" not in stream
        assert not stopped

    def test_unterminated_thinking_closed_at_finalize(self):
        token_map = {
            1: "<|content_thinking|>",
            2: "half a thought",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2])

        assert stream == "<think>half a thought</think>"
        assert visible == stream

    def test_non_inkling_models_unaffected(self):
        tokenizer = InklingTokenizer({0: "a", 1: "b"})
        factory = detect_output_parser(
            "llama-3-8b",
            tokenizer,
            {"model_type": "llama"},
        )
        assert factory is None


class TestMuseGlimmerOutputParserSession:
    """Muse Glimmer channel protocol: <|start|>role to=X<|message|>body."""

    def _factory(self, token_map, model_config=None):
        tokenizer = InklingTokenizer(token_map)
        factory = detect_output_parser(
            "Muse-Glimmer-30B",
            tokenizer,
            model_config or {"model_type": "muse_glimmer"},
        )
        assert factory is not None
        assert factory.kind == "muse_glimmer"
        return tokenizer, factory

    def _run(self, session, token_ids):
        stream, visible, stopped = [], [], False
        for token_id in token_ids:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            if result.is_stop:
                stopped = True
                break
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)
        return "".join(stream), "".join(visible), stopped, final

    def test_reasoning_then_answer(self):
        token_map = {
            1: " to=self",
            2: "<|message|>",
            3: "let me ",
            4: "reason",
            5: "<|eom|>",
            6: "<|start|>",
            7: "assistant to=user",
            8: "<|message|>",
            9: "Paris.",
            10: "<|eot|>",
            11: "<|end_of_text|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(
            session, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

        assert stream == "<think>let me reason</think>Paris."
        assert visible == stream
        assert stopped
        assert final.tool_calls == []
        assert 10 in factory.stop_token_ids
        assert 11 in factory.stop_token_ids

    def test_bare_answer_without_recipient(self):
        token_map = {
            1: "<|message|>",
            2: "Hello",
            3: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2, 3])

        assert stream == "Hello"
        assert visible == "Hello"
        assert stopped
        assert final.tool_calls == []

    def test_tool_call_suppressed_and_parsed(self):
        token_map = {
            1: " to=self",
            2: "<|message|>",
            3: "need weather",
            4: "<|eom|>",
            5: "<|start|>",
            6: "assistant to=get_weather",
            7: "<|message|>",
            8: (
                '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
                '<atem:parameter name="city">Seoul</atem:parameter>\n'
                '<atem:parameter name="days">3</atem:parameter>\n'
                "</atem:invoke>\n</atem:function_calls>"
            ),
            9: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(
            session, [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

        assert stream == "<think>need weather</think>"
        assert visible == stream
        assert stopped
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["name"] == "get_weather"
        assert json.loads(final.tool_calls[0]["arguments"]) == {
            "city": "Seoul",
            "days": 3,
        }
        assert final.finish_reason == "tool_calls"

    def test_tool_call_first_turn_without_leading_space(self):
        # The streaming detokenizer strips the leading space off the first
        # segment of a generation, so a turn that opens directly with a tool
        # call (no to=self message first — typical right after a tool-error
        # result) reaches the parser as "to=bash..." with nothing between it
        # and the synthetic "<|start|>assistant" prepend at finalize.
        token_map = {
            1: "to=bash",
            2: "<|message|>",
            3: (
                '<atem:function_calls>\n<atem:invoke name="bash">\n'
                "<atem:parameter name=\"command\">ls -la /tmp/reports"
                "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
            ),
            4: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2, 3, 4])

        assert visible == ""
        assert stopped
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0]["name"] == "bash"
        assert json.loads(final.tool_calls[0]["arguments"]) == {
            "command": "ls -la /tmp/reports"
        }
        assert final.finish_reason == "tool_calls"

    def test_multiple_tool_calls_across_messages(self):
        token_map = {
            1: " to=alpha",
            2: "<|message|>",
            3: (
                '<atem:function_calls>\n<atem:invoke name="alpha">\n'
                '<atem:parameter name="x">1</atem:parameter>\n'
                "</atem:invoke>\n</atem:function_calls>"
            ),
            4: "<|eom|>",
            5: "<|start|>",
            6: "assistant to=beta",
            7: "<|message|>",
            8: (
                '<atem:function_calls>\n<atem:invoke name="beta">\n'
                '<atem:parameter name="y">2</atem:parameter>\n'
                "</atem:invoke>\n</atem:function_calls>"
            ),
            9: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(
            session, [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

        assert stream == ""
        assert visible == ""
        assert stopped
        assert [c["name"] for c in final.tool_calls] == ["alpha", "beta"]

    def test_atem_value_typing_without_schema(self):
        token_map = {
            1: " to=fn",
            2: "<|message|>",
            3: (
                '<atem:function_calls>\n<atem:invoke name="fn">\n'
                '<atem:parameter name="count">42</atem:parameter>\n'
                '<atem:parameter name="flag">true</atem:parameter>\n'
                '<atem:parameter name="nothing">null</atem:parameter>\n'
                '<atem:parameter name="obj">{"a": 1}</atem:parameter>\n'
                '<atem:parameter name="text">multi\nline "quoted"\n</atem:parameter>\n'
                "</atem:invoke>\n</atem:function_calls>"
            ),
            4: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        _, _, stopped, final = self._run(session, [1, 2, 3, 4])

        assert stopped
        args = json.loads(final.tool_calls[0]["arguments"])
        assert args["count"] == 42
        assert args["flag"] is True
        assert args["nothing"] is None
        assert args["obj"] == {"a": 1}
        assert args["text"] == 'multi\nline "quoted"\n'

    def test_atem_schema_keeps_numeric_strings(self):
        token_map = {
            1: " to=get_weather",
            2: "<|message|>",
            3: (
                '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
                '<atem:parameter name="zip">04524</atem:parameter>\n'
                '<atem:parameter name="days">3</atem:parameter>\n'
                "</atem:invoke>\n</atem:function_calls>"
            ),
            4: "<|eot|>",
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zip": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        tokenizer, factory = self._factory(token_map)
        assert factory.create_session_with_tools is not None
        session = factory.create_session_with_tools(tokenizer, tools)
        _, _, stopped, final = self._run(session, [1, 2, 3, 4])

        assert stopped
        args = json.loads(final.tool_calls[0]["arguments"])
        assert args["zip"] == "04524"
        assert args["days"] == 3

    def test_marker_split_across_tokens(self):
        token_map = {
            1: " to=self",
            2: "<|message|>",
            3: "thinking",
            4: "<|eo",
            5: "m|>",
            6: "<|start|>",
            7: "assistant to=user",
            8: "<|message|>",
            9: "done",
            10: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(
            session, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

        assert stream == "<think>thinking</think>done"
        assert visible == stream
        assert stopped

    def test_unterminated_thinking_closed_at_finalize(self):
        token_map = {
            1: " to=self",
            2: "<|message|>",
            3: "still going",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2, 3])

        assert stream == "<think>still going</think>"
        assert visible == stream
        assert not stopped

    def test_namespaced_recipient_is_tool(self):
        token_map = {
            1: " to=browser.search",
            2: "<|message|>",
            3: (
                '<atem:function_calls>\n<atem:invoke name="browser.search">\n'
                '<atem:parameter name="query">weather</atem:parameter>\n'
                "</atem:invoke>\n</atem:function_calls>"
            ),
            4: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2, 3, 4])

        assert stream == ""
        assert visible == ""
        assert final.tool_calls[0]["name"] == "browser.search"

    def test_atem_example_in_reasoning_not_parsed(self):
        token_map = {
            1: " to=self",
            2: "<|message|>",
            3: (
                "I could call it like "
                '<atem:function_calls><atem:invoke name="fake">'
                '<atem:parameter name="a">1</atem:parameter>'
                "</atem:invoke></atem:function_calls> but I will answer."
            ),
            4: "<|eom|>",
            5: "<|start|>",
            6: "assistant to=user",
            7: "<|message|>",
            8: "No tool needed.",
            9: "<|eot|>",
        }
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        _, _, stopped, final = self._run(session, [1, 2, 3, 4, 5, 6, 7, 8, 9])

        assert stopped
        assert final.tool_calls == []

    def test_off_protocol_text_flushes_after_head_limit(self):
        long_text = "word " * 30  # 150 chars, no markers at all
        token_map = {1: long_text, 2: "more text"}
        tokenizer, factory = self._factory(token_map)
        session = factory.create_session(tokenizer)
        stream, visible, stopped, final = self._run(session, [1, 2])

        combined = stream
        assert long_text.strip()[:20] in combined
        assert "more text" in combined
        assert not stopped

    def test_non_muse_model_not_claimed(self):
        tokenizer = InklingTokenizer({})
        factory = detect_output_parser(
            "llama-3-8b",
            tokenizer,
            {"model_type": "llama"},
        )
        assert factory is None
