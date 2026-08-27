import json
from collections.abc import Generator
from types import SimpleNamespace
from typing import cast

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tool_parsers import gemma4, glm47, mistral, qwen3_coder

from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx import builder as mlx_builder
from exo.worker.runner.llm_inference.model_output_parsers import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _responses(parts: list[str]) -> Generator[GenerationResponse]:
    for token, text in enumerate(parts):
        yield GenerationResponse(
            text=text,
            token=token,
            finish_reason="stop" if token == len(parts) - 1 else None,
            usage=None,
        )


@pytest.mark.parametrize("model_family", ["qwen3-coder", "step-3.5-flash"])
def test_qwen_style_tool_calls_survive_split_stream_markers(
    model_family: str,
) -> None:
    body = (
        "\n<function=write_file>\n"
        "<parameter=path>\nmain.py\n</parameter>\n"
        "<parameter=content>\nprint(1)\n</parameter>\n"
        "</function>\n"
    )
    parts = ["<tool", "_call>", body, "</tool", "_call>"]

    results = list(
        parse_tool_calls(
            _responses(parts),
            make_mlx_parser(
                qwen3_coder.tool_call_start,
                qwen3_coder.tool_call_end,
                qwen3_coder.parse_tool_call,
            ),
            tools=None,
        )
    )

    assert model_family
    assert len(results) == 1
    assert isinstance(results[0], ToolCallResponse)
    assert results[0].tool_calls[0].name == "write_file"
    assert json.loads(results[0].tool_calls[0].arguments) == {
        "path": "main.py",
        "content": "print(1)",
    }


def test_qwen_style_tool_call_is_invariant_at_every_character_split() -> None:
    full = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\nmain.py\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    parser = make_mlx_parser(
        qwen3_coder.tool_call_start,
        qwen3_coder.tool_call_end,
        qwen3_coder.parse_tool_call,
    )

    for split_at in range(1, len(full)):
        results = list(
            parse_tool_calls(
                _responses([full[:split_at], full[split_at:]]), parser, tools=None
            )
        )
        assert len(results) == 1, split_at
        assert isinstance(results[0], ToolCallResponse), split_at
        assert results[0].tool_calls[0].name == "write_file", split_at


def test_gemma4_tool_call_survives_split_stream_markers() -> None:
    body = 'call:write_file{path:<|"|>main.py<|"|>,content:<|"|>print(1)<|"|>}'
    parts = ["<|tool", "_call>", body, "<tool_", "call|>"]

    results = list(
        parse_tool_calls(
            _responses(parts),
            make_mlx_parser(
                gemma4.tool_call_start,
                gemma4.tool_call_end,
                gemma4.parse_tool_call,
            ),
            tools=None,
        )
    )

    assert len(results) == 1
    assert isinstance(results[0], ToolCallResponse)
    assert results[0].tool_calls[0].name == "write_file"
    assert json.loads(results[0].tool_calls[0].arguments) == {
        "path": "main.py",
        "content": "print(1)",
    }


def test_glm52_tool_call_survives_split_stream_markers() -> None:
    body = (
        "write_file"
        "<arg_key>path</arg_key><arg_value>main.py</arg_value>"
        "<arg_key>content</arg_key><arg_value>print(1)</arg_value>"
    )
    parts = ["<tool", "_call>", body, "</tool_", "call>"]

    results = list(
        parse_tool_calls(
            _responses(parts),
            make_mlx_parser(
                glm47.tool_call_start,
                glm47.tool_call_end,
                glm47.parse_tool_call,
            ),
            tools=None,
        )
    )

    assert len(results) == 1
    assert isinstance(results[0], ToolCallResponse)
    assert results[0].tool_calls[0].name == "write_file"
    assert json.loads(results[0].tool_calls[0].arguments) == {
        "path": "main.py",
        "content": "print(1)",
    }


def test_mistral_tool_call_without_end_marker_waits_for_finish() -> None:
    parts = ["[TOOL", "_CALLS]write", "_file[ARGS]", '{"path":"main.py"}', ""]

    results = list(
        parse_tool_calls(
            _responses(parts),
            make_mlx_parser(
                mistral.tool_call_start,
                mistral.tool_call_end,
                mistral.parse_tool_call,
            ),
            tools=None,
        )
    )

    assert len(results) == 1
    assert isinstance(results[0], ToolCallResponse)
    assert results[0].tool_calls[0].name == "write_file"
    assert json.loads(results[0].tool_calls[0].arguments) == {"path": "main.py"}


def test_mistral_tool_call_parses_when_generator_ends_without_finish() -> None:
    def unfinished() -> Generator[GenerationResponse]:
        yield GenerationResponse(
            text='[TOOL_CALLS]write_file[ARGS]{"path":"main.py"}',
            token=0,
            finish_reason=None,
            usage=None,
        )

    parser = make_mlx_parser(
        mistral.tool_call_start,
        mistral.tool_call_end,
        mistral.parse_tool_call,
    )
    results = list(parse_tool_calls(unfinished(), parser, tools=None))

    assert len(results) == 1
    assert isinstance(results[0], ToolCallResponse)
    assert results[0].tool_calls[0].name == "write_file"


def test_mistral_empty_end_marker_is_wired_by_mlx_builder() -> None:
    tokenizer = cast(
        TokenizerWrapper,
        cast(
            object,
            SimpleNamespace(
                tool_call_start=mistral.tool_call_start,
                tool_call_end=mistral.tool_call_end,
                tool_parser=mistral.parse_tool_call,
            ),
        ),
    )

    parser = mlx_builder._tokenizer_tool_parser(tokenizer)  # pyright: ignore[reportPrivateUsage]

    assert parser is not None
    assert parser.start_parsing == "[TOOL_CALLS]"
    assert parser.end_parsing == ""
