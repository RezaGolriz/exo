"""Regression tests for exo-explore/exo#2115."""

# pyright: reportAny=false, reportUnknownMemberType=false

from typing import Any
from unittest.mock import MagicMock

import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.utils_mlx import (
    _coerce_chat_template_text,  # pyright: ignore[reportPrivateUsage]
    apply_chat_template,
    render_chat_template,
)


def _make_task_params(**overrides: Any) -> TextGenerationTaskParams:
    defaults: dict[str, Any] = {
        "model": ModelId("mlx-community/test-vlm-4bit"),
        "input": [InputMessage(role="user", content=InputMessageContent("hi"))],
    }
    defaults.update(overrides)
    return TextGenerationTaskParams(**defaults)


def test_string_partial_content_is_spliced_onto_rendered_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered>"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Draw a cat"},
        {"role": "assistant", "content": "Sure, here is "},
    ]

    prompt = render_chat_template(tokenizer, messages, _make_task_params())

    assert prompt == "<rendered>Sure, here is "
    rendered_messages = tokenizer.apply_chat_template.call_args.args[0]
    assert all(message.get("role") != "assistant" for message in rendered_messages)
    assert tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"]
    assert (
        "continue_final_message" not in tokenizer.apply_chat_template.call_args.kwargs
    )


@pytest.mark.parametrize(
    "structured_content",
    [
        [{"type": "text", "text": "Here: "}, {"type": "image"}],
        {"type": "image"},
    ],
)
def test_structured_partial_content_preserves_media(
    structured_content: object,
) -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered>"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Draw a cat"},
        {"role": "assistant", "content": structured_content},
    ]

    prompt = render_chat_template(tokenizer, messages, _make_task_params())

    assert prompt == "<rendered>"
    rendered_messages = tokenizer.apply_chat_template.call_args.args[0]
    assert rendered_messages[-1]["role"] == "assistant"
    assert rendered_messages[-1]["content"] == structured_content
    assert tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"]
    assert (
        "continue_final_message" not in tokenizer.apply_chat_template.call_args.kwargs
    )


def test_public_apply_chat_template_handles_multimodal_assistant_prefill() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered>"
    chat_template_messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "Show a cat"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here: "}, {"type": "image"}],
        },
    ]
    task_params = _make_task_params(chat_template_messages=chat_template_messages)

    assert apply_chat_template(tokenizer, task_params) == "<rendered>"
    rendered_messages = tokenizer.apply_chat_template.call_args.args[0]
    assert rendered_messages[-1]["content"] == chat_template_messages[-1]["content"]


def test_text_only_structured_assistant_prefill_keeps_continuation_semantics() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered>"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Complete this"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first "},
                {"type": "text", "text": "second"},
            ],
        },
    ]

    prompt = render_chat_template(tokenizer, messages, _make_task_params())

    assert prompt == "<rendered>first second"
    rendered_messages = tokenizer.apply_chat_template.call_args.args[0]
    assert all(message.get("role") != "assistant" for message in rendered_messages)
    assert tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"]
    assert (
        "continue_final_message" not in tokenizer.apply_chat_template.call_args.kwargs
    )


def test_empty_structured_assistant_prefill_is_removed() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered>"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Complete this"},
        {"role": "assistant", "content": []},
    ]

    assert (
        render_chat_template(tokenizer, messages, _make_task_params()) == "<rendered>"
    )
    rendered_messages = tokenizer.apply_chat_template.call_args.args[0]
    assert all(message.get("role") != "assistant" for message in rendered_messages)


def test_chat_template_text_str_passthrough() -> None:
    assert _coerce_chat_template_text("hello") == "hello"


def test_chat_template_text_unwraps_single_rendered_prompt() -> None:
    assert _coerce_chat_template_text(["hello world"]) == "hello world"


@pytest.mark.parametrize(
    "value",
    [[], ["first conversation", "second conversation"], [{"type": "image"}], 42, None],
)
def test_chat_template_text_rejects_unexpected_type(value: object) -> None:
    with pytest.raises(TypeError, match="expected"):
        _coerce_chat_template_text(value)


def test_render_chat_template_unwraps_single_tokenizer_result() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = ["<a><b>"]
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

    assert render_chat_template(tokenizer, messages, _make_task_params()) == "<a><b>"


def test_assistant_tool_call_is_not_dropped() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<tool-call-history><assistant>"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Check the weather"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        },
    ]

    prompt = render_chat_template(tokenizer, messages, _make_task_params())

    assert prompt == "<tool-call-history><assistant>"
    rendered_messages = tokenizer.apply_chat_template.call_args.args[0]
    assert rendered_messages[-1]["tool_calls"][0]["id"] == "call-1"
    assert tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"]
    assert (
        "continue_final_message" not in tokenizer.apply_chat_template.call_args.kwargs
    )


def test_reasoning_metadata_does_not_disable_text_prefill() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<rendered>"
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Complete this"},
        {
            "role": "assistant",
            "content": "partial",
            "reasoning_content": "prior reasoning",
        },
    ]

    assert render_chat_template(tokenizer, messages, _make_task_params()) == (
        "<rendered>partial"
    )
