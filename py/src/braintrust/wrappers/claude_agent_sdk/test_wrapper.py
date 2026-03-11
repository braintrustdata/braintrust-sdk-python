"""Tests for the Claude Agent SDK wrapper."""

import asyncio
import dataclasses
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

# Try to import the Claude Agent SDK - skip tests if not available
try:
    import claude_agent_sdk as _claude_agent_sdk

    claude_agent_sdk = cast(Any, _claude_agent_sdk)
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    claude_agent_sdk = cast(Any, None)
    CLAUDE_SDK_AVAILABLE = False
    print("Claude Agent SDK not installed, skipping integration tests")

from braintrust import logger
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk
from braintrust.wrappers.claude_agent_sdk._test_transport import make_cassette_transport
from braintrust.wrappers.claude_agent_sdk._wrapper import (
    ToolSpanTracker,
    _build_llm_input,
    _create_client_wrapper_class,
    _create_tool_wrapper_class,
    _extract_usage_from_result_message,
    _parse_tool_name,
    _serialize_content_blocks,
    _serialize_system_message,
    _serialize_tool_result_output,
    _thread_local,
)
from braintrust.wrappers.test_utils import verify_autoinstrument_script

PROJECT_NAME = "test-claude-agent-sdk"
TEST_MODEL = "claude-haiku-4-5-20251001"
REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture
def memory_logger():
    """Memory-based logger for testing span creation."""
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.skipif(not CLAUDE_SDK_AVAILABLE, reason="Claude Agent SDK not installed")
@pytest.mark.asyncio
async def test_calculator_with_multiple_operations(memory_logger):
    """Test claude_agent.py example - calculator with multiple operations."""
    assert not memory_logger.pop()

    original_client = claude_agent_sdk.ClaudeSDKClient
    original_tool_class = claude_agent_sdk.SdkMcpTool
    claude_agent_sdk.ClaudeSDKClient = _create_client_wrapper_class(original_client)
    claude_agent_sdk.SdkMcpTool = _create_tool_wrapper_class(original_tool_class)

    try:
        # Create calculator tool
        async def calculator_handler(args):
            operation = args["operation"]
            a = args["a"]
            b = args["b"]

            if operation == "multiply":
                result = a * b
            elif operation == "subtract":
                result = a - b
            elif operation == "add":
                result = a + b
            elif operation == "divide":
                if b == 0:
                    return {
                        "content": [{"type": "text", "text": "Error: Division by zero"}],
                        "isError": True,
                    }
                result = a / b
            else:
                return {
                    "content": [{"type": "text", "text": f"Unknown operation: {operation}"}],
                    "isError": True,
                }

            return {
                "content": [{"type": "text", "text": f"The result of {operation}({a}, {b}) is {result}"}],
            }

        calculator_tool = claude_agent_sdk.SdkMcpTool(
            name="calculator",
            description="Performs basic arithmetic operations",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform",
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["operation", "a", "b"],
            },
            handler=calculator_handler,
        )

        options = claude_agent_sdk.ClaudeAgentOptions(
            model=TEST_MODEL,
            permission_mode="bypassPermissions",
            mcp_servers={
                "calculator": claude_agent_sdk.create_sdk_mcp_server(
                    name="calculator",
                    version="1.0.0",
                    tools=[calculator_tool],
                )
            },
        )
        transport = make_cassette_transport(
            cassette_name="test_calculator_with_multiple_operations",
            prompt="",
            options=options,
        )

        result_message = None
        async with claude_agent_sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query("What is 15 multiplied by 7? Then subtract 5 from the result.")
            async for message in client.receive_response():
                if type(message).__name__ == "ResultMessage":
                    result_message = message

    finally:
        claude_agent_sdk.ClaudeSDKClient = original_client
        claude_agent_sdk.SdkMcpTool = original_tool_class

    spans = memory_logger.pop()

    task_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TASK]
    assert len(task_spans) == 1, f"Should have exactly one task span, got {len(task_spans)}"

    task_span = task_spans[0]
    assert task_span["span_attributes"]["name"] == "Claude Agent"
    assert "15 multiplied by 7" in task_span["input"]
    assert task_span["output"] is not None

    assert result_message is not None, "Should have received result message"
    if hasattr(result_message, "num_turns"):
        assert task_span.get("metadata", {}).get("num_turns") is not None
    if hasattr(result_message, "session_id"):
        assert task_span.get("metadata", {}).get("session_id") is not None

    llm_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.LLM]
    assert len(llm_spans) >= 1, f"Should have at least one LLM span, got {len(llm_spans)}"
    llm_span_ids = {span["span_id"] for span in llm_spans}
    _assert_llm_spans_have_time_to_first_token(llm_spans)

    llm_spans_with_metrics = [s for s in llm_spans if "prompt_tokens" in s.get("metrics", {})]
    assert len(llm_spans_with_metrics) >= 1, "At least one LLM span should have token metrics"

    for llm_span in llm_spans:
        assert llm_span["span_attributes"]["name"] == "anthropic.messages.create"
        assert isinstance(llm_span["output"], list)
        assert len(llm_span["output"]) > 0
        for metric_name in ("prompt_tokens", "completion_tokens", "tokens"):
            if metric_name in llm_span.get("metrics", {}):
                assert llm_span["metrics"][metric_name] > 0
    tool_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TOOL]
    for tool_span in tool_spans:
        assert tool_span["span_attributes"]["name"] == "calculator"
        assert tool_span["input"] is not None
        assert tool_span["output"] is not None
        assert any(parent_id in llm_span_ids for parent_id in tool_span["span_parents"])

    root_span_id = task_span["span_id"]
    for llm_span in llm_spans:
        assert llm_span["root_span_id"] == root_span_id
        assert root_span_id in llm_span["span_parents"]

    for tool_span in tool_spans:
        assert tool_span["root_span_id"] == root_span_id
        assert any(parent_id in llm_span_ids for parent_id in tool_span["span_parents"])


def _make_message(content: str) -> dict:
    """Create a streaming format message dict."""
    return {"type": "user", "message": {"role": "user", "content": content}}


def _assert_structured_input(task_span: dict, expected_contents: list[str]) -> None:
    """Assert that task span input is a structured list with expected content."""
    inp = task_span.get("input")
    assert isinstance(inp, list), f"Expected list input, got {type(inp).__name__}: {inp}"
    assert [x["message"]["content"] for x in inp] == expected_contents


def _assert_llm_spans_have_time_to_first_token(llm_spans: list[dict[str, Any]]) -> None:
    assert llm_spans, "Expected at least one LLM span"
    for llm_span in llm_spans:
        assert "time_to_first_token" in llm_span.get("metrics", {})
        assert llm_span["metrics"]["time_to_first_token"] >= 0


def _sdk_version_at_least(version: str) -> bool:
    if not CLAUDE_SDK_AVAILABLE:
        return False

    def parse(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdigit())

    return parse(getattr(claude_agent_sdk, "__version__", "0")) >= parse(version)


class CustomAsyncIterable:
    """Custom AsyncIterable class (not a generator) for testing."""

    def __init__(self, messages: list[dict]):
        self._messages = messages

    def __aiter__(self):
        return CustomAsyncIterator(self._messages)


class CustomAsyncIterator:
    """Iterator for CustomAsyncIterable."""

    def __init__(self, messages: list[dict]):
        self._messages = messages
        self._index = 0

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._index]
        self._index += 1
        return msg


@pytest.mark.skipif(not CLAUDE_SDK_AVAILABLE, reason="Claude Agent SDK not installed")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cassette_name,input_factory,expected_contents",
    [
        pytest.param(
            "test_query_async_iterable_asyncgen_single",
            lambda: (msg async for msg in _single_message_generator()),
            ["What is 2 + 2?"],
            id="asyncgen_single",
        ),
        pytest.param(
            "test_query_async_iterable_asyncgen_multi",
            lambda: (msg async for msg in _multi_message_generator()),
            ["Part 1", "Part 2"],
            id="asyncgen_multi",
        ),
        pytest.param(
            "test_query_async_iterable_custom_async_iterable",
            lambda: CustomAsyncIterable([_make_message("Custom 1"), _make_message("Custom 2")]),
            ["Custom 1", "Custom 2"],
            id="custom_async_iterable",
        ),
    ],
)
async def test_query_async_iterable(memory_logger, cassette_name, input_factory, expected_contents):
    """Test that async iterable inputs are captured as structured lists."""
    assert not memory_logger.pop()

    original_client = claude_agent_sdk.ClaudeSDKClient
    claude_agent_sdk.ClaudeSDKClient = _create_client_wrapper_class(original_client)

    try:
        options = claude_agent_sdk.ClaudeAgentOptions(
            model=TEST_MODEL,
            permission_mode="bypassPermissions",
        )
        transport = make_cassette_transport(
            cassette_name=cassette_name,
            prompt="",
            options=options,
        )

        async with claude_agent_sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(input_factory())
            async for message in client.receive_response():
                if type(message).__name__ == "ResultMessage":
                    break
    finally:
        claude_agent_sdk.ClaudeSDKClient = original_client

    spans = memory_logger.pop()

    task_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TASK]
    assert len(task_spans) >= 1, f"Should have at least one task span, got {len(task_spans)}"

    task_span = next(
        (s for s in task_spans if s["span_attributes"]["name"] == "Claude Agent"),
        task_spans[0],
    )
    _assert_structured_input(task_span, expected_contents)

    llm_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.LLM]
    _assert_llm_spans_have_time_to_first_token(llm_spans)


@pytest.mark.skipif(not CLAUDE_SDK_AVAILABLE, reason="Claude Agent SDK not installed")
@pytest.mark.asyncio
async def test_bundled_subagent_creates_task_span(memory_logger):
    assert not memory_logger.pop()
    if not _sdk_version_at_least("0.1.48"):
        pytest.skip("Bundled subagent task events were not observed on older Claude Agent SDK versions")

    original_client = claude_agent_sdk.ClaudeSDKClient
    claude_agent_sdk.ClaudeSDKClient = _create_client_wrapper_class(original_client)

    try:
        options = claude_agent_sdk.ClaudeAgentOptions(
            model=TEST_MODEL,
            cwd=REPO_ROOT,
            permission_mode="bypassPermissions",
            max_turns=8,
        )
        transport = make_cassette_transport(
            cassette_name="test_bundled_subagent_creates_task_span",
            prompt="",
            options=options,
        )

        async with claude_agent_sdk.ClaudeSDKClient(options=options, transport=transport) as client:
            await client.query(
                "You must delegate this task to the bundled general-purpose agent. "
                "Have that agent inspect the current repository and reply with only the repository name. "
                "Do not answer directly without using the subagent."
            )
            async for message in client.receive_response():
                if type(message).__name__ == "ResultMessage":
                    break
    finally:
        claude_agent_sdk.ClaudeSDKClient = original_client

    spans = memory_logger.pop()

    task_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TASK]
    assert len(task_spans) >= 2, f"Expected root task span and subagent span, got {len(task_spans)}"

    root_task_span = _find_span_by_name(task_spans, "Claude Agent")
    subagent_spans = [s for s in task_spans if s["span_attributes"]["name"] != "Claude Agent"]
    tool_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TOOL]
    assert subagent_spans, "Expected at least one subagent task span"
    assert any(s.get("metadata", {}).get("task_id") for s in subagent_spans)
    for subagent_span in subagent_spans:
        assert subagent_span["root_span_id"] == root_task_span["span_id"]
        parents = set(subagent_span["span_parents"])
        tool_use_id = subagent_span.get("metadata", {}).get("tool_use_id")
        matching_tool_span = next(
            (s for s in tool_spans if s.get("metadata", {}).get("gen_ai.tool.call.id") == tool_use_id),
            None,
        )
        if matching_tool_span is not None:
            assert matching_tool_span["span_id"] in parents
        else:
            assert root_task_span["span_id"] in parents

    assert root_task_span.get("metadata", {}).get("task_events"), "Expected task events on root task span"

    llm_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.LLM]
    _assert_llm_spans_have_time_to_first_token(llm_spans)

async def _single_message_generator():
    """Generator yielding a single message."""
    yield _make_message("What is 2 + 2?")


async def _multi_message_generator():
    """Generator yielding multiple messages."""
    yield _make_message("Part 1")
    yield _make_message("Part 2")


@dataclasses.dataclass
class TextBlock:
    text: str


@dataclasses.dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclasses.dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any
    is_error: bool | None = None


@dataclasses.dataclass
class AssistantMessage:
    content: list[Any]
    model: str = TEST_MODEL


@dataclasses.dataclass
class UserMessage:
    content: list[Any]


@dataclasses.dataclass
class TaskStartedMessage:
    subtype: str
    data: dict[str, Any]
    task_id: str
    description: str
    uuid: str
    session_id: str
    tool_use_id: str | None = None
    task_type: str | None = None


@dataclasses.dataclass
class TaskProgressMessage:
    subtype: str
    data: dict[str, Any]
    task_id: str
    description: str
    usage: dict[str, Any]
    uuid: str
    session_id: str
    tool_use_id: str | None = None
    last_tool_name: str | None = None


@dataclasses.dataclass
class TaskNotificationMessage:
    subtype: str
    data: dict[str, Any]
    task_id: str
    status: str
    output_file: str
    summary: str
    uuid: str
    session_id: str
    tool_use_id: str | None = None
    usage: dict[str, Any] | None = None


class ResultMessage:
    def __init__(
        self,
        *,
        input_tokens: int = 1,
        output_tokens: int = 1,
        cache_creation_input_tokens: int = 0,
        num_turns: int = 1,
        session_id: str = "session-123",
    ):
        self.usage = types.SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )
        self.num_turns = num_turns
        self.session_id = session_id


def _make_fake_sdk_mcp_tool_class():
    class FakeSdkMcpTool:
        def __init__(self, name, description, input_schema, handler, **kwargs):
            del kwargs
            self.name = name
            self.description = description
            self.input_schema = input_schema
            self.handler = handler

    return FakeSdkMcpTool


def _find_spans_by_type(spans: list[dict[str, Any]], span_type: str) -> list[dict[str, Any]]:
    return [span for span in spans if span.get("span_attributes", {}).get("type") == span_type]


def _find_span_by_name(spans: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for span in spans:
        if span["span_attributes"]["name"] == name:
            return span

    available_names = [span["span_attributes"]["name"] for span in spans]
    raise AssertionError(f"Expected span named {name!r}. Available spans: {available_names}")


def _clear_tool_span_tracker() -> None:
    if hasattr(_thread_local, "tool_span_tracker"):
        delattr(_thread_local, "tool_span_tracker")


@pytest.mark.parametrize(
    "tool_name,expected",
    [
        pytest.param(
            "calculator",
            {
                "raw_name": "calculator",
                "display_name": "calculator",
                "is_mcp": False,
                "mcp_server": None,
            },
            id="plain",
        ),
        pytest.param(
            "mcp__filesystem__team__read_file",
            {
                "raw_name": "mcp__filesystem__team__read_file",
                "display_name": "read_file",
                "is_mcp": True,
                "mcp_server": "filesystem__team",
            },
            id="mcp_with_embedded_delimiters",
        ),
    ],
)
def test_parse_tool_name(tool_name, expected):
    parsed = _parse_tool_name(tool_name)

    assert parsed.raw_name == expected["raw_name"]
    assert parsed.display_name == expected["display_name"]
    assert parsed.is_mcp == expected["is_mcp"]
    assert parsed.mcp_server == expected["mcp_server"]


def test_tool_span_tracker_lifecycle(memory_logger):
    assert not memory_logger.pop()

    tracker = ToolSpanTracker()

    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK) as task_span:
        llm_span = start_span(
            name="anthropic.messages.create",
            type=SpanTypeAttribute.LLM,
            parent=task_span.export(),
        )
        tracker.start_tool_spans(
            AssistantMessage(
                content=[
                    TextBlock("Let me calculate that."),
                    ToolUseBlock(id="call-4", name="calculator", input={"operation": "multiply", "a": 6, "b": 7}),
                ]
            ),
            llm_span.export(),
        )
        tracker.finish_tool_spans(
            UserMessage(content=[ToolResultBlock(tool_use_id="call-4", content=[TextBlock("42")])])
        )
        llm_span.end()

    spans = memory_logger.pop()
    llm_span_log = _find_span_by_name(spans, "anthropic.messages.create")
    tool_span = _find_span_by_name(spans, "calculator")

    assert tool_span["input"] == {"operation": "multiply", "a": 6, "b": 7}
    assert tool_span["output"] == {"content": "42"}
    assert tool_span["metadata"]["gen_ai.tool.name"] == "calculator"
    assert tool_span["metadata"]["gen_ai.tool.call.id"] == "call-4"
    assert llm_span_log["span_id"] in tool_span["span_parents"]


def test_tool_span_tracker_logs_errors(memory_logger):
    assert not memory_logger.pop()

    tracker = ToolSpanTracker()

    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK) as task_span:
        llm_span = start_span(
            name="anthropic.messages.create",
            type=SpanTypeAttribute.LLM,
            parent=task_span.export(),
        )
        tracker.start_tool_spans(
            AssistantMessage(content=[ToolUseBlock(id="call-err", name="calculator", input={"a": 1, "b": 0})]),
            llm_span.export(),
        )
        tracker.finish_tool_spans(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="call-err", content=[TextBlock("Division by zero")], is_error=True)]
            )
        )
        llm_span.end()

    spans = memory_logger.pop()
    tool_span = _find_span_by_name(spans, "calculator")

    assert tool_span["output"] == {"content": "Division by zero", "is_error": True}
    assert tool_span["error"] == "Division by zero"


def test_tool_span_tracker_cleanup_closes_unmatched_spans(memory_logger):
    assert not memory_logger.pop()

    tracker = ToolSpanTracker()

    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK) as task_span:
        llm_span = start_span(
            name="anthropic.messages.create",
            type=SpanTypeAttribute.LLM,
            parent=task_span.export(),
        )
        tracker.start_tool_spans(
            AssistantMessage(content=[ToolUseBlock(id="call-dangling", name="weather", input={"city": "Toronto"})]),
            llm_span.export(),
        )
        tracker.cleanup()
        llm_span.end()

    spans = memory_logger.pop()
    tool_span = _find_span_by_name(spans, "weather")

    assert tool_span["input"] == {"city": "Toronto"}
    assert tool_span.get("output") is None


def test_serialize_content_blocks_keeps_malformed_text_block_payload():
    malformed_tool_result = ToolResultBlock(
        tool_use_id="call-malformed",
        content=[{"type": "text"}],
    )

    serialized = _serialize_content_blocks([malformed_tool_result])

    assert serialized == [
        {
            "tool_use_id": "call-malformed",
            "content": [{"type": "text"}],
            "type": "tool_result",
        }
    ]


@pytest.mark.asyncio
async def test_wrapped_tool_handler_creates_fallback_tool_span_without_active_stream(memory_logger):
    assert not memory_logger.pop()

    wrapped_tool_class = _create_tool_wrapper_class(_make_fake_sdk_mcp_tool_class())

    async def calculator_handler(args):
        return {"content": [{"type": "text", "text": f"{args['a'] * args['b']}"}]}

    calculator_tool = wrapped_tool_class(
        name="calculator",
        description="Multiply two numbers",
        input_schema={"type": "object"},
        handler=calculator_handler,
    )

    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK):
        result = await calculator_tool.handler({"operation": "multiply", "a": 6, "b": 7})

    assert result == {"content": [{"type": "text", "text": "42"}]}

    spans = memory_logger.pop()
    tool_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.TOOL), "calculator")

    assert tool_span["input"] == {"operation": "multiply", "a": 6, "b": 7}
    assert tool_span["output"] == {"content": [{"type": "text", "text": "42"}]}


def test_serialize_tool_result_output_flattens_text_blocks_and_errors():
    tool_result = ToolResultBlock(
        tool_use_id="call-err",
        content=[TextBlock("Division by zero")],
        is_error=True,
    )

    output = _serialize_tool_result_output(tool_result)

    assert output == {"content": "Division by zero", "is_error": True}


@pytest.mark.parametrize(
    "message,expected",
    [
        pytest.param(
            TaskStartedMessage(
                subtype="task_started",
                data={"subtype": "task_started", "task_id": "task-1"},
                task_id="task-1",
                description="Inspect the repository",
                uuid="msg-start",
                session_id="session-123",
                task_type="general-purpose",
            ),
            {
                "subtype": "task_started",
                "task_id": "task-1",
                "description": "Inspect the repository",
                "uuid": "msg-start",
                "session_id": "session-123",
                "task_type": "general-purpose",
            },
            id="task_started",
        ),
        pytest.param(
            TaskProgressMessage(
                subtype="task_progress",
                data={"subtype": "task_progress", "task_id": "task-1"},
                task_id="task-1",
                description="Running Bash",
                usage={"total_tokens": 11, "tool_uses": 1, "duration_ms": 250},
                uuid="msg-progress",
                session_id="session-123",
                tool_use_id="call-bash",
                last_tool_name="Bash",
            ),
            {
                "subtype": "task_progress",
                "task_id": "task-1",
                "description": "Running Bash",
                "uuid": "msg-progress",
                "session_id": "session-123",
                "tool_use_id": "call-bash",
                "last_tool_name": "Bash",
                "usage": {"total_tokens": 11, "tool_uses": 1, "duration_ms": 250},
            },
            id="task_progress",
        ),
        pytest.param(
            TaskNotificationMessage(
                subtype="task_notification",
                data={"subtype": "task_notification", "task_id": "task-1"},
                task_id="task-1",
                status="completed",
                output_file="/tmp/report.txt",
                summary="Repository inspection completed",
                uuid="msg-notify",
                session_id="session-123",
                tool_use_id="call-bash",
                usage={"total_tokens": 15, "tool_uses": 1, "duration_ms": 400},
            ),
            {
                "subtype": "task_notification",
                "task_id": "task-1",
                "uuid": "msg-notify",
                "session_id": "session-123",
                "tool_use_id": "call-bash",
                "status": "completed",
                "output_file": "/tmp/report.txt",
                "summary": "Repository inspection completed",
                "usage": {"total_tokens": 15, "tool_uses": 1, "duration_ms": 400},
            },
            id="task_notification",
        ),
    ],
)
def test_serialize_system_message_extracts_known_fields(message, expected):
    assert _serialize_system_message(message) == expected


def test_extract_usage_from_result_message_normalizes_anthropic_tokens():
    metrics = _extract_usage_from_result_message(ResultMessage(input_tokens=5, output_tokens=3, cache_creation_input_tokens=2))

    assert metrics == {
        "prompt_tokens": 7.0,
        "completion_tokens": 3.0,
        "prompt_cache_creation_tokens": 2.0,
        "tokens": 10.0,
    }


@pytest.mark.parametrize(
    "prompt,conversation_history,expected",
    [
        pytest.param(
            "What is 2 + 2?",
            [],
            [{"content": "What is 2 + 2?", "role": "user"}],
            id="prompt_only",
        ),
        pytest.param(
            "What is 2 + 2?",
            [
                {"role": "assistant", "content": "Let me calculate that."},
                {"role": "user", "content": "Please continue."},
            ],
            [
                {"content": "What is 2 + 2?", "role": "user"},
                {"role": "assistant", "content": "Let me calculate that."},
                {"role": "user", "content": "Please continue."},
            ],
            id="prompt_with_history",
        ),
        pytest.param(
            None,
            [
                {"role": "assistant", "content": "Let me calculate that."},
                {"role": "user", "content": "Please continue."},
            ],
            [
                {"role": "assistant", "content": "Let me calculate that."},
                {"role": "user", "content": "Please continue."},
            ],
            id="history_only",
        ),
    ],
)
def test_build_llm_input(prompt, conversation_history, expected):
    assert _build_llm_input(prompt, conversation_history) == expected


def test_tool_span_tracker_records_mcp_metadata(memory_logger):
    assert not memory_logger.pop()

    tracker = ToolSpanTracker()

    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK) as task_span:
        llm_span = start_span(
            name="anthropic.messages.create",
            type=SpanTypeAttribute.LLM,
            parent=task_span.export(),
        )
        tracker.start_tool_spans(
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="call-mcp",
                        name="mcp__filesystem__team__read_file",
                        input={"path": "/tmp/test.txt"},
                    )
                ]
            ),
            llm_span.export(),
        )
        tracker.finish_tool_spans(
            UserMessage(content=[ToolResultBlock(tool_use_id="call-mcp", content=[TextBlock("file contents")])])
        )
        llm_span.end()

    spans = memory_logger.pop()
    tool_span = _find_span_by_name(spans, "read_file")

    assert tool_span["input"] == {"path": "/tmp/test.txt"}
    assert tool_span["output"] == {"content": "file contents"}
    assert tool_span["metadata"]["gen_ai.tool.name"] == "read_file"
    assert tool_span["metadata"]["gen_ai.tool.call.id"] == "call-mcp"
    assert tool_span["metadata"]["gen_ai.operation.name"] == "execute_tool"
    assert tool_span["metadata"]["mcp.method.name"] == "tools/call"
    assert tool_span["metadata"]["mcp.server"] == "filesystem__team"
    assert tool_span["metadata"]["raw_tool_name"] == "mcp__filesystem__team__read_file"


@pytest.mark.asyncio
async def test_wrapped_tool_handler_keeps_nested_traces_under_stream_tool_span(memory_logger):
    assert not memory_logger.pop()

    wrapped_tool_class = _create_tool_wrapper_class(_make_fake_sdk_mcp_tool_class())

    async def calculator_handler(args):
        nested_span = start_span(name="nested_tool_work")
        nested_span.log(input=args)
        nested_span.end()
        return {"content": [{"type": "text", "text": "42"}]}

    calculator_tool = wrapped_tool_class(
        name="calculator",
        description="Multiply two numbers",
        input_schema={"type": "object"},
        handler=calculator_handler,
    )

    tracker = ToolSpanTracker()
    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK) as task_span:
        llm_span = start_span(
            name="anthropic.messages.create",
            type=SpanTypeAttribute.LLM,
            parent=task_span.export(),
        )
        tracker.start_tool_spans(
            AssistantMessage(
                content=[
                    ToolUseBlock(id="call-4", name="calculator", input={"operation": "multiply", "a": 6, "b": 7}),
                ]
            ),
            llm_span.export(),
        )
        _thread_local.tool_span_tracker = tracker
        try:
            result = await calculator_tool.handler({"operation": "multiply", "a": 6, "b": 7})
            tracker.finish_tool_spans(
                UserMessage(content=[ToolResultBlock(tool_use_id="call-4", content=[TextBlock("42")])])
            )
        finally:
            _clear_tool_span_tracker()
            tracker.cleanup()
            llm_span.end()

    assert result == {"content": [{"type": "text", "text": "42"}]}

    spans = memory_logger.pop()
    tool_span = _find_span_by_name(_find_spans_by_type(spans, SpanTypeAttribute.TOOL), "calculator")
    nested_span = _find_span_by_name(spans, "nested_tool_work")

    assert tool_span["span_id"] in nested_span["span_parents"]


@pytest.mark.asyncio
async def test_wrapped_tool_handler_matches_same_name_tool_spans_by_input(memory_logger):
    assert not memory_logger.pop()

    wrapped_tool_class = _create_tool_wrapper_class(_make_fake_sdk_mcp_tool_class())

    async def calculator_handler(args):
        nested_span = start_span(name=f"nested_tool_work_{args['a']}")
        nested_span.log(input=args)
        nested_span.end()
        return {"content": [{"type": "text", "text": str(args['a'] + args['b'])}]}

    calculator_tool = wrapped_tool_class(
        name="calculator",
        description="Add two numbers",
        input_schema={"type": "object"},
        handler=calculator_handler,
    )

    tracker = ToolSpanTracker()
    with start_span(name="Claude Agent", type=SpanTypeAttribute.TASK) as task_span:
        llm_span = start_span(
            name="anthropic.messages.create",
            type=SpanTypeAttribute.LLM,
            parent=task_span.export(),
        )
        tracker.start_tool_spans(
            AssistantMessage(
                content=[
                    ToolUseBlock(id="call-1", name="calculator", input={"operation": "add", "a": 2, "b": 3}),
                    ToolUseBlock(id="call-2", name="calculator", input={"operation": "add", "a": 10, "b": 5}),
                ]
            ),
            llm_span.export(),
        )
        _thread_local.tool_span_tracker = tracker
        try:
            await calculator_tool.handler({"operation": "add", "a": 10, "b": 5})
            await calculator_tool.handler({"operation": "add", "a": 2, "b": 3})
            tracker.finish_tool_spans(
                UserMessage(
                    content=[
                        ToolResultBlock(tool_use_id="call-1", content=[TextBlock("5")]),
                        ToolResultBlock(tool_use_id="call-2", content=[TextBlock("15")]),
                    ]
                )
            )
        finally:
            _clear_tool_span_tracker()
            tracker.cleanup()
            llm_span.end()

    spans = memory_logger.pop()
    calculator_spans = [
        span
        for span in _find_spans_by_type(spans, SpanTypeAttribute.TOOL)
        if span["span_attributes"]["name"] == "calculator"
    ]
    tool_span_by_input = {tuple(sorted(span["input"].items())): span for span in calculator_spans}
    nested_span_first = _find_span_by_name(spans, "nested_tool_work_2")
    nested_span_second = _find_span_by_name(spans, "nested_tool_work_10")

    assert tool_span_by_input[(("a", 2), ("b", 3), ("operation", "add"))]["span_id"] in nested_span_first["span_parents"]
    assert tool_span_by_input[(("a", 10), ("b", 5), ("operation", "add"))]["span_id"] in nested_span_second["span_parents"]


class TestAutoInstrumentClaudeAgentSDK:
    """Tests for auto_instrument() with Claude Agent SDK."""

    @pytest.mark.skipif(not CLAUDE_SDK_AVAILABLE, reason="Claude Agent SDK not installed")
    def test_auto_instrument_claude_agent_sdk(self):
        """Test auto_instrument patches Claude Agent SDK and creates spans."""
        verify_autoinstrument_script("test_auto_claude_agent_sdk.py")

@pytest.mark.skipif(not CLAUDE_SDK_AVAILABLE, reason="Claude Agent SDK not installed")
@pytest.mark.asyncio
async def test_setup_claude_agent_sdk_repro_import_before_setup(memory_logger, monkeypatch):
    """Regression test for https://github.com/braintrustdata/braintrust-sdk-python/issues/7."""
    assert not memory_logger.pop()
    original_client = claude_agent_sdk.ClaudeSDKClient
    original_tool_class = claude_agent_sdk.SdkMcpTool
    original_tool_fn = claude_agent_sdk.tool

    consumer_module_name = "test_issue7_repro_module"
    consumer_module = types.ModuleType(consumer_module_name)
    consumer_module.ClaudeSDKClient = original_client
    consumer_module.ClaudeAgentOptions = claude_agent_sdk.ClaudeAgentOptions
    consumer_module.SdkMcpTool = original_tool_class
    consumer_module.tool = original_tool_fn
    monkeypatch.setitem(sys.modules, consumer_module_name, consumer_module)

    loop_errors = []
    received_types = []

    try:
        assert setup_claude_agent_sdk(project=PROJECT_NAME, api_key=logger.TEST_API_KEY)
        assert getattr(consumer_module, "ClaudeSDKClient") is not original_client
        assert getattr(consumer_module, "SdkMcpTool") is not original_tool_class
        assert getattr(consumer_module, "tool") is not original_tool_fn
        assert claude_agent_sdk.SdkMcpTool is not original_tool_class
        assert claude_agent_sdk.tool is not original_tool_fn

        async def main() -> None:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(lambda loop, ctx: loop_errors.append(ctx.get("exception") or ctx.get("message")))

            options = getattr(consumer_module, "ClaudeAgentOptions")(
                model="claude-3-5-haiku-20241022",
                permission_mode="bypassPermissions",
            )
            transport = make_cassette_transport(
                cassette_name="test_auto_claude_agent_sdk",
                prompt="",
                options=options,
            )
            async with getattr(consumer_module, "ClaudeSDKClient")(options=options, transport=transport) as client:
                await client.query("Say hi")
                async for message in client.receive_response():
                    received_types.append(type(message).__name__)

        await main()
    finally:
        claude_agent_sdk.ClaudeSDKClient = original_client
        claude_agent_sdk.SdkMcpTool = original_tool_class
        claude_agent_sdk.tool = original_tool_fn

    assert loop_errors == []
    assert "AssistantMessage" in received_types
    assert received_types[-1] == "ResultMessage"

    spans = memory_logger.pop()
    task_spans = [s for s in spans if s["span_attributes"]["type"] == SpanTypeAttribute.TASK]
    assert len(task_spans) == 1
    assert task_spans[0]["span_attributes"]["name"] == "Claude Agent"
    assert task_spans[0]["input"] == "Say hi"
