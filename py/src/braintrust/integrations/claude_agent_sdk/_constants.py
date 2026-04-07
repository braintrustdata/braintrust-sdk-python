from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


class MessageClassName(str, Enum):
    ASSISTANT = "AssistantMessage"
    USER = "UserMessage"
    RESULT = "ResultMessage"
    SYSTEM = "SystemMessage"
    TASK_STARTED = "TaskStartedMessage"
    TASK_PROGRESS = "TaskProgressMessage"
    TASK_NOTIFICATION = "TaskNotificationMessage"


class BlockClassName(str, Enum):
    TEXT = "TextBlock"
    THINKING = "ThinkingBlock"
    TOOL_USE = "ToolUseBlock"
    TOOL_RESULT = "ToolResultBlock"


class SerializedContentType(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True)
class ToolMetadataKeys:
    tool_name: str = "gen_ai.tool.name"
    tool_call_id: str = "gen_ai.tool.call.id"
    raw_tool_name: str = "raw_tool_name"
    operation_name: str = "gen_ai.operation.name"
    mcp_method_name: str = "mcp.method.name"
    mcp_server: str = "mcp.server"


@dataclass(frozen=True)
class MCPToolMetadataValues:
    operation_name: str = "execute_tool"
    method_name: str = "tools/call"


DEFAULT_TOOL_NAME: Final[str] = "unknown"

CLAUDE_AGENT_TASK_SPAN_NAME: Final[str] = "Claude Agent"
ANTHROPIC_MESSAGES_CREATE_SPAN_NAME: Final[str] = "anthropic.messages.create"

MCP_TOOL_PREFIX: Final[str] = "mcp__"
MCP_TOOL_NAME_DELIMITER: Final[str] = "__"

TOOL_METADATA: Final[ToolMetadataKeys] = ToolMetadataKeys()
MCP_TOOL_METADATA: Final[MCPToolMetadataValues] = MCPToolMetadataValues()

SERIALIZED_CONTENT_TYPE_BY_BLOCK_CLASS: Final[Mapping[str, SerializedContentType]] = MappingProxyType(
    {
        BlockClassName.TEXT: SerializedContentType.TEXT,
        BlockClassName.THINKING: SerializedContentType.THINKING,
        BlockClassName.TOOL_USE: SerializedContentType.TOOL_USE,
        BlockClassName.TOOL_RESULT: SerializedContentType.TOOL_RESULT,
    }
)

SYSTEM_MESSAGE_TYPES: Final[frozenset[MessageClassName]] = frozenset(
    {
        MessageClassName.SYSTEM,
        MessageClassName.TASK_STARTED,
        MessageClassName.TASK_PROGRESS,
        MessageClassName.TASK_NOTIFICATION,
    }
)
