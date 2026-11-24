import dataclasses
import json
from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict, Union, cast

import jsonschema


JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]


class VoiceCall(TypedDict):
    """Serializable, single-use call context. Forward securely; do not log it."""

    connection_url: str
    session_token: str
    expires_at: str
    session_id: str
    span_id: str
    parent: str
    timeout_seconds: int


class CallerTurn(TypedDict):
    id: str
    speaker: Literal["caller"]
    text: str


class VoiceCallResult(TypedDict):
    agent_output: JSONValue
    caller_transcript: list[CallerTurn]
    recording: dict[str, JSONValue]
    reason: str
    session_id: str
    span_id: str


class SimulatorOutput(TypedDict):
    transcript: list[CallerTurn]
    recording: dict[str, JSONValue]
    reason: str


_TRANSCRIPT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["id", "speaker", "text"],
        "properties": {"id": {"type": "string"}, "speaker": {"const": "caller"}, "text": {"type": "string"}},
    },
}
_RECORDING_SCHEMA = {
    "type": "object",
    "required": ["audio", "metadata"],
    "properties": {"audio": {"type": "object"}, "metadata": {"type": "object"}},
}


def parse_output(value: object) -> SimulatorOutput:
    jsonschema.validate(
        value,
        {
            "type": "object",
            "required": ["transcript", "recording", "reason"],
            "properties": {
                "transcript": _TRANSCRIPT_SCHEMA,
                "recording": _RECORDING_SCHEMA,
                "reason": {"type": "string"},
            },
        },
    )
    return cast(SimulatorOutput, value)


def parse_result(value: object) -> VoiceCallResult:
    json.dumps(value, allow_nan=False)
    jsonschema.validate(
        value,
        {
            "type": "object",
            "required": list(VoiceCallResult.__required_keys__),
            "properties": {
                "caller_transcript": _TRANSCRIPT_SCHEMA,
                "recording": _RECORDING_SCHEMA,
                **{key: {"type": "string"} for key in ["reason", "session_id", "span_id"]},
            },
        },
    )
    return cast(VoiceCallResult, value)


@dataclasses.dataclass(frozen=True)
class SimulatedUser:
    name: str
    instructions: str
    voice: dict[str, JSONValue] | None = None

    def __post_init__(self):
        if not self.name.strip() or not self.instructions.strip():
            raise ValueError("SimulatedUser requires a name and instructions")


@dataclasses.dataclass(frozen=True)
class SimulatedUserRef:
    id: str | None = None
    name: str | None = None

    def __post_init__(self):
        if bool(self.id) == bool(self.name):
            raise ValueError("Specify exactly one simulated user ID or name")


@dataclasses.dataclass(frozen=True)
class LiveKitAgent:
    server_url: str
    agent_name: str
    api_key: str | None = dataclasses.field(default=None, repr=False)
    api_secret: str | None = dataclasses.field(default=None, repr=False)

    def __post_init__(self):
        if not self.server_url.strip() or not self.agent_name.strip():
            raise ValueError("LiveKitAgent requires a server URL and agent name")


AgentConnector = Callable[[VoiceCall], Awaitable[VoiceCallResult]]
Caller = Union[SimulatedUser, SimulatedUserRef]
Initiator = Literal["agent", "simulated_user"]
