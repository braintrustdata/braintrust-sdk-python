from dataclasses import dataclass
from typing import Any

from braintrust.span_types import SpanTypeAttribute


@dataclass(frozen=True)
class BenchmarkDataclass:
    model: str
    temperature: float
    tags: list[str]
    metadata: dict[str, Any]


class PydanticLikeV2:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        if not exclude_none:
            return dict(self._payload)
        return {k: v for k, v in self._payload.items() if v is not None}


class PydanticLikeV1:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def dict(self, exclude_none: bool = True) -> dict[str, Any]:
        if not exclude_none:
            return dict(self._payload)
        return {k: v for k, v in self._payload.items() if v is not None}


class StringifiableKey:
    def __init__(self, label: str):
        self.label = label

    def __str__(self) -> str:
        return f"key:{self.label}"


class NonStringifiableKey:
    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


def make_small_payload() -> dict[str, Any]:
    return {
        "input": {"prompt": "Summarize this email", "attempt": 1},
        "metadata": {"user_id": "user-123", "session_id": "sess-123"},
        "scores": {"helpfulness": 0.9},
        "tags": ["support", "email"],
    }


def make_medium_payload() -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarize the following issue thread."},
    ]
    for idx in range(8):
        messages.append(
            {
                "role": "assistant" if idx % 2 else "user",
                "content": f"message-{idx}",
                "metadata": {
                    "turn": idx,
                    "token_count": 64 + idx,
                    "tool_calls": [{"name": "lookup", "args": {"id": idx}}],
                },
            }
        )

    return {
        "input": {"messages": messages},
        "metadata": {
            "user_id": "user-123",
            "session_id": "sess-123",
            "workspace_id": "workspace-456",
            "feature_flags": {"structured_output": True, "tool_calling": True},
        },
        "metrics": {"prompt_tokens": 512, "completion_tokens": 128, "latency_ms": 183.4},
        "span_attributes": {"type": "llm", "model": "gpt-4.1", "provider": "openai"},
        "tags": ["support", "email", "benchmark"],
    }


def make_large_payload() -> dict[str, Any]:
    messages = []
    for idx in range(48):
        messages.append(
            {
                "role": "assistant" if idx % 2 else "user",
                "content": f"message-{idx}-" + ("x" * 80),
                "metadata": {
                    "turn": idx,
                    "token_count": 256 + idx,
                    "tool_calls": [
                        {"name": "lookup", "args": {"id": idx, "scope": "thread"}},
                        {"name": "render", "args": {"template": "summary", "format": "markdown"}},
                    ],
                },
            }
        )

    docs = []
    for idx in range(20):
        docs.append(
            {
                "id": f"doc-{idx}",
                "title": f"Document {idx}",
                "score": 0.95 - idx / 100,
                "metadata": {"source": "kb", "lang": "en", "chunk": idx},
            }
        )

    return {
        "input": {"messages": messages, "retrieved_documents": docs},
        "output": {"summary": "done", "citations": [doc["id"] for doc in docs]},
        "metadata": {
            "user_id": "user-123",
            "session_id": "sess-123",
            "workspace_id": "workspace-456",
            "feature_flags": {
                "structured_output": True,
                "tool_calling": True,
                "reasoning_tokens": True,
            },
            "routing": {"tier": "premium", "region": "us-west-2", "experiment": "bench-large"},
        },
        "metrics": {"prompt_tokens": 4096, "completion_tokens": 640, "latency_ms": 812.7},
        "span_attributes": {"type": "llm", "model": "gpt-4.1", "provider": "openai"},
        "tags": ["support", "email", "benchmark", "large"],
    }


def make_circular_payload() -> dict[str, Any]:
    payload = make_medium_payload()
    payload["self"] = payload
    payload["input"]["parent"] = payload["input"]
    return payload


def make_non_string_key_payload() -> dict[Any, Any]:
    return {
        1: "integer-key",
        ("tuple", "key"): {"nested": True},
        StringifiableKey("custom"): [1, 2, 3],
        NonStringifiableKey(): "fallback-key-path",
    }


def make_dataclass_value() -> BenchmarkDataclass:
    return BenchmarkDataclass(
        model="gpt-4.1",
        temperature=0.2,
        tags=["support", "triage"],
        metadata={"user_id": "user-123", "session_id": "sess-123"},
    )


def make_pydantic_v2_like_value() -> PydanticLikeV2:
    return PydanticLikeV2(
        {
            "model": "gpt-4.1",
            "temperature": 0.2,
            "user_id": "user-123",
            "optional_field": None,
        }
    )


def make_pydantic_v1_like_value() -> PydanticLikeV1:
    return PydanticLikeV1(
        {
            "model": "gpt-4.1",
            "temperature": 0.2,
            "user_id": "user-123",
            "optional_field": None,
        }
    )


def make_to_bt_safe_cases() -> list[tuple[str, Any]]:
    return [
        ("primitive-int", 42),
        ("primitive-float-nan", float("nan")),
        ("str-subclass-enum", SpanTypeAttribute.TOOL),
        ("dataclass", make_dataclass_value()),
        ("pydantic-v2-like", make_pydantic_v2_like_value()),
        ("pydantic-v1-like", make_pydantic_v1_like_value()),
    ]


def make_bt_safe_deep_copy_cases() -> list[tuple[str, Any]]:
    return [
        ("small", make_small_payload()),
        ("medium", make_medium_payload()),
        ("large", make_large_payload()),
        ("circular", make_circular_payload()),
        ("non-string-keys", make_non_string_key_payload()),
    ]
