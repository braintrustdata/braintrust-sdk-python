import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast


T = TypeVar("T")
EnvValue = bool | float | int | str
_Parser = Callable[[str], EnvValue | None]


def parse_float(value: str) -> float | None:
    """Parse a finite float from a string."""
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def parse_int(value: str) -> int | None:
    """Parse an integer from a string."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_bool(value: str) -> bool | None:
    """Parse common boolean environment variable values.

    Accepted true values: true, 1, yes, y, on.
    Accepted false values: false, 0, no, n, off.
    Empty or unrecognized values are invalid and fall back to the EnvVar default.
    """
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y", "on"):
        return True
    if normalized in ("false", "0", "no", "n", "off"):
        return False
    return None


def parse_string(value: str) -> str | None:
    """Parse a string environment variable.

    Empty strings are treated as unset so callers fall back to their default.
    """
    return value or None


class EnvParser(Enum):
    FLOAT = (parse_float,)
    INT = (parse_int,)
    BOOL = (parse_bool,)
    STRING = (parse_string,)

    def __init__(self, parser: _Parser):
        self.parser = parser


@dataclass(frozen=True)
class EnvVar:
    name: str
    parser: EnvParser

    def get(self, default: T) -> T:
        value = os.environ.get(self.name)
        if value is None:
            return default

        parsed = self.parser.parser(value)
        if parsed is None:
            return default
        return cast(T, parsed)


class BraintrustEnv:
    HTTP_TIMEOUT = EnvVar("BRAINTRUST_HTTP_TIMEOUT", EnvParser.FLOAT)
    SYNC_FLUSH = EnvVar("BRAINTRUST_SYNC_FLUSH", EnvParser.BOOL)
    MAX_REQUEST_SIZE = EnvVar("BRAINTRUST_MAX_REQUEST_SIZE", EnvParser.INT)
    DEFAULT_BATCH_SIZE = EnvVar("BRAINTRUST_DEFAULT_BATCH_SIZE", EnvParser.INT)
    NUM_RETRIES = EnvVar("BRAINTRUST_NUM_RETRIES", EnvParser.INT)
    QUEUE_SIZE = EnvVar("BRAINTRUST_QUEUE_SIZE", EnvParser.INT)
    QUEUE_DROP_LOGGING_PERIOD = EnvVar("BRAINTRUST_QUEUE_DROP_LOGGING_PERIOD", EnvParser.FLOAT)
    FAILED_PUBLISH_PAYLOADS_DIR = EnvVar("BRAINTRUST_FAILED_PUBLISH_PAYLOADS_DIR", EnvParser.STRING)
    ALL_PUBLISH_PAYLOADS_DIR = EnvVar("BRAINTRUST_ALL_PUBLISH_PAYLOADS_DIR", EnvParser.STRING)
    DISABLE_ATEXIT_FLUSH = EnvVar("BRAINTRUST_DISABLE_ATEXIT_FLUSH", EnvParser.BOOL)
    OTEL_COMPAT = EnvVar("BRAINTRUST_OTEL_COMPAT", EnvParser.BOOL)
