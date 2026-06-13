import io
import math
import os
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast


T = TypeVar("T")
EnvValue = bool | float | int | str
_Parser = Callable[[str], EnvValue | None]
BRAINTRUST_ENV_FILE = ".env.braintrust"
BRAINTRUST_ENV_SEARCH_PARENT_LIMIT = 64


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

    Empty or whitespace-only strings are treated as unset so callers fall back to their default.
    """
    return value if value.strip() else None


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

    def get(self, default: T, *, use_dotenv: bool = False) -> T:
        parsed = self._parse_value(os.environ.get(self.name))
        if parsed is not None:
            return cast(T, parsed)

        if use_dotenv:
            parsed = self._get_dotenv_value()
            if parsed is not None:
                return cast(T, parsed)

        return default

    def _parse_value(self, value: str | None) -> EnvValue | None:
        if value is None:
            return None
        return self.parser.parser(value)

    def _get_dotenv_value(self) -> EnvValue | None:
        try:
            directory = os.getcwd()
        except OSError:
            return None

        for _ in range(BRAINTRUST_ENV_SEARCH_PARENT_LIMIT + 1):
            env_path = os.path.join(directory, BRAINTRUST_ENV_FILE)
            try:
                with open(env_path, encoding="utf-8") as f:
                    return self._parse_dotenv_contents(f.read())
            except FileNotFoundError:
                pass
            except OSError:
                return None

            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent

        return None

    def _parse_dotenv_contents(self, contents: str) -> EnvValue | None:
        try:
            from dotenv import dotenv_values

            parsed = dotenv_values(stream=io.StringIO(contents), interpolate=False)
            return self._parse_value(parsed.get(self.name))
        except ImportError:
            pass
        except Exception:
            return None

        for line in contents.splitlines():
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].lstrip()
            if "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            if key.strip() != self.name:
                continue

            lexer = shlex.shlex(value.lstrip(), posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            try:
                parts = list(lexer)
            except ValueError:
                return None
            if not parts:
                return None
            return self._parse_value(parts[0])

        return None


class BraintrustEnv:
    API_KEY = EnvVar("BRAINTRUST_API_KEY", EnvParser.STRING)
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
