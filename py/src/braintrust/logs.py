"""Adapters for forwarding standard-library log records to Braintrust."""

import logging
from typing import Any

from .logger import Logger, LogLevel


_STANDARD_LOG_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", logging.NOTSET, "", 0, "", (), None))) | {
    "asctime",
    "message",
}
_IGNORED_LOGGER_PREFIXES = ("braintrust", "urllib3")


def _log_level(level: int) -> LogLevel:
    if level >= logging.CRITICAL:
        return "fatal"
    if level >= logging.ERROR:
        return "error"
    if level >= logging.WARNING:
        return "warn"
    if level >= logging.INFO:
        return "info"
    if level >= logging.DEBUG:
        return "debug"
    return "trace"


def _is_ignored_logger(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in _IGNORED_LOGGER_PREFIXES)


def _record_metadata(record: logging.LogRecord) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_LOG_RECORD_ATTRIBUTES and not key.startswith("_")
    }

    if record.args and isinstance(record.msg, str):
        metadata["braintrust.template"] = record.msg
        parameters = record.args.items() if isinstance(record.args, dict) else enumerate(record.args)
        metadata.update({f"braintrust.template.parameter.{key}": value for key, value in parameters})

    metadata.update(
        {
            "logger.name": record.name,
            "code.file.path": record.pathname,
            "code.function.name": record.funcName,
            "code.line.number": record.lineno,
        }
    )

    return metadata


class BraintrustLogHandler(logging.Handler):
    """Forward Python ``logging`` records to a Braintrust logger.

    Attach this handler explicitly with ``logging.Logger.addHandler``. Records
    emitted by Braintrust and its HTTP transport are ignored to prevent logging
    recursion.
    """

    def __init__(self, logger: Logger, level: int | str = logging.NOTSET):
        super().__init__(level=level)
        self._logger = logger

    def emit(self, record: logging.LogRecord) -> None:
        if _is_ignored_logger(record.name):
            return

        try:
            self._logger._emit_log_record(
                body=self.format(record),
                level=_log_level(record.levelno),
                metadata=_record_metadata(record),
                captured_at=record.created,
            )
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self._logger.flush()


__all__ = ["BraintrustLogHandler"]
