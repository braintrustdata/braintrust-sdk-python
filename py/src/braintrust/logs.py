"""Adapters for forwarding standard-library log records to Braintrust."""

import logging
import threading
from collections.abc import Mapping
from typing import Any

from .api._transport import _is_internal_http_transport
from .logger import Logger, LogLevel


_STANDARD_LOG_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", logging.NOTSET, "", 0, "", (), None))) | {
    "asctime",
    "message",
}
_IGNORED_LOGGER_PREFIXES = ("braintrust",)
_INTERNAL_HTTP_TRANSPORT_RECORD_ATTRIBUTE = "_braintrust_internal_http_transport"
_INTERNAL_HTTP_TRANSPORT_FACTORY_ATTRIBUTE = "_braintrust_internal_http_transport_factory"
_LOG_RECORD_FACTORY_LOCK = threading.Lock()


def _install_internal_http_transport_record_factory() -> None:
    with _LOG_RECORD_FACTORY_LOCK:
        current_factory = logging.getLogRecordFactory()
        if getattr(current_factory, _INTERNAL_HTTP_TRANSPORT_FACTORY_ATTRIBUTE, False):
            return

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = current_factory(*args, **kwargs)
            if _is_internal_http_transport():
                setattr(record, _INTERNAL_HTTP_TRANSPORT_RECORD_ATTRIBUTE, True)
            return record

        setattr(record_factory, _INTERNAL_HTTP_TRANSPORT_FACTORY_ATTRIBUTE, True)
        logging.setLogRecordFactory(record_factory)


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


def _percent_parameter_kinds(template: str) -> tuple[bool, bool]:
    uses_positional = False
    uses_named = False
    index = 0
    while index < len(template):
        if template[index] != "%":
            index += 1
            continue

        percent_run_start = index
        while index < len(template) and template[index] == "%":
            index += 1
        if (index - percent_run_start) % 2 == 0 or index == len(template):
            continue

        if template[index] == "(":
            uses_named = True
        else:
            uses_positional = True
        if uses_positional and uses_named:
            break

    return uses_positional, uses_named


class _InternalLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            not _is_ignored_logger(record.name)
            and not getattr(record, _INTERNAL_HTTP_TRANSPORT_RECORD_ATTRIBUTE, False)
            and not _is_internal_http_transport()
        )


def _record_metadata(record: logging.LogRecord) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_LOG_RECORD_ATTRIBUTES and not key.startswith("_")
    }

    if record.args and isinstance(record.msg, str):
        metadata["braintrust.template"] = record.msg
        if isinstance(record.args, Mapping):
            uses_positional, uses_named = _percent_parameter_kinds(record.msg)
            parameters = []
            if uses_positional or not uses_named:
                parameters.append((0, record.args))
            if uses_named:
                parameters.extend(record.args.items())
        else:
            parameters = enumerate(record.args)
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
    emitted by Braintrust or while its HTTP transport is active are ignored to
    prevent logging recursion.
    """

    def __init__(self, logger: Logger, level: int | str = logging.NOTSET):
        super().__init__(level=level)
        _install_internal_http_transport_record_factory()
        self._logger = logger
        # Handler.handle() runs filters before acquiring its lock. Filtering
        # internal transport logs here prevents a shutdown flush from waiting
        # on a worker thread blocked on that same lock.
        self.addFilter(_InternalLogFilter())

    def emit(self, record: logging.LogRecord) -> None:
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
