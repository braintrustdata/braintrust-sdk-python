"""Adapters for forwarding standard-library log records to Braintrust."""

import logging
import threading
import weakref
from collections.abc import Mapping
from typing import Any

from .api._transport import _is_internal_http_transport
from .logger import Logger, LogLevel


_STANDARD_LOG_RECORD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", logging.NOTSET, "", 0, "", (), None))) | {
    "asctime",
    "message",
}
_IGNORED_LOGGER_PREFIXES = ("braintrust",)
_URLLIB3_TRANSPORT_LOGGERS = ("urllib3.connectionpool", "urllib3.connection")
_INTERNAL_HTTP_TRANSPORT_RECORD_ATTRIBUTE = "_braintrust_internal_http_transport"
_LOG_RECORD_FACTORY_ATTRIBUTE = "_braintrust_log_record_factory"
_TEMPLATE_RECORD_ATTRIBUTE = "_braintrust_template"
_TEMPLATE_ARGUMENTS_RECORD_ATTRIBUTE = "_braintrust_template_arguments"
_SPAN_CONTEXTS_RECORD_ATTRIBUTE = "_braintrust_span_contexts"
_LOGGING_HOOKS_LOCK = threading.Lock()
_CONTEXT_MANAGERS: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
_MISSING = object()


def _install_log_record_factory(logger: Logger) -> None:
    context_manager = logger.state.context_manager
    with _LOGGING_HOOKS_LOCK:
        _CONTEXT_MANAGERS[logger.state.id] = context_manager
        current_factory = logging.getLogRecordFactory()
        if getattr(current_factory, _LOG_RECORD_FACTORY_ATTRIBUTE, False):
            return

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = current_factory(*args, **kwargs)
            if record.args and isinstance(record.msg, str):
                setattr(record, _TEMPLATE_RECORD_ATTRIBUTE, record.msg)
                setattr(record, _TEMPLATE_ARGUMENTS_RECORD_ATTRIBUTE, record.args)

            with _LOGGING_HOOKS_LOCK:
                context_managers = tuple(_CONTEXT_MANAGERS.items())
            span_contexts: dict[str, tuple[str, str] | None] = {}
            for state_id, context_manager in context_managers:
                span_info = context_manager.get_current_span_info()
                span_contexts[state_id] = (span_info.trace_id, span_info.span_id) if span_info else None
            setattr(record, _SPAN_CONTEXTS_RECORD_ATTRIBUTE, span_contexts)
            return record

        setattr(record_factory, _LOG_RECORD_FACTORY_ATTRIBUTE, True)
        logging.setLogRecordFactory(record_factory)


class _InternalHTTPTransportMarker(logging.Filter):
    """Tag Braintrust transport records without hiding them from other handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _is_internal_http_transport():
            setattr(record, _INTERNAL_HTTP_TRANSPORT_RECORD_ATTRIBUTE, True)
        return True


_INTERNAL_HTTP_TRANSPORT_MARKER = _InternalHTTPTransportMarker()


def _install_internal_http_transport_marker() -> None:
    with _LOGGING_HOOKS_LOCK:
        for logger_name in _URLLIB3_TRANSPORT_LOGGERS:
            source_logger = logging.getLogger(logger_name)
            if _INTERNAL_HTTP_TRANSPORT_MARKER not in source_logger.filters:
                source_logger.addFilter(_INTERNAL_HTTP_TRANSPORT_MARKER)


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
        return not _is_ignored_logger(record.name) and not getattr(
            record, _INTERNAL_HTTP_TRANSPORT_RECORD_ATTRIBUTE, False
        )


def _record_metadata(record: logging.LogRecord) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_LOG_RECORD_ATTRIBUTES and not key.startswith("_")
    }

    template = getattr(record, _TEMPLATE_RECORD_ATTRIBUTE, record.msg)
    template_arguments = getattr(record, _TEMPLATE_ARGUMENTS_RECORD_ATTRIBUTE, record.args)
    if template_arguments and isinstance(template, str):
        metadata["braintrust.template"] = template
        if isinstance(template_arguments, Mapping):
            uses_positional, uses_named = _percent_parameter_kinds(template)
            parameters = []
            if uses_positional or not uses_named:
                parameters.append((0, template_arguments))
            if uses_named:
                parameters.extend(template_arguments.items())
        else:
            parameters = enumerate(template_arguments)
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
    emitted by Braintrust or by urllib3 for Braintrust HTTP transport calls are
    ignored to prevent logging recursion.
    """

    def __init__(self, logger: Logger, level: int | str = logging.NOTSET):
        super().__init__(level=level)
        self._logger = logger
        _install_log_record_factory(logger)
        _install_internal_http_transport_marker()
        # Handler.handle() runs filters before acquiring its lock. Filtering
        # internal transport logs here prevents a shutdown flush from waiting
        # on a worker thread blocked on that same lock.
        self.addFilter(_InternalLogFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            span_contexts = getattr(record, _SPAN_CONTEXTS_RECORD_ATTRIBUTE, {})
            span_context = span_contexts.get(self._logger.state.id, _MISSING)
            span_kwargs: dict[str, Any] = {}
            if span_context is not _MISSING:
                span_kwargs["lookup_current_span"] = False
                if span_context is not None:
                    span_kwargs["root_span_id"], span_kwargs["span_id"] = span_context
            self._logger._emit_log_record(
                body=self.format(record),
                level=_log_level(record.levelno),
                metadata=_record_metadata(record),
                captured_at=record.created,
                **span_kwargs,
            )
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self._logger.flush()


__all__ = ["BraintrustLogHandler"]
