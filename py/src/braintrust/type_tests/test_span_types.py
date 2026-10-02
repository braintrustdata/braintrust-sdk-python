import braintrust
from braintrust.logger import Span
from typing_extensions import assert_type


def span_context_manager_keeps_span_type() -> None:
    with braintrust.start_span(name="typed") as span:
        assert_type(span, Span)
        span.log(output="ok")


def start_span_accepts_extracted_trace_context(headers: dict[str, str]) -> Span:
    return braintrust.start_span(name="child", parent=braintrust.extract_trace_context(headers))
