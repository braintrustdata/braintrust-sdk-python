from braintrust.trace import SpanData, SpanFilters, Trace


async def accepts_span_filters(trace: Trace) -> list[SpanData]:
    filters: SpanFilters = {"name": ["search"], "metadata": {"model": None}}
    await trace.get_spans(filters=filters)
    await trace.get_spans(span_type=["tool"], filters=filters)
    return await trace.get_spans(filters={"duration": {"min": 0.5}}, include_scorers=True)
