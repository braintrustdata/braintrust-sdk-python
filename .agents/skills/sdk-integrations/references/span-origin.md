# Span Origin

Every span an integration creates MUST carry `context.span_origin.instrumentation.name = "<provider>-auto"` (e.g. `openai-auto`, `anthropic-auto`, `adk-auto`, `google-genai-auto`). Match the JS SDK exactly for cross-SDK filtering.

`start_span(...)` and all provider-level `start_span` methods accept `internal: SpanInternalOptions | None`, a TypedDict reserved for SDK internals — external callers should not use it. Integrations pass `internal={"instrumentation": "<provider>-auto"}`. When unset, spans fall back to the channel default (`braintrust-python-logger` / `braintrust-python-otel`).

The name does NOT propagate through the parent/child edge — each `start_span` call decides independently. User-owned spans nested inside a wrapped call (e.g. a `@traced` scorer) must NOT inherit the integration's name.

Standard pattern at the top of `tracing.py` (or `callbacks.py` / `plugin.py`):

```python
from braintrust.logger import start_span as _bt_start_span

_INSTRUMENTATION = "<provider>-auto"


def start_span(*args, **kwargs):
    internal = dict(kwargs.get("internal") or {})
    internal.setdefault("instrumentation", _INSTRUMENTATION)
    kwargs["internal"] = internal
    return _bt_start_span(*args, **kwargs)
```

Module-level `start_span(...)` calls flow through automatically. Calls that go through a `Logger` / `Experiment` / parent-`Span` instance (`logger.start_span(...)`, `parent.start_span(...)`) need `internal={"instrumentation": _INSTRUMENTATION}` explicitly. Tests assert the exported provenance: `span["context"]["span_origin"]["instrumentation"]["name"] == "<provider>-auto"`.
