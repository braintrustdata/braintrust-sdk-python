# Pipecat tracing compatibility

`setup_pipecat()` and `wrap_pipeline_worker()` trace cascade operations through
Pipecat's observer, public frame-processing methods, and public STT/TTS async
generators. Provider SDK spans join the framework task in the executing service
task. Uncovered transports emit `pipecat.*` model children instead. Model usage
and timing belong on those children, not their task wrappers.

## Custom service clients

Pipecat does not expose a universal public accessor for the client that executes
an operation. Braintrust recognizes known native execution routes in the tested
Pipecat versions. For custom services that override these routes, declare the
actual executing client before running the pipeline:

```python
from braintrust.integrations.pipecat import set_model_client

# The same client your custom service actually calls:
set_model_client(my_llm_service, operation="llm_response", client=openai_client)

# A custom service that sends its own HTTP or WebSocket request:
set_model_client(my_tts_service, operation="tts", client=None)
```

Supported operations are `llm_response`, `stt`, and `tts`. A client covered by a
Braintrust integration retains ownership when that integration is disabled;
only the framework task is then emitted. An uncovered client, including `None`,
gets a framework model child. Do not bind an unused SDK client or a client used
only to read configuration. Update the binding before each operation if the
service changes its execution path. Ownership is captured when each operation
starts, so later configuration changes do not affect queued earlier operations.

An unknown, unbound custom route emits its framework task and a diagnostic;
Braintrust does not guess ownership from provider names, application bytecode,
or whether a provider span appeared. Provider instrumentation can still emit
its natural child under that task. Explicit binding is required to opt an
unknown transport into framework model spans. This is a deliberate replacement
for the earlier static bytecode inference.

## Remaining compatibility boundaries

- Pipecat 1.3 lacks the public `settings` accessor. The integration uses the
  legacy settings store only as a fallback.
- Known native request implementations have no public executing-client getter.
  `_compat.py` matches their implementation identities before reading the
  legacy client store. Application overrides require an explicit binding.
- Native transcript aggregators do not expose consumed transcription frame IDs
  when writing context. Small, capability-gated aggregation hooks retain exact
  frame/message lineage. Missing evidence omits optional links; matching text
  does not establish provenance.
- Pipecat 1.3 has no public event-handler removal API. Its single shared callback
  captures no observer or session, and uses only the originating event context.
  Newer versions unregister the callback during cleanup.
- LLM provider context uses a bounded observer handshake. Delayed delivery
  reserves the task identity for provider children while the queued observer
  supplies its actual parent and operation data. Shutdown closes any remaining
  pending task as incomplete, without extending the wait or buffering audio.

The GPT Live hooks are unchanged by the cascade compatibility refactor.
Non-OpenAI native adapters and WebSocket TTS require additional runtime coverage;
current provider-backed regressions use real OpenAI services and recorded HTTP
responses on Pipecat 1.3.0 and 1.16.0.
