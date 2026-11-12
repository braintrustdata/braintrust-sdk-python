# Span Design, Metrics, And Attachments

Repo-specific notes on top of the instrumentation spec (`braintrustdata/braintrust-spec` `skills/instrumentation-spec/references/instrumentation-guide.md`). The spec wins on any conflict.

## Span Design

### Type + name

- `llm` — one provider API call
- `task` — parent for an agent run, pipeline step, or named operation
- `tool` — model-initiated tool/function execution

Use spec-recommended names (`Chat Completion`, `anthropic.messages.create`, `generate_content`). For new providers, pick a stable provider-specific name.

### Completion-style vs agentic-style

- **Completion** (single request/response, user runs any tool calls): one `llm` span, no children. Tool calls appear in `output`.
- **Agentic** (SDK/framework runs the tool loop): one parent `task` wrapping child `llm` (per model call) + `tool` (per tool exec) spans in execution order.

Completion-style *frameworks* (LiteLLM, OpenRouter, etc.) still emit `llm` spans with `metadata.provider` set to the underlying provider, NOT the framework name. Agentic frameworks (Vercel AI SDK w/ tools, LangChain agents, OpenAI Agents SDK, Claude Agent SDK, ADK, Agno) emit the full parent+children tree.

### Payload format

Canonical shape is OpenAI Chat Completions format. Providers with dedicated UI normalizers (OpenAI, Anthropic, Google) MAY preserve provider-native payloads if `metadata.provider` is set correctly; all other providers MUST convert to OpenAI shape. If a provider passes system prompt separately, insert it as a `role: "system"` entry.

### Fields

- `input` — meaningful user request (messages / prompt / provider-native input)
- `output` — meaningful provider result (normalized, not opaque SDK instances)
- `metadata` — see below
- `metrics` — spec-listed keys only (see [Metrics](#metrics))
- `error` — pass the `Exception` instance directly to span logging; do not pre-format

**Every `llm` span MUST include `metadata.model` and `metadata.provider`** (`provider` = whose pricing applies, even when going through a gateway or framework).

**Tool definitions go in `metadata.tools`** (OpenAI-shaped regardless of underlying provider), NOT in `input` messages. Preserve provider-native built-in tool types (Anthropic `computer_use`, OpenAI `web_search`) as opaque JSON-serializable entries. Never log executable tool handlers.

**Prompt provenance goes in `metadata.prompt`** (`id`, `project_id`, `version`, `variables`, plus `prompt_session_id` for playground calls). Not in `input`, request payloads, or `metadata.tools`. Strip carrier fields like `span_info` before logging.

**`metadata` allowlist-per-provider:** capture the spec-defined keys plus provider-specific detail fields (request/response ids, safety data, model-family flags) chosen deliberately. Do not dump entire raw request/response objects. Redact secrets. If a provider-specific field ends up broadly useful, promote it to the spec.

### Token metrics

The integration that directly owns the model/provider API response owns token accounting. Orchestration/framework integrations should not log token metrics when underlying provider integrations create leaf spans with usage. Agentic parent `task` spans MAY aggregate across children when the framework doesn't delegate to a separately instrumented provider client. Do not add "if OpenAI is patched, skip metrics" checks — define clear ownership instead.

**Framework `llm`-like spans that delegate to a separately instrumented client:** if the framework's per-model-call span sits above a call the framework itself dispatches through another patched client (e.g. DSPy's LM callback → LiteLLM → OpenAI), do NOT type the framework's span `llm`. Two nested `llm` spans for one API call is confusing in the UI and — because the framework typically cannot supply tokens from its own callback contract — would produce a token-less `llm` span that violates the "tokens required for LLM spans" rule. Leave the framework span untyped (or `task`), keep `metadata.model` / `metadata.provider` on it for attribution, and let the underlying provider integration own the `llm` leaf with usage. The same logic applies to any framework that always delegates the transport layer to another instrumentable client.

### Shaping guidance

- Flatten positional args into named fields; normalize SDK objects into dicts/lists/scalars; drop noisy transport fields.
- Aggregate streaming chunks into one final `output` + stream-specific `metrics`. One `llm` span per API call, not per chunk.
- Do not over-serialize. Braintrust serializes at send/log time. Integration tracing only needs readable Python dicts/lists/scalars and materialized attachments.
- Keep wrapper bodies thin: prepare traced input, open span, call provider, normalize result, log.
- Do not wrap `span.log(...)` / `span.set_attributes(...)` in broad try/except. Braintrust span methods are boundary-safe.
- **Rerank exception to "no silent caps":** rerank result lists can be huge and are noisy in the span UI; cap the `output` list at a fixed max (e.g. 100 entries) via a named constant. The cap is compact-span hygiene, not coverage bounding — the model already ranked everything.


## Metrics

Only emit spec-listed keys:
- `tokens`, `prompt_tokens`, `completion_tokens` — required for LLM spans; values MUST be non-negative
- `time_to_first_token` — required for streaming; SDK measures from request start to first chunk
- `completion_reasoning_tokens` — required when provider reports reasoning tokens (o-series etc.)
- `prompt_cached_tokens`, `prompt_cache_creation_tokens`, `prompt_cache_creation_5m_tokens`, `prompt_cache_creation_1h_tokens` — provider prompt caching
- `prompt_audio_tokens`, `completion_audio_tokens`, `completion_image_tokens` — audio/image models when provider reports
- `start`, `end` — standard span timing

Map provider modality breakdowns only to standardized metrics: input audio → `prompt_audio_tokens`, output audio → `completion_audio_tokens`, and output image → `completion_image_tokens`. These are subsets of prompt/completion usage, not additional tokens. Preserve unstandardized modality breakdowns as deliberately named provider metadata rather than inventing metric keys.

For streaming, still produce one span per API call with accumulated `input`/`output`. Capture usage from stream metadata (e.g. OpenAI `stream_options.include_usage`) when surfaced.

For reasoning models, capture the full output structure (reasoning summaries + message blocks) and include prior reasoning in `input` for multi-turn calls.

## Multimodal And Attachments

Inline binary media (images, PDFs, audio, video) MUST be replaced by a Braintrust attachment at the leaf position. Do not add a separate top-level `attachments` list.

- Prefer `_materialize_attachment(...)` in `integrations/utils.py`. Don't reimplement base64/file decoding.
- Convert raw `bytes` / base64 / data URLs / file inputs / generated media → `braintrust.logger.Attachment`. Preserve remote URLs as strings — do not fetch to create an attachment.
- Payload shape after materialization:
  - images → `{"image_url": {"url": attachment}}`
  - non-image media/documents/files → `{"file": {"file_data": attachment, "filename": resolved.filename}}`
- Do not force non-image payloads through `image_url`.
- If materialization fails, keep the original value; never drop, null it out, or raise.
- Preserve non-attachment values while walking nested payloads unless intentionally normalizing.
- Keep MIME type, size, safety data, filenames, provider ids next to the attachment.
- Providers with UI normalizers (OpenAI, Anthropic, Google) — MAY preserve the provider-native structure and only replace the raw media leaf.
- Generated media → log the attachment in `output`, not `metadata`. For streaming, aggregate into one final `output` attachment.
