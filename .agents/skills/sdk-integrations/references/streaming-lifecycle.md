# Streaming Lifecycle Review

Open this when an integration wraps a streamed response or tracks an event-driven session or response.

## Wrapped streams

Review every exit path, not just full consumption. Check the real provider's
iterator, context-manager, cancellation, and garbage-collection behavior before
choosing a wrapper. A proxy can preserve transport cleanup while still losing
the span's final output and end time.

- Cover exhaustion, provider errors, explicit close/cancel, and context-manager
  exit where the provider supports it. Preserve exception and return semantics.
- Cover `break` followed by dropping the last stream reference, and dropping a
  stream before consuming any chunks. `break` alone does not close a retained
  iterator; do not promise immediate finalization while callers still hold it.
- Use existing recordings for partial-consumption tests. Drop the proxy, force
  collection, and assert partial (or empty) output, `metrics.end`, correct
  parentage, and no duplicate finalization after explicit close or exhaustion.
- A GC fallback must not retain the stream through its callback or closure.
  For async streams, do not run or schedule event-loop work from a finalizer;
  finalize trace state and preserve the provider's own cleanup behavior.
- Check that neither iteration nor cleanup leaves the stream span current in
  the caller's context. GC is a best-effort fallback, not a substitute for
  deterministic cleanup when the caller explicitly closes the stream.

## Event-driven sessions and responses

- Identify which events end remote work and which end local processing. If queued events can arrive after the remote model ends, keep a draining phase until framework teardown. Give each lifecycle explicit phases when ordered transitions would otherwise be spread across several flags. Track independent completion signals together on the record that owns them and allow either arrival order.
- Keep correlation state for each entity together. Remove active entries when their work completes. Bound recent routing history for long-lived sessions, keeping entries needed by in-flight work. Preserve transcript and output content required for span export.
- On normal teardown and cancellation, finish pending spans once and release tracker-owned payload buffers and indexes. Extend a recorded pipeline test to verify cleanup after teardown. Retain a reference to the tracker when checking that cleanup releases its state.
