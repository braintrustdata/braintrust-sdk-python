# Streaming Lifecycle Review

Open this when an integration wraps a streamed response (sync or async iterator, context manager, or provider stream object).

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
