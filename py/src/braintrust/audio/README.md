# Shared audio recording components

Internal SDK components, alongside the shared logger and other `braintrust` packages. Framework integrations depend on this package; it imports no framework. No new public setup API is introduced.

```text
Framework integration          audio/                    logger.py
  native lifecycle hooks          recording: PCM/encoding
  transport/sample association →  alignment: sample ranges
  turn/provider metadata          budget: retained bytes
                                  worker: bounded jobs
  span + recording descriptors ─────────────────────────→ Attachment / flush
```

`CallRecording.capture(channel, pcm, sample_rate, channels, ...)` accepts signed 16-bit little-endian PCM and local observation times. The current two-channel recorder assumes continuous input on channel 0 and intermittent output on channel 1, produces 24 kHz stereo, and returns sample ranges on that clock. Integrations must verify those assumptions and preserve sample provenance. It is not a universal clock-reconciliation or media-playback layer.

NumPy and SoundFile are imported only during encoding. Both WAV and Ogg currently use them; `braintrust[audio]` installs the optional dependencies. SoundFile uses CFFI and native libsndfile; Ogg requires a libsndfile build with Opus support. No FFmpeg subprocess is used. Simple existing PCM-to-WAV attachment helpers elsewhere in the SDK remain dependency-free.

Capture budgets and worker admission are shared across integrations in the process. Source PCM and encoded attachments have separate process-wide budgets. Encoded bytes remain budgeted for the lifetime of the exporter-owned attachment. Neither budget alone bounds total RSS or encoder scratch space. Frame hooks, capture opt-in, metadata filtering, span parentage, and framework shutdown remain the integration's responsibility. Attachment transport stays in the existing SDK logger.

Run `nox -s test_audio` for encoding, resource, and dependency-boundary tests. Dependency-free tests also run in `test_core`; that session excludes the codec-dependent recording tests.
