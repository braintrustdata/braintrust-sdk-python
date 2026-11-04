# LiveKit cascade instrumentation

Install `braintrust` and optionally `braintrust-audio`, alongside LiveKit's
providers. This integration qualifies LiveKit Python **1.8.10** and supports
STT → text LLM → TTS. Call setup in each worker before starting sessions:

```python
import braintrust
from braintrust.integrations.livekit_agents import setup_livekit_agents

braintrust.init_logger(project="voice-agent")
setup_livekit_agents()
```

`braintrust.auto.auto_instrument()` enables the same hooks. No OpenTelemetry
exporter or custom agent subclass is required. If another exporter independently
sends LiveKit's native OTel scope to Braintrust, disable that scope there to avoid
duplicate traces.

## Recording and privacy

Each direction requires its own environment opt-in:

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
```

Installing codecs, configuring recording options, and the generic
`BRAINTRUST_CAPTURE_ATTACHMENTS` flag do not enable capture. LiveKit's effective
PII policy also applies: explicit `set_tracer_provider(..., allow_pii=False)`,
`LIVEKIT_TELEMETRY_ALLOW_PII`, project/job redaction, and session redaction can
withhold content. Project redaction wins. Tightening policy stops recording and
removes unexported content. Already uploaded content cannot be retracted.

LiveKit `record=False` and `record={"audio": False}` control LiveKit's destination.
They do not disable Braintrust's explicitly enabled capture. LiveKit redaction
applies across destinations.

**Audio upload uses `Attachment.upload()` directly.** Span customizers and logger
masking apply to subsequent trace records. Removing a reference does not prevent
or undo the upload. Control audio egress using the dedicated flags and LiveKit
privacy policy.

## Encoding and flushing

With the extension installed, recordings use Opus in Ogg automatically. Without
it, default capture warns once and falls back to raw PCM. Explicit Opus requires
a compatible extension. PCM and WAV do not require native media dependencies.

```python
from braintrust.audio import AudioEncoding, AudioRecordingOptions

setup_livekit_agents(audio=AudioRecordingOptions(
    segment_duration_seconds=60,
    flush_threshold_bytes=32 * 1024 * 1024,
    encoding=AudioEncoding(codec="opus", container="ogg"),
))
# Alternatives: AudioEncoding(codec="pcm", container=None), or container="wav".
```

Codec and container are separate. Unsupported combinations raise during explicit
setup. Repeated setup without arguments retains explicit configuration; active
calls keep their original snapshot.

The Opus backend supports mono/stereo, with up to 256000 bits/s per channel
(512000 bits/s for stereo). Explicit bitrates are validated against the enabled
recording directions before capture.

Duration and byte thresholds request proactive flushing. They are not capture
limits. Slow encoding, upload, or playback confirmation may temporarily retain
additional buffers; capture continues. Encoding and upload run off the voice
event loop. Each segment is independently playable, and ready metadata is
published only after confirmed upload success. Unavailable segments are omitted
with their known placement and a safe reason.

## Traces and media scope

Schema v3 emits `session`, `user_turn`, `agent_turn`, `user_speech`, `stt`,
`llm_response`, tool-name spans, `tts`, and playback-confirmed `agent_speech`.
Recording attachments live on the session. Speech selections use the shared
session timeline, including silence and real pauses. With both directions,
channel 0 is session input and channel 1 is agent output; a single enabled
direction uses channel 0. Unidentified/silent channels have empty source lists.

Input is audio consumed by the session. Output includes only audio the sink
confirms was played. Synchronized playback text is used when available; generated
text is never substituted for heard text. Relationships are optional and use
direct native IDs or explicit ownership. Legacy UUID span IDs omit W3C links.

Other room tracks, background audio outside session output, realtime/half-cascade,
duplex, and dedicated handling for mid-session stream replacement are outside
this integration's scope. Agent handoffs retaining media streams are supported.

Diarization attribution requires a known recognition-run scope. If native event
replacement loses that scope, speaker attribution is omitted. Transcript tracing
continues under the same privacy policy.

## Shutdown

`await session.aclose()` drains accepted encoding/upload work, queues terminal
metadata, then flushes the logger off the event loop. Job shutdown has an
idempotent fallback. Conversation timestamps exclude upload time. A configured
shutdown deadline marks unfinished segments omitted and fences late completions;
an upload already in progress may still finish. Abrupt process termination is
best effort.

Audio uploads use the existing `BRAINTRUST_HTTP_TIMEOUT` setting (60 seconds by
default) for each HTTP request. This bounds network inactivity independently of
the shutdown deadline; it cannot forcibly terminate a running native encoder or
an in-progress upload. Ordinary attachments retain their existing behavior.

An explicit final flush should follow session close:

```python
await session.aclose()
await asyncio.to_thread(braintrust.flush)
```

For local RoomIO testing on macOS, run the pinned `livekit/livekit-server:v1.11.0`
container with ports 7880, 7881 and 7882/udp mapped to localhost, then use
`BRAINTRUST_LIVEKIT_SERVER_EXTERNAL=1 mise exec -- uv run nox -s 'test_livekit_agents(latest)'`.

## Audio extension release

`integrations/audio-py` is an independent distribution. The `audio-py` CI workflow
tests Python 3.10–3.14 on Linux, Windows and macOS and builds its wheel/sdist.
`Publish Python audio extension` builds and validates a selected commit, defaults
to dry run, and uses the existing publishing environment plus PyPI OIDC when
explicitly dispatched for publication. Configure its PyPI trusted publisher
before the first release and release the core SDK containing this contract first.
