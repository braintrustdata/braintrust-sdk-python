# LiveKit cascade implementation handoff

Worktree: `/Users/david/src/braintrust/braintrust-sdk-python-livekit-cascade`.
Top branch: `feat/livekit-cascade-v3`; fetched Python main base:
`cf69e6139bc0c7d8fb08f05519b66f7760600ad8`. Changes are committed locally under
David Elner's authorship. Nothing was pushed or published, and the design
worktree was not modified.

## Local review stack

| Proposed PR | Branch | PR base | Commit organization |
| --- | --- | --- | --- |
| Common audio recording | `feat/audio-recording-core` | `main` at the fetched base above | `6c033ff1`: common implementation and tests |
| Optional audio codecs | `feat/audio-codecs` | `feat/audio-recording-core` | `83f3b9e0`: package, codec tests, CI and codec-only lock changes |
| LiveKit Python cascade | `feat/livekit-cascade-v3` | `feat/audio-codecs` | `394348f2`: mechanical version/cassette cleanup; `56799ced`: adapter and regressions; final documentation commit |

Each PR should be reviewed against the preceding branch. The local `main`
worktree was left untouched; use the fetched base SHA for the first local diff.
The first branch passed its audio and core suites from an isolated committed
checkout. The second branch passed audio and codec suites from that checkout.
LiveKit-specific README examples are added only in the third branch, where the
new setup interface exists. The final source tree preserves the validated beta
implementation. Branch isolation found one Python 3.14 test setup issue: the
provider-privacy probe now runs inside an async test so LiveKit can obtain an
event loop. This does not change product behavior. The final committed-source
checkout passed all 36 LiveKit tests on Python 3.14.7 with no skips.

## Delivered

Phase 1 adds framework-independent positioned PCM recording, source/participant/
speaker registries, v3 metadata, validated around-middleware, and the independent
`braintrust-audio` distribution. Normalization, WAV/Opus encoding and upload run
off the voice event loop. A 48 kHz sample clock preserves fractional placement,
mixed source rates, stereo downmix, silence and channel alignment. Duration/byte
thresholds request flushing with temporary overshoot. Two process-local workers
admit waiting sessions fairly; actual native consumers retain their buffers
through cancellation. Unavailable audio and terminal failures produce omitted
segments with known placement.

Phase 2 replaces provider-specific LiveKit patchers with direct framework hooks
for sessions, user/agent turns, speech, STT, text LLM, TTS and tools. Explicit
owners cross async boundaries. Canonical operations suppress duplicate provider
telemetry while preserving customer spans. Participants, sources and diarization
speakers have separate registries. Optional relationships use native tool-call
IDs or synthesis ownership, without historical payload lookup. Native turn
completion and preemptive continuation hooks preserve framework ownership.

The output tap stages immutable frame copies before capture awaits and commits
only sink-accepted, playback-confirmed samples. It handles interruptions,
pause/resume, retained-media handoffs, duplicate/reentrant callbacks and capture
acknowledgement races. A watermark holds segments while playback can contribute.
Only native synchronized text is used as heard text.

Effective LiveKit privacy applies before capture/upload and to queued trace
exports: explicit provider policy before/after setup, environment policy and
job/project/session redaction. Normal close and concurrent native job-shutdown
callbacks settle media, drain audio, queue terminal metadata, then flush the
ordinary logger. Conversation timestamps exclude upload work. Deadlines fence
late completion.

The scope update is implemented: `Attachment.upload()` runs in a worker;
references and ready metadata are logged only after success. Customizers apply
to resulting records; removal cannot veto or undo an upload. There are no
receipts, removal notifications or exporter drain barriers. Core changes add
private per-span privacy filters, provider ownership checks and an audio-only
HTTP timeout override. Ordinary attachment behavior has regression coverage.

## Customer setup

Use the core SDK containing this change and optionally `braintrust-audio`, with
`livekit-agents==1.8.10` and matching provider plugins. Enabled capture defaults
to Opus in Ogg with the extension; without it, warn once and use raw PCM.
Explicit unsupported configuration raises.

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
```

```python
import braintrust
from braintrust.integrations.livekit_agents import setup_livekit_agents

braintrust.init_logger(project="voice-agent")
setup_livekit_agents()
```

`braintrust.auto.auto_instrument()` uses the same hooks. No OTel exporter or
custom agent subclass is needed. The generic attachment flag does not enable
audio, and installation/configuration does not override privacy.

```python
from braintrust.audio import AudioEncoding, AudioRecordingOptions

setup_livekit_agents(audio=AudioRecordingOptions(
    segment_duration_seconds=60,
    flush_threshold_bytes=32 * 1024 * 1024,
    shutdown_timeout_seconds=30,
    encoding=AudioEncoding(codec="opus", container="ogg"),
))
```

PCM without a container and PCM/WAV need no native codecs. See the complete
[setup and privacy guide](livekit-cascade.md) and executable
[AgentServer example](../examples/livekit/cascade.py).

## Validation

Phase 1 was validated before starting phase 2. Core/common counts include the
beta cleanup; the codec matrix and wheel/build checks retain the initial
qualification results. Beta reruns are detailed in the review section below.

| Check | Result |
| --- | --- |
| Core nox session, without optional provider/media packages | 1,146 passed, 106 skipped, 12 xfailed |
| Common audio, including additional FIFO regression | 21 passed |
| Newest LiveKit nox qualification | 36 passed in the beta rerun, no skips; initial replay-only cassette audit found no unused recordings |
| Additional tool relationship tests | W3C and legacy-ID cases passed using recorded tool continuation |
| Additional RoomIO relationship checks | All 3 cascades passed with native playback-to-TTS links |
| Codecs on CPython 3.10.22, 3.11.18, 3.12.14, 3.13.18, 3.14.7 | 13 passed on each version |
| Types | Pyright 0 errors; mypy passed; 37 runtime tests passed |
| Core wheel, independent of editable source | 11 passed |
| Installed core and extension wheels, outside source tree | 13 codec tests passed |
| Core/extension wheel and sdist builds | Passed |
| Repository fixup, changed/new-file pre-commit, pylint | Passed; narrow suppressions for generated protobuf test members |
| Cassette cleanup and diff whitespace | Passed |

The beta LiveKit qualification ran on CPython 3.12 with SDK/plugins pinned to
1.8.10. Stack isolation subsequently qualified the same SDK on Python 3.14.7.
RoomIO tests use two real RTC rooms and local server
`livekit/livekit-server:v1.11.0`, image digest
`sha256:100b9a870616d02f5e3795b34e0b593b5054a26f8131a94fd3fa322ed3154b16`.
Recorded OpenAI HTTP exchanges supply STT/streamed LLM/TTS without chargeable
routine model calls. Real Silero VAD drives consumed microphone audio into the
cascade. Real JobContext/protocol objects exercise concurrent shutdown callbacks.
Narrow injections cover rare playback/capture error paths. Local attachment
tests intercept upload rather than sending audio to cloud storage.

Experiments found and fixed phantom missing tails from floating-point playback
offsets, omitted-tail re-encoding after terminal failure, shutdown ordering and
test logger isolation. Recorder tests cover 16/24/44.1/48 kHz, fractional
positions, split packets, downmix, silence, congestion without loss, delayed
watermarks, cancellation, privacy, upload failure, ready ordering and late fences.
Codec tests independently decode tiny tails through 60-second segments, default
and explicit Opus/Ogg, stereo separation and supported encoder options.

A fast span-lifecycle benchmark completed: root create/end about 16.8 us, small
payload 22.4 us, medium payload 36.9 us. This is a local smoke baseline, not a
controlled before/after comparison; pyperf reported sample noise.

## Refreshed sources and implementation choices

The complete reviewed design and current v3 schema were read; metadata.links and
segment states were rechecked. LiveKit 1.8.10 was verified as newest stable and
qualified against release source `ff962a019b6500e08f8f1fcd7f55673361bd9e91`.
TypeScript main was refreshed to `f42d9dcaa0fe8c67a6bc5b5a22f875ff98f908f3`
as an architecture/DX reference. Pipecat PR 836 head
`6d5155737bacb830eb794e9db3ea7e8349af0365` supplied selective worker prior art;
its obsolete schema was not ported.

PyAV is pinned to 16.1.0 instead of 19.x to cover CPython 3.10–3.14 wheels.
Codecs remain absent from core dependencies. The middleware contract is
versioned internally; no public leases, exact byte reservations or encoder
plumbing are required. Defaults are 60-second / 32 MiB proactive thresholds and
a 30-second shutdown deadline. Audio HTTP requests use the existing HTTP timeout
setting, defaulting to 60 seconds, without changing ordinary attachments.

## Remaining release validation and limits

This handoff does not claim every design release gate has been exercised:

- Recorded streaming-STT transport, multi-final STT and diarization identity
  reuse across recognition runs remain unqualified end to end. The hooks and
  scoped identity handling exist; the real input fixture uses nonstreaming
  OpenAI STT with VAD.
- Preemptive generation is exercised. Separate preemptive-discard/reuse races
  still need dedicated real conversation fixtures.
- Actual RoomIO, roomless tracing, retained-media handoff, interruption and
  manual resume are covered. Independent customer sinks and a complete native
  false-interruption detection/resume sequence remain unqualified.
- Codec tests ran on all five Python versions on macOS ARM. The added
  Linux/Windows/macOS CI matrix and publish workflow have not been executed
  from this unpushed worktree. LiveKit was locally qualified on Python 3.12/3.14;
  other eligible versions remain for CI.
- Local multiple-session isolation and worker contention are covered. Sustained
  concurrent-call production load bounds have not been benchmarked.
- Real cloud attachment upload and the fully credentialed example were not run.
  Local tests cover the real reference/API boundary; a core test intercepts all
  three attachment upload HTTP steps.
- Shutdown cannot kill a running native encoder or retract an in-flight upload.
  Timeout omission fences late ready publication. Abrupt termination is best
  effort.
- Realtime/half-cascade, duplex, unrelated tracks/background output and dedicated
  mid-session stream replacement remain outside scope.

Before publishing the extension, release a core SDK with this internal contract,
update its minimum core dependency from this checkout's 0.49.1 baseline to that
actual release, and configure its PyPI trusted publisher. The manual workflow
defaults to dry run and uses the existing publish approval environments. No
release workflow was dispatched.

## Customer beta review corrections

The beta pass preserves recorder/session-recorder/codec boundaries, direct span
ownership and attachment uploads, playback watermarks, staged frame ownership,
and cancellation/terminal-state handling. It makes these targeted corrections:

- Supplemental automatic provider calls inside tools carry the active privacy
  callable into the existing per-span export customizers. Queued input, output
  and errors are removed if privacy later tightens. Unrelated provider calls and
  customer-created spans retain their ordinary behavior. The recorded regression
  failed before this change and passed afterward, checking both unaffected cases.
- Observed `llm_response.input` is now a normalized chat-message array. Native
  function calls become assistant tool calls; outputs become tool-result
  messages. Text uses the model-facing raw text; inline images/audio, cache
  breakpoints and native bookkeeping are excluded. The existing recorded tool
  continuation test asserts the initial and continuation model inputs in both
  ID modes, including LiveKit's reserialized function arguments and tool result.
- Real full/resumed RoomIO tests inspect uploaded PCM, verify the expected
  samples, and compare sample counts and placement to speech selections. The
  resume test no longer accepts any positive duration. LiveKit 1.8.10 explicitly
  discards its device queue on pause, so the expected recording is the supplied
  600 ms minus that independently observed native discard. Selection duration
  allows only endpoint rounding (one ms per selection), and PCM counts allow
  one sample per selection. The pinned sink's discard is bounded by its 200 ms
  device queue plus one in-flight 50 ms native frame. No unplayed samples are
  added to reach the generated duration.
- If timestamp alignment replaces a SpeechEvent and loses recognition scope,
  diarization attribution is omitted. The mapper no longer invents a new scope
  for each final transcript. Scoped attribution remains available when ownership
  survives; this does not claim new end-to-end diarization qualification.
- Removed the redundant in-process media-import assertion; the isolated
  subprocess check remains. Added annotations to the touched chat converter,
  provider privacy customizer and upload-test boundary.

Beta validation completed:

| Check | Beta result |
| --- | --- |
| Focused recorded privacy and tool regressions | Red before the fixes, green afterward; both ID modes covered |
| Full pinned LiveKit nox session on Python 3.12.14 | 36 passed, no skips, including all 11 real RoomIO playback modes |
| Core/logger nox session on Python 3.14.7 | 1,146 passed, 106 expected skips, 12 xfailed |
| Common audio nox session | 21 passed; count decreased by one redundant import test |
| Codec nox session on Python 3.14.7 | 13 passed |
| Type checks | Pyright/mypy passed; 37 runtime tests passed |
| Touched-file pylint, repository fixup, changed/new-file pre-commit | Passed |

Full and resumed playback each validate actual uploaded PCM, with the resumed
case comparing against native discard rather than accepting arbitrary loss.
No RoomIO tests were skipped. No codec or recorder architecture changed, so the
broader platform/Python codec matrix and wheel builds were not repeated in this
beta pass. The initial results above remain historical qualification evidence.

Live Braintrust/OpenAI/LiveKit credential environment variables were absent.
A real uploaded conversation with the intended providers and Opus/Ogg remains
outstanding. The cached local LiveKit 1.11.0 server was used for RoomIO validation.
Broader platform/load/custom-sink/exhaustive lifecycle qualification remains
outside this targeted beta pass. The extension's minimum core dependency still
needs the actual release containing the audio contract before publication; the
baseline 0.49.1 requirement is not a claim of compatibility with an older
published core lacking that contract. No publication or push occurred during
the beta pass; the subsequent local commit stack is described above.

## Changed files

The manifest below is relative to the fetched base. `A` denotes added files;
ignored build artifacts and environments are excluded.

```text
M	.github/workflows/checks.yaml
M	py/nox_session_constraints.py
M	py/noxfile.py
M	py/pyproject.toml
M	py/src/braintrust/integrations/auto_test_scripts/test_auto_livekit_agents.py
M	py/src/braintrust/integrations/livekit_agents/__init__.py
D	py/src/braintrust/integrations/livekit_agents/cassettes/1.8.8/test_livekit_agents_agent_speaking_e2e.yaml
D	py/src/braintrust/integrations/livekit_agents/cassettes/1.8.8/test_livekit_agents_e2e_parenting_under_custom_span.yaml
D	py/src/braintrust/integrations/livekit_agents/cassettes/1.8.8/test_livekit_agents_function_tool_e2e.yaml
D	py/src/braintrust/integrations/livekit_agents/cassettes/1.8.8/test_openai_tts_stream_closes_span_when_closed_early.yaml
M	py/src/braintrust/integrations/livekit_agents/cassettes/latest/test_livekit_agents_e2e_parenting_under_custom_span.yaml
D	py/src/braintrust/integrations/livekit_agents/cassettes/latest/test_openai_tts_stream_closes_span_when_closed_early.yaml
M	py/src/braintrust/integrations/livekit_agents/integration.py
M	py/src/braintrust/integrations/livekit_agents/patchers.py
M	py/src/braintrust/integrations/livekit_agents/test_livekit_agents.py
M	py/src/braintrust/integrations/livekit_agents/tracing.py
M	py/src/braintrust/logger.py
M	py/src/braintrust/test_logger.py
M	py/uv.lock
A	.github/workflows/audio-py-test.yaml
A	.github/workflows/publish-audio-py.yaml
A	docs/livekit-cascade-implementation.md
A	docs/livekit-cascade.md
A	examples/livekit/README.md
A	examples/livekit/cascade.py
A	examples/livekit/pyproject.toml
A	integrations/audio-py/README.md
A	integrations/audio-py/pyproject.toml
A	integrations/audio-py/src/braintrust_audio/__init__.py
A	integrations/audio-py/src/braintrust_audio/py.typed
A	integrations/audio-py/tests/test_codec.py
A	py/src/braintrust/_audio/__init__.py
A	py/src/braintrust/_audio/middleware.py
A	py/src/braintrust/_audio/recorder.py
A	py/src/braintrust/_audio/session.py
A	py/src/braintrust/_audio/test_audio.py
A	py/src/braintrust/_audio/test_session.py
A	py/src/braintrust/_audio/types.py
A	py/src/braintrust/_audio/worker.py
A	py/src/braintrust/_instrumentation.py
A	py/src/braintrust/audio.py
A	py/src/braintrust/integrations/auto_test_scripts/test_auto_livekit_agents_before_import.py
A	py/src/braintrust/integrations/livekit_agents/compat.py
A	py/src/braintrust/integrations/livekit_agents/conftest.py
A	py/src/braintrust/integrations/livekit_agents/playback.py
A	py/src/braintrust/integrations/livekit_agents/privacy.py
A	py/src/braintrust/integrations/livekit_agents/test_cascade.py
A	py/src/braintrust/integrations/livekit_agents/test_lifecycle.py
A	py/src/braintrust/integrations/livekit_agents/test_playback.py
A	py/src/braintrust/integrations/livekit_agents/test_privacy.py
```
