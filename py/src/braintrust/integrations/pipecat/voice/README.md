# Pipecat voice instrumentation

The existing public setup installs one observer and discovers a supported voice pipeline. Shared recording, sample ranges, worker admission, and byte budgets live in [`braintrust.audio`](../../../audio/README.md).

```python
from braintrust import init_logger
from braintrust.integrations.pipecat import setup_pipecat

init_logger(project="voice")
setup_pipecat(capture_audio_attachments=True, audio_format="ogg")
# Construct and run the existing PipelineWorker normally.
# Its observer cleanup finalizes recordings before flushing.
```

Install `braintrust[audio]` for Ogg/Opus or WAV encoding. Audio defaults off. Existing user/agent capture arguments and environment variables still apply independently. With audio disabled, no PCM retention, sample-tracking hooks, encoder jobs, or attachments are created. Transcripts and native metadata remain observable.

| Contract | Behavior |
|---|---|
| Native voice path | Pipecat 1.12.0, one input/output transport, universal user/assistant aggregators, segmented STT or OpenAI Realtime. Other pipeline shapes/versions retain existing frame tracing. |
| Public interfaces | `setup_pipecat`, automatic integration setup, `wrap_pipeline_worker`, and an explicitly supplied `BraintrustPipecatObserver` share the same injection path. `trace_turns=False` retains the existing observer behavior. |
| Trace | `user_turn`, `assistant_turn`, `stt`, `llm_response`, `tts`, and actual tool names retain native boundaries and metadata. Realtime audio uses `pipecat.audio_output`; tool-only model responses can be pipeline children. `turn.id` / `turn.reply_to` and `continuation.tool_call_ids` preserve observed associations. |
| Audio | Combined stereo call; segmented-STT caller clips on user turns; generated clips on TTS spans. Selections reference successfully captured samples. |
| TTFB | Native `contrib.pipecat.ttfb` measurements follow the emitting processor and a unique active operation. STT measurements follow `run_stt` into its eventual span. Ambiguous/unmatched metrics remain on the pipeline; arrays are bounded to 32 with omission counts. |
| Turn detection | Native analyzer predictions in `contrib.pipecat.turn_metrics` on the user turn; preserve processor, metric class, confidence, completion and reported milliseconds. Retain 32 predictions and count omissions. Unassociated observations remain on the pipeline. |
| Measurements | `contrib.pipecat.measurements` retains native metric types and payloads on the emitting processor's unique active operation. Unattributable measurements remain on the pipeline. Each list retains at most 32 entries with `braintrust.measurements.omitted` counting excess observations. Metrics are not duplicated as frame events. |
| Events and configuration | Initial service/configuration observations are stored once in `contrib.pipecat.configuration` (up to 32 source/type pairs). Changes, speech boundaries, interruptions, cancellation, and errors retain compact timestamped events; routine start/end/connection notifications and startup zero metrics are excluded. Event payloads omit transport bookkeeping and empty fields. The pipeline-wide event budget remains 256 entries / 64 KiB. |
| Input alignment | Segmented STT input provenance or successful OpenAI input writes plus server-VAD item offsets. Clears/reconnects invalidate unsupported mappings. |
| Output alignment | Equal-rate mono PCM without a mixer. Other paths retain recordings without asserting unsupported selections. |
| Memory | Configurable 32 MiB retained call PCM by default; separate 8 MiB caller/generated source limits; 64 MiB shared source budget; 32 MiB encoded-attachment budget. These are retained-data bounds, not total RSS bounds. |
| Recording limits | Rotate call segments every 60 seconds or at half the call buffer limit; stop capture at 30 minutes by default. Configure these with `RecordingOptions`. Completed segments and valid prefixes survive limits. |
| Admission | One encoder running, one queued; bounded packet/clip metadata. Overload produces an omission reason. |
| Cleanup | Seal capture, restore installed hooks, end operation spans, encode off-loop, update attachments, then flush. Repeated finalization shares one task; cancelling a waiter does not cancel its work. |
| Privacy | Standard span customizers apply to native metadata, messages, tool arguments/results, and attachment updates. Audio bytes are excluded from event metadata. |

Run `nox -s test_audio 'test_pipecat(latest)' 'test_pipecat(1.3.0)'`. Recorded OpenAI response objects cover tool requests and continuations; transport/queue tests cover interruption, padding, failed writes, and recording opt-out. The `voice` modules are integration internals, not another customer setup API.
