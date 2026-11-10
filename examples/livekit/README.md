# LiveKit voice tracing and recording

The integration supports LiveKit 1.8.9–1.8.10; this example pins 1.8.10. Set the usual LiveKit and OpenAI
credentials, plus `BRAINTRUST_API_KEY`. Use Python 3.12–3.14 and install with
`uv sync` from this directory.

```sh
export BRAINTRUST_CAPTURE_USER_AUDIO_ATTACHMENTS=true
export BRAINTRUST_CAPTURE_AGENT_AUDIO_ATTACHMENTS=true
uv run voice_agent.py dev
```

Set `LIVEKIT_MODE` to choose the pipeline:

```sh
LIVEKIT_MODE=realtime uv run voice_agent.py dev
```

| `LIVEKIT_MODE`      | Pipeline                          | Models used by this example                                   |
| ------------------- | --------------------------------- | ------------------------------------------------------------- |
| `cascade` (default) | STT → text LLM → TTS              | `gpt-4o-mini-transcribe` → `gpt-4.1-mini` → `gpt-4o-mini-tts` |
| `half-cascade`      | Audio → Realtime text model → TTS | `gpt-realtime` with text output → `tts-1`                     |
| `realtime`          | Native speech-to-speech           | `gpt-realtime`                                                |

With `braintrust[audio]`, enabled recordings default to Opus in Ogg. Installing
the extra or setting the generic `BRAINTRUST_CAPTURE_ATTACHMENTS` flag does not
enable audio capture.

The example enables both LiveKit and OpenAI instrumentation:

```python
from braintrust.integrations.livekit_agents import setup_livekit_agents
from braintrust.integrations.openai import setup_openai

setup_livekit_agents()
setup_openai()
```

LiveKit records turns and speech; OpenAI records model requests and usage under
those operations. `auto_instrument()` also enables both integrations.

```text
session
├── user_turn
│   └── stt
│       └── Transcription
└── agent_turn
    ├── llm_response (task)
    │   └── Chat Completion (llm, token usage)
    ├── tool
    ├── tts
    └── agent_speech
```

For adapters without known provider instrumentation, LiveKit emits an LLM child
from the model data exposed by the framework. It does not infer provider support
from whether a span appears.
