# Voice cassette maintenance

Run from `py/`. Replay needs no credentials:

```sh
mise exec -- uv run nox -s 'test_pipecat(latest)' -- --vcr-record=none
mise exec -- uv run nox -s 'test_pipecat(1.3.0)' -- --vcr-record=none
```

To re-record the voice conversations, set `OPENAI_API_KEY` in your shell, then run:

```sh
mise exec -- uv run nox -s 'test_pipecat(latest)' -- --vcr-record=all -k voice_conversation
```

- Re-recording makes short, billable OpenAI calls. No other API keys are needed. Existing fixtures are replaced only after the conversation passes its assertions.
- HTTP uses VCR; Realtime uses `_test_websocket.py` to record/replay WebSocket messages. `_test_audio_cassettes.py` stores audio separately and restores exact bytes for replay.
- Commit YAML/JSON manifests **and** companion `.audio/` directories under `pipecat/cassettes/latest/` together. Re-recording regenerates these WAVs; do not edit or compress them manually. WebSocket offsets refer to PCM bytes, excluding the WAV header.
- `order.wav` is the cascade caller input: 16 kHz mono PCM16, “Where is my order number one zero four two?”, generated with macOS's Samantha voice. Realtime uses `order-24khz.wav`, a pre-resampled copy of that request. Keeping its input bytes fixed avoids platform-dependent resampler rounding during strict WebSocket replay. Changing either input requires re-recording its conversation.
- After re-recording, review the fixture diff and run replay with `--vcr-record=none` before committing. Missing or truncated audio files fail replay.
