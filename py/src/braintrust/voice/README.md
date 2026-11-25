# WebSocket voice evals

Install `braintrust[voice]` and configure normal Braintrust credentials. The helpers require data plane feature version 149 and an enabled voice session service.

Create your simulated user ahead of time in the existing UI or API and pass its ID. `voice_task` runs a fresh call for each dataset row, templating the saved user's instructions from the row's `input` through the existing backend renderer. The low-level API names that input `scenario`.

```python
import os

from braintrust import Eval
from braintrust.voice import voice_task


async def connect_agent(call):
    return await agent_service.run_eval_call(call)


Eval(
    "Voice evals",
    data=cases,
    task=voice_task(
        agent=connect_agent,
        simulated_user_id=os.environ["SIMULATED_USER_ID"],
    ),
    scores=scorers,
)
```

The connector forwards the JSON call context to your authenticated agent service and awaits its completed result. The context contains single-use connection credentials and an exported tracing parent. Forward it securely and do not log it. A local agent can call the same receiver directly.

## Agent receiver

Construct and run the agent inside `voice.connect` so existing SDK instrumentation inherits the eval parent. Connect your agent's audio callbacks to the session:

```python
from braintrust import voice


async def run_eval_call(call):
    async with voice.connect(call) as session:
        agent = create_agent(
            send_audio=session.send_audio,
            on_interrupt=session.clear_output,
        )
        session.set_audio_sink(
            agent.play_audio,
            clear=agent.clear_audio,
            ended=agent.stop,
        )
        output = await agent.run()
    return session.result(agent_output=output)
```

`create_agent` represents your application's audio bridge. `send_audio(audio, sample_rate, channels)` accepts the agent's signed 16-bit PCM and handles conversion and pacing. The sink receives 24 kHz mono PCM; its async playback callback must return after consumption so the SDK acknowledges the correct samples. Its async `clear` callback discards interrupted playback, and `ended` stops the agent through its normal lifecycle. Asynchronous transport errors can be reported with `await session.fail(error)`.

The SDK owns the WebSocket handshake, pacing, playback acknowledgements, interruption epochs, stop/finalization, and trace flushing. Framework transports are separate integrations.

## Results and custom tasks

`session.result()` returns `agent_output`, `caller_transcript`, `recording`, `reason`, `session_id`, and `span_id` after finalization. The transcript contains only simulated caller speech. Recordings include both participants through attachment references and timeline metadata. The application supplies its actual output and propagates agent failures.

Use `await run_voice_call(agent=..., simulated_user_id=..., input=input)` in a custom eval task. Both helpers require an active eval task; the project is inferred from its experiment, or supplied explicitly as `project_id`. Expected output and scorer configuration stay in the eval.

Calls retain the backend's ten-minute limit. Missing terminal events, agent or recording errors, cancellation, and unexpected disconnects fail the call. Established calls are never reconnected or replayed automatically.
