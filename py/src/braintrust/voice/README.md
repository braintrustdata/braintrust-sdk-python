# Voice evals

`voice_task` runs a fresh voice call for each eval row. The SDK manages the simulator connection, audio pacing, interruptions, tracing, recordings, and cleanup. It supports a Pipecat agent in the eval process or another service, and a built-in LiveKit connector.

Simulated users are saved, long-lived objects reused across calls and evals. Set up your simulated user ahead of time through the existing UI or API, then pass its ID. These SDK helpers only start calls with existing users; they do not create or update simulated users.

Saved-user instructions are templated from each dataset row's `input`, just as in existing conversation evals. `voice_task` passes that input to the same backend template renderer automatically; no additional configuration is needed. The existing low-level session API calls this payload `scenario`.

Install `braintrust[voice]` and your agent framework. Audio conversion uses PyAV, included in the extra on Python 3.12 and later. Configure the normal Braintrust credentials and deployment URL. The helpers require data plane feature version 149 and an enabled voice session service.

## Pipecat

Adapt your existing agent entrypoint to accept a native Pipecat transport. Construct and run the pipeline inside `voice.connect`, so the existing Braintrust instrumentation inherits the eval's parent.

```python
from braintrust import voice


async def run_eval_call(call):
    async with voice.connect(call) as session:
        agent_output = await run_agent(
            transport=session.pipecat_transport(),
        )
    return session.result(agent_output=agent_output)
```

`run_agent` is your application's existing pipeline runner. Place `transport.input()` and `transport.output()` at the pipeline's input and output, enable your normal Braintrust Pipecat instrumentation, and return your application's observed result. Neither audio frames nor playback acknowledgements belong in the eval script.

Use that handler directly for a local agent. This example reads your saved simulated user's ID from `SIMULATED_USER_ID`. Its instructions reference `{{request}}`, supplied by each dataset row's `input.request`:

```python
import os

from braintrust import Eval
from braintrust.voice import voice_task


def transferred_correctly(output, expected):
    return float(output["agent_output"]["transferred_to"] == expected)


Eval(
    "Voice evals",
    data=[
        {
            "input": {"request": "Ask to speak with billing"},
            "expected": "billing",
        }
    ],
    task=voice_task(
        agent=run_eval_call,
        simulated_user_id=os.environ["SIMULATED_USER_ID"],
    ),
    scores=[transferred_correctly],
    max_concurrency=1,
)
```

For a remote agent, the connector forwards the call context to your authenticated service and awaits the handler's result:

```python
async def connect_agent(call):
    return await agent_service.run_eval_call(call)
```

The context is a plain JSON object containing single-use connection credentials and an exported tracing parent. Forward it securely and do not log it. The receiving service runs the same `run_eval_call` handler with its own Braintrust credentials. The connector must await completion and propagate errors, rather than returning a job ID or an initial acknowledgement.

## LiveKit

On the eval side, replace the callback with a built-in connector:

```python
import os

from braintrust.voice import LiveKitAgent

agent = LiveKitAgent(
    server_url=os.environ["LIVEKIT_URL"],
    agent_name="support-agent",
)
```

Pass `agent` to `voice_task`. The connector reads `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`, joins a fresh room, and dispatches the named agent.

In the existing LiveKit entrypoint, enable the normal Braintrust instrumentation and wrap the call session after joining the room:

```python
async def handle_call(ctx):
    await ctx.connect()
    session = create_agent_session()
    async with voice.livekit_agent(session, ctx.room) as call:
        await run_existing_call(session, ctx.room)
        call.agent_output = get_call_result()
```

`create_agent_session`, `run_existing_call`, and `get_call_result` represent your existing application logic. The helper reads the caller's parent, keeps it active during the call, closes the session, flushes the trace, and reports the actual result or failure back to the eval. Calls without a voice-eval participant retain their existing tracing behavior. Closing the LiveKit room belongs outside this context so the completion message can be delivered.

## Results and custom tasks

`session.result()` returns `agent_output`, `caller_transcript`, `recording`, `reason`, `session_id`, and `span_id`. `agent_output` is `None` unless the application supplies a result. The transcript contains only simulated caller speech. The recording contains both participants, with attachment references and timeline metadata in the existing segmented audio schema.

Both helpers require `simulated_user_id`. Each call snapshots the saved user's configuration and renders its instructions from the row's `input` using the existing conversation template renderer.

Use `await run_voice_call(agent=..., simulated_user_id=..., input=input)` inside a custom task. Both helpers infer the project and tracing parent from the active SDK context. Outside an eval or project logger, supply `project_id`. Only `input` is used to template the saved user's instructions; expected output stays with the eval's scorers.

Timeouts, missing terminal events, agent failures, recording failures, and unexpected disconnects fail the call. Established calls are never reconnected or replayed automatically.
