import asyncio
import json

import pytest
from braintrust import Eval, EvalAsync, logger
from braintrust.api._transport import HTTPConnection
from braintrust.voice import _call, connect, voice_task
from braintrust.voice.test_session import simulator, terminal
from braintrust.voice.test_session import tracing as tracing


web = pytest.importorskip("aiohttp.web")
aiohttp = pytest.importorskip("aiohttp")


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
async def test_eval_local_and_remote_calls_template_from_dataset_input(tracing, remote):
    # Eval must reuse a saved user and send only input, including across a real HTTP receiver.
    requests = []

    async def handler(socket):
        async for message in socket:
            if isinstance(message, str) and json.loads(message)["type"] == "stop":
                await socket.send(json.dumps(terminal()))
                return

    async def run_agent(call):
        async with connect(call) as session:
            with logger.start_span(name="Transfer tool"):
                agent_output = {"transferred_to": "billing"}
        return session.result(agent_output=agent_output)

    with logger.start_span(name="Eval parent") as parent:
        async with simulator(handler, parent.export()) as handle:

            async def version(_request):
                return web.json_response({"ff_version": 149})

            async def create(request):
                assert request.match_info["project_id"] == ("project" if remote else "voice-test")
                assert request.match_info["id"] == "saved-caller"
                body = await request.json()
                requests.append(body)
                handle["parent"] = body["parent"]
                return web.json_response(
                    {k: v for k, v in handle.items() if k not in ("parent", "timeout_seconds")}, status=201
                )

            async def receiver(request):
                return web.json_response(await run_agent(await request.json()))

            app = web.Application()
            app.router.add_get("/version", version)
            app.router.add_post("/v1/project/{project_id}/simulated_user/{id}/session", create)
            app.router.add_post("/run-eval-call", receiver)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            address = runner.addresses[0]
            base = f"http://127.0.0.1:{address[1]}"
            state = logger._internal_get_global_state()
            state.logged_in = True
            state._api_conn = HTTPConnection(base)
            try:
                async with aiohttp.ClientSession() as client:

                    async def connector(call):
                        if not remote:
                            return await run_agent(call)
                        async with client.post(base + "/run-eval-call", json=call) as response:
                            response.raise_for_status()
                            return await response.json()

                    def transferred(output, expected):
                        return float(output["agent_output"]["transferred_to"] == expected)

                    kwargs = dict(
                        name="Voice evals",
                        parent=parent.export(),
                        data=[{"input": {"request": "Ask for billing"}, "expected": "billing"}],
                        task=voice_task(
                            agent=connector, simulated_user_id="saved-caller", project_id="project" if remote else None
                        ),
                        scores=[transferred],
                        state=state,
                        enable_cache=False,
                    )
                    if remote:
                        result = await EvalAsync(**kwargs)
                    else:
                        result = await asyncio.to_thread(Eval, **kwargs)
                    assert result.results[0].error is None
                    assert result.results[0].scores == {"transferred": 1.0}
            finally:
                await runner.cleanup()
    assert len(requests) == 1
    assert requests[0]["scenario"] == {"request": "Ask for billing"}
    assert set(requests[0]) == {"scenario", "initiator", "parent", "timeout_seconds"}
    assert "expected" not in json.dumps(requests[0])


@pytest.mark.asyncio
async def test_wrong_connector_result_never_logs_session_credentials(tracing, monkeypatch):
    # Accidentally returning the call context must fail without putting its token in an eval error.
    token = "private-session-token"
    call = {"session_id": "session", "span_id": "simulator", "session_token": token}
    monkeypatch.setattr(_call, "_create_call", lambda *_args: call)

    async def connector(context):
        return context

    with pytest.raises(ValueError, match="completed session.result") as error:
        await _call.run_voice_call(agent=connector, simulated_user_id="saved-caller", input={})
    assert token not in str(error.value)
    assert token not in json.dumps(tracing.pop())
