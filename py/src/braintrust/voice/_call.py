import asyncio
import dataclasses
import importlib
from collections.abc import Awaitable, Callable
from urllib.parse import quote

import jsonschema

from .. import logger
from ..span_identifier_v3 import SpanObjectTypeV3
from ..span_identifier_v4 import SpanComponentsV4
from ..span_types import SpanTypeAttribute
from ..util import response_raise_for_status
from ._types import (
    AgentConnector,
    Initiator,
    JSONValue,
    LiveKitAgent,
    SimulatedUserRef,
    VoiceCall,
    VoiceCallResult,
    parse_result,
)


MIN_FF_VERSION = 149


def _project_id(parent: str, project_id: str | None, state: logger.BraintrustState) -> str:
    if project_id:
        return project_id
    components = SpanComponentsV4.from_str(parent)
    if components.object_type == SpanObjectTypeV3.PROJECT_LOGS and components.object_id:
        return components.object_id
    if components.object_type == SpanObjectTypeV3.EXPERIMENT and components.object_id:
        experiment = state.current_experiment
        if experiment is not None and experiment.id == components.object_id:
            return experiment.project.id
        response = state.api_conn().get_json(f"v1/experiment/{quote(components.object_id, safe='')}")
        value = response.get("project_id")
        if isinstance(value, str):
            return value
    raise ValueError("Pass project_id or run the voice call inside an Eval or project logger")


def _create_call(
    caller: SimulatedUserRef,
    scenario: dict[str, JSONValue],
    initiator: Initiator,
    timeout: int,
    project_id: str | None,
    parent: str,
    state: logger.BraintrustState,
) -> VoiceCall:
    state.login()
    version = state.api_conn().get_json("version").get("ff_version")
    if not isinstance(version, int) or version < MIN_FF_VERSION:
        raise RuntimeError(f"Voice eval helpers require data plane feature version {MIN_FF_VERSION} or later")
    components = SpanComponentsV4.from_str(parent)
    if components.compute_object_metadata_args and components.object_type == SpanObjectTypeV3.PROJECT_LOGS:
        metadata = logger._compute_logger_metadata(state=state, **components.compute_object_metadata_args)
        parent = dataclasses.replace(
            components, object_id=metadata.project.id, compute_object_metadata_args=None
        ).to_str()
    project_id = _project_id(parent, project_id, state)
    caller_id = caller.id
    if caller_id is None:
        response = state.api_conn().get_json(
            f"v1/project/{quote(project_id, safe='')}/simulated_user",
            {"simulated_user_name": caller.name, "limit": 2},
        )
        matches = response.get("objects")
        if not isinstance(matches, list) or len(matches) != 1:
            raise ValueError(f"Expected exactly one simulated user named {caller.name!r}; use an ID to disambiguate")
        caller_id = matches[0]["id"]
    response = state.api_conn().post(
        f"v1/project/{quote(project_id, safe='')}/simulated_user/{quote(caller_id, safe='')}/session",
        json={
            "scenario": scenario,
            "initiator": initiator,
            "parent": parent,
            "timeout_seconds": timeout,
        },
        timeout=60,
    )
    response_raise_for_status(response)
    handle = response.json()
    return VoiceCall(
        connection_url=handle["connection_url"],
        session_token=handle["session_token"],
        expires_at=handle["expires_at"],
        session_id=handle["session_id"],
        span_id=handle["span_id"],
        parent=parent,
        timeout_seconds=timeout,
    )


async def run_voice_call(
    *,
    agent: AgentConnector | LiveKitAgent,
    simulated_user: SimulatedUserRef,
    scenario: dict[str, JSONValue],
    initiator: Initiator = "agent",
    timeout: int = 600,
    project_id: str | None = None,
    parent: str | None = None,
    state: logger.BraintrustState | None = None,
) -> VoiceCallResult:
    """Run a fresh call with a saved user. The connector awaits completion and returns session.result()."""
    if type(timeout) is not int or not 15 <= timeout <= 600:
        raise ValueError("timeout must be an integer between 15 and 600 seconds")
    if initiator not in ("agent", "simulated_user"):
        raise ValueError("initiator must be agent or simulated_user")
    if isinstance(agent, LiveKitAgent):
        from ._livekit import credentials

        importlib.import_module("livekit.api")
        importlib.import_module("livekit.rtc")
        credentials(agent)
    elif not callable(agent):
        raise TypeError("agent must be an async call connector or LiveKitAgent")
    if state is None:
        state = logger._internal_get_global_state()
    if parent is None and project_id and not logger.get_span_parent_object(state=state).export():
        parent = logger.init_logger(project_id=project_id, state=state, set_current=False).export()
    with logger.start_span(name="Voice call", type=SpanTypeAttribute.TASK, parent=parent, state=state) as span:
        call = await asyncio.to_thread(
            _create_call, simulated_user, scenario, initiator, timeout, project_id, span.export(), state
        )
        span.log(
            input=scenario, metadata={"voice_session_id": call["session_id"], "simulator_span_id": call["span_id"]}
        )
        if isinstance(agent, LiveKitAgent):
            from ._livekit import run_livekit

            result = await asyncio.wait_for(run_livekit(call, agent), timeout + 150)
        else:
            result = await asyncio.wait_for(agent(call), timeout + 150)
        try:
            result = parse_result(result)
        except (jsonschema.ValidationError, TypeError, ValueError):
            raise ValueError(
                "connect_agent must return the completed session.result(), not a job or call-start response"
            ) from None
        if result["session_id"] != call["session_id"]:
            raise ValueError(
                "connect_agent must return the completed session.result(), not a job or call-start response"
            )
        span.log(output=result)
        return result


def voice_task(
    *,
    agent: AgentConnector | LiveKitAgent,
    simulated_user: SimulatedUserRef,
    initiator: Initiator = "agent",
    timeout: int = 600,
    project_id: str | None = None,
) -> Callable[[dict[str, JSONValue]], Awaitable[VoiceCallResult]]:
    """Create an Eval task reusing a saved user. Only input becomes the caller's scenario."""

    async def task(input: dict[str, JSONValue]) -> VoiceCallResult:
        return await run_voice_call(
            agent=agent,
            simulated_user=simulated_user,
            scenario=input,
            initiator=initiator,
            timeout=timeout,
            project_id=project_id,
        )

    return task
