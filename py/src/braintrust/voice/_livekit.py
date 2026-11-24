import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, TypeVar

from .. import logger
from ..span_types import SpanTypeAttribute
from ._session import connect
from ._types import JSONValue, LiveKitAgent, VoiceCall, VoiceCallResult


if TYPE_CHECKING:
    from livekit import rtc
    from livekit.agents import AgentSession, ErrorEvent


_RESULT_TOPIC = "braintrust.voice_eval.result"
_PARENT_ATTRIBUTE = "x-bt-parent"
UserData = TypeVar("UserData")


def credentials(agent: LiveKitAgent) -> tuple[str, str]:
    key = agent.api_key or os.environ.get("LIVEKIT_API_KEY")
    secret = agent.api_secret or os.environ.get("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise ValueError("LiveKitAgent requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET")
    return key, secret


async def run_livekit(call: VoiceCall, agent: LiveKitAgent) -> VoiceCallResult:
    from livekit import api, rtc

    key, secret = credentials(agent)
    token = (
        api.AccessToken(key, secret)
        .with_identity(f"braintrust-eval-{call['session_id']}")
        .with_ttl(timedelta(minutes=15))
        .with_attributes({_PARENT_ATTRIBUTE: call["parent"], "braintrust.voice_eval": "1"})
        .with_grants(api.VideoGrants(room_join=True, room=call["session_id"], can_publish=True, can_subscribe=True))
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=agent.agent_name)], empty_timeout=60, departure_timeout=10
            )
        )
    )
    readers: set[asyncio.Task[None]] = set()
    completed: asyncio.Future[dict[str, JSONValue]] = asyncio.get_running_loop().create_future()
    async with connect(call) as session:
        room = rtc.Room()
        source = rtc.AudioSource(24000, 1, queue_size_ms=20)

        async def receive_track(track: "rtc.Track"):
            audio = rtc.AudioStream(track, sample_rate=24000, num_channels=1, capacity=100)
            try:
                async for event in audio:
                    await session._write_audio(bytes(event.frame.data), 24000, 1)
            finally:
                await audio.aclose()

        def subscribed(
            track: "rtc.RemoteTrack", _publication: "rtc.RemoteTrackPublication", participant: "rtc.RemoteParticipant"
        ):
            if (
                track.kind != rtc.TrackKind.KIND_AUDIO
                or participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
            ):
                return
            task = asyncio.create_task(receive_track(track))
            readers.add(task)

            def finished(task: asyncio.Task[None]):
                readers.discard(task)
                error = None if task.cancelled() else task.exception()
                if error is not None and not completed.done():
                    completed.set_exception(error)

            task.add_done_callback(finished)

        def data_received(packet: "rtc.DataPacket"):
            if packet.topic != _RESULT_TOPIC or packet.participant is None or completed.done():
                return
            if packet.participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                return
            try:
                result = json.loads(packet.data)
                if not isinstance(result, dict) or result.get("session_id") != call["session_id"]:
                    raise ValueError("Invalid LiveKit voice eval result")
                completed.set_result(result)
            except Exception as error:
                completed.set_exception(error)

        def disconnected(participant: "rtc.RemoteParticipant"):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT and not completed.done():
                completed.set_exception(
                    RuntimeError("LiveKit agent left without reporting completion; use voice.livekit_agent")
                )

        room.on("track_subscribed", subscribed)
        room.on("data_received", data_received)
        room.on("participant_disconnected", disconnected)
        try:
            await room.connect(agent.server_url, token.to_jwt())
            track = rtc.LocalAudioTrack.create_audio_track("simulated-user", source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )

            async def consume(audio: bytes):
                await source.capture_frame(rtc.AudioFrame(audio, 24000, 1, len(audio) // 2))
                await source.wait_for_playout()

            async def clear():
                source.clear_queue()

            async def ended():
                if session._error and not completed.done():
                    completed.set_exception(session._error)
                elif not completed.done():
                    for participant in room.remote_participants.values():
                        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                            await room.local_participant.perform_rpc(
                                destination_identity=participant.identity,
                                method="braintrust.voice_eval.stop",
                                payload="",
                                response_timeout=10,
                            )

            session._set_sink(consume, clear=clear, ended=ended)
            result = await asyncio.wait_for(completed, call["timeout_seconds"] + 30)
            if result.get("error"):
                raise RuntimeError(str(result["error"]))
            agent_output = result.get("agent_output")
        finally:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            try:
                await asyncio.wait_for(room.disconnect(), 5)
            finally:
                await source.aclose()
    return session.result(agent_output=agent_output)


class LiveKitCall:
    """Application result for a LiveKit eval call. Set agent_output from observed application state."""

    def __init__(self):
        self.agent_output: JSONValue = None


@asynccontextmanager
async def livekit_agent(session: "AgentSession[UserData]", room: "rtc.Room") -> AsyncGenerator[LiveKitCall, None]:
    """Scope an instrumented LiveKit agent to its caller and report runtime failures to its eval.

    Enter after joining the room, before session.start(). Regular calls without
    the voice-eval participant marker keep their existing tracing behavior.
    """
    participant_ready = asyncio.Event()

    def joined(_participant: "rtc.RemoteParticipant"):
        participant_ready.set()

    room.on("participant_connected", joined)
    try:
        while not room.remote_participants:
            await asyncio.wait_for(participant_ready.wait(), 60)
            participant_ready.clear()
        callers = [p for p in room.remote_participants.values() if p.attributes.get("braintrust.voice_eval") == "1"]
    finally:
        room.off("participant_connected", joined)
    call = LiveKitCall()
    if not callers:
        yield call
        return
    if len(callers) != 1:
        raise ValueError("A voice eval room must have exactly one simulated caller")
    caller = callers[0]
    error: Exception | None = None

    def failed(event: "ErrorEvent"):
        nonlocal error
        error = error or RuntimeError(str(event.error))

    session.on("error", failed)

    async def stop(invocation: "rtc.RpcInvocationData"):
        if invocation.caller_identity != caller.identity:
            raise RuntimeError("Only the simulated caller can end this eval")
        await session.aclose()
        return ""

    room.local_participant.register_rpc_method("braintrust.voice_eval.stop")(stop)
    span = logger.start_span(
        name="Voice agent", type=SpanTypeAttribute.TASK, parent=caller.attributes[_PARENT_ATTRIBUTE]
    )
    try:
        with span:
            try:
                yield call
                if error:
                    raise error
            except BaseException as cause:
                error = error or RuntimeError(str(cause))
                raise
            finally:
                try:
                    await session.aclose()
                except Exception as cause:
                    error = error or cause
                    raise
    finally:
        session.off("error", failed)
        room.local_participant.unregister_rpc_method("braintrust.voice_eval.stop")
        await asyncio.to_thread(span.flush)
        await room.local_participant.publish_data(
            json.dumps(
                {
                    "session_id": room.name,
                    "agent_output": call.agent_output,
                    **({"error": str(error)} if error else {}),
                }
            ),
            reliable=True,
            destination_identities=[caller.identity],
            topic=_RESULT_TOPIC,
        )
