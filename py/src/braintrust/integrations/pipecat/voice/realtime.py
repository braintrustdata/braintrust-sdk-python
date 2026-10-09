"""Pinned OpenAI Realtime hooks: native event IDs and sent-sample provenance.

Server VAD + uninterrupted PCM input only. Buffer clears invalidate selections.
No STT operation is fabricated for asynchronous provider transcription events.
"""

from contextvars import ContextVar

from braintrust._audio import ms_to_samples

from ..llm_metrics import _metadata_from_processor
from .instrumentation import native_value


class RealtimeCapture:
    def __init__(self, observer, service, aggregator):
        self.observer = observer
        self.items = {}
        self.consumed = []
        self.associations = []
        self.sent_samples = 0
        self.sent_ranges: list[tuple[int, int, dict[str, int]]] = []
        self.invalid = False
        self.omitted = 0
        self.frame_token = object()
        self.active = ContextVar("realtime_response", default=None)
        self.service = service
        self.user_item = None
        self.user_turns = {}
        self.pending_tool_results = []
        original_tool_result = observer.hooks.original(service, "_send_tool_result")

        async def tool_result(tool_call_id, result):
            self.pending_tool_results.append(tool_call_id)
            return await original_tool_result(tool_call_id, result)

        observer.hooks.set(service, "_send_tool_result", tool_result)

        async def user_started(aggregator, strategy):
            if self.user_item and observer.turns.user:
                self.user_turns[self.user_item] = observer.turns.user
                if len(self.user_turns) > 256:
                    self.user_turns.pop(next(iter(self.user_turns)))
                observer.turns.user["span"].log(metadata={"openai.item_id": self.user_item})

        observer.hooks.event(aggregator, "on_user_turn_started", user_started)
        original_item = observer.hooks.original(service, "_handle_evt_conversation_item_added")

        async def item_added(evt):
            if evt.item.type == "function_call" and observer.llm is None:
                turn = self.user_turns.get(self.user_item)
                observer.context_reply_to = turn["span"].span_id if turn else None
                observer.llm_turn = None
                observer.llm_text = []
                observer.llm_tool_calls = []
                observer.llm = observer.root.start_span(
                    name="llm_response",
                    type="llm",
                    set_current=False,
                    internal={"instrumentation": "pipecat-auto"},
                    metadata={
                        **_metadata_from_processor(service),
                        "openai.item_id": evt.item.id,
                        "openai.call_id": evt.item.call_id,
                        "turn.reply_to": observer.context_reply_to,
                    },
                )
                observer.ttfb.start("llm", service, observer.llm.log)
            return await original_item(evt)

        observer.hooks.set(service, "_handle_evt_conversation_item_added", item_added)
        original_push = observer.hooks.original(service, "push_frame")

        async def push(frame, *args, **kwargs):
            context = self.active.get()
            if context and type(frame).__name__ in {
                "TTSStartedFrame",
                "TTSStoppedFrame",
                "TTSAudioRawFrame",
                "TTSTextFrame",
            }:
                # Keep provenance on the queued frame, not in a call-long ID map.
                frame._braintrust_realtime_context = (self.frame_token, context)
            return await original_push(frame, *args, **kwargs)

        observer.hooks.set(service, "push_frame", push)
        for name in ("_handle_evt_audio_delta", "_handle_evt_audio_transcript_delta", "_handle_evt_response_done"):
            original = observer.hooks.original(service, name)

            async def response(evt, original=original):
                response_id = getattr(evt, "response_id", None) or getattr(getattr(evt, "response", None), "id", None)
                token = self.active.set(response_id)
                response_span = observer.llm
                if response_span and response_id:
                    response_span.log(metadata={"openai.response.id": response_id})
                    if getattr(evt, "response", None) is not None:
                        response_span.log(metadata={"openai.response": native_value(evt.response)})
                        observer.llm_tool_calls = [
                            {
                                "id": item.call_id,
                                "type": "function",
                                "function": {"name": item.name, "arguments": item.arguments},
                            }
                            for item in evt.response.output
                            if item.type == "function_call"
                        ]
                try:
                    result = await original(evt)
                    if response_span and getattr(evt, "response", None) is not None:
                        calls = [item for item in evt.response.output if item.type == "function_call"]
                        if calls:
                            response_span.log(
                                output=[
                                    {
                                        "role": "assistant",
                                        "content": "".join(observer.llm_text) or None,
                                        "tool_calls": [
                                            {
                                                "id": item.call_id,
                                                "type": "function",
                                                "function": {"name": item.name, "arguments": item.arguments},
                                            }
                                            for item in calls
                                        ],
                                    }
                                ]
                            )
                    state = observer.tts.get(response_id)
                    if state:
                        state["span"].log(
                            metadata={
                                "openai.response.id": response_id,
                                "openai.item_id": getattr(evt, "item_id", None),
                            }
                        )
                    return result
                finally:
                    self.active.reset(token)

            observer.hooks.set(service, name, response)

        for name, field in (
            ("_handle_evt_speech_started", "audio_start_ms"),
            ("_handle_evt_speech_stopped", "audio_end_ms"),
        ):
            original = observer.hooks.original(service, name)

            async def speech(evt, original=original, field=field):
                if field == "audio_start_ms":
                    self.user_item = evt.item_id
                if evt.item_id not in self.items and len(self.items) >= 256:
                    self.items.pop(next(iter(self.items)))
                    self.omitted += 1
                self.items.setdefault(evt.item_id, {})[field] = getattr(evt, field)
                result = await original(evt)
                self.publish()
                return result

            observer.hooks.set(service, name, speech)

        original_accept = observer.hooks.original(aggregator, "_handle_transcription")
        original_commit = observer.hooks.original(aggregator, "_push_aggregation")
        original_reset = observer.hooks.original(aggregator, "reset")

        async def accept(frame):
            result = await original_accept(frame)
            if frame.text.strip() and len(self.consumed) < 256:
                self.consumed.append(native_value(frame))
            return result

        async def reset():
            self.consumed.clear()
            return await original_reset()

        async def commit(*args, **kwargs):
            consumed = self.consumed[:]
            turn = observer.turns.user or observer.turns.messages.get(("user", aggregator._user_turn_start_timestamp))
            if turn:
                observer.context_reply_to = turn["span"].span_id
            result = await original_commit(*args, **kwargs)
            if result and turn:
                ids = [f.get("result", {}).get("item_id") for f in consumed if isinstance(f.get("result"), dict)]
                turn["span"].log(
                    metadata={
                        "contrib.pipecat.transcriptions": consumed,
                        "openai.item_ids": ids,
                        "braintrust.user_capture.association": "aggregator_consumed_frames",
                    }
                )
                if len(self.associations) >= 256:
                    self.associations.pop(0)
                    self.omitted += 1
                self.associations.append((turn["span"], ids))
                self.publish()
                observer.context_reply_to = turn["span"].span_id
            return result

        observer.hooks.set(aggregator, "_handle_transcription", accept)
        observer.hooks.set(aggregator, "_push_aggregation", commit)
        observer.hooks.set(aggregator, "reset", reset)
        # Opt-out never installs the per-audio-frame send hook or stores sample ranges.
        if observer.capture_user_audio:
            original_send = observer.hooks.original(service, "_send_user_audio")
            original_event = observer.hooks.original(service, "send_client_event")
            original_connect = observer.hooks.original(service, "_connect")
            audio_frame = ContextVar("realtime_input_frame", default=None)
            self.bound_socket = None

            async def connect():
                result = await original_connect()
                socket = service._websocket
                if socket is not None and socket is not self.bound_socket:
                    if self.bound_socket is not None:
                        self.invalid = True  # Provider buffer clock may restart.
                    self.bound_socket = socket
                    original_socket_send = observer.hooks.original(socket, "send")

                    async def socket_send(message, *args, **kwargs):
                        frame = audio_frame.get()
                        try:
                            result = await original_socket_send(message, *args, **kwargs)
                        except BaseException:
                            if frame is not None:
                                self.invalid = True
                            raise
                        if frame is not None:
                            record_sent(frame)
                        return result

                    observer.hooks.set(socket, "send", socket_send)
                return result

            def record_sent(frame):
                if frame.sample_rate != 24000 or frame.num_channels != 1:
                    self.invalid = True
                try:
                    interval = observer.call_recording.capture(0, frame.audio, frame.sample_rate, frame.num_channels)
                except Exception:
                    observer.call_recording.omit("capture_error")
                    interval = None
                size = len(frame.audio) // 2
                if interval and not self.invalid:
                    merged = False
                    if self.sent_ranges:
                        start, end, previous = self.sent_ranges[-1]
                        if end == self.sent_samples and previous["end"] == interval["start"]:
                            self.sent_ranges[-1] = (
                                start,
                                self.sent_samples + size,
                                {"start": previous["start"], "end": interval["end"]},
                            )
                            merged = True
                    if not merged:
                        if len(self.sent_ranges) >= 16000:
                            self.sent_ranges.pop(0)
                            self.omitted += 1
                        self.sent_ranges.append((self.sent_samples, self.sent_samples + size, interval))
                self.sent_samples += size
                if self.associations:
                    self.publish()

            async def send_audio(frame):
                token = audio_frame.set(frame)
                try:
                    return await original_send(frame)
                finally:
                    audio_frame.reset(token)

            observer.hooks.set(service, "_connect", connect)

            async def send_event(evt):
                if evt.type == "input_audio_buffer.clear":
                    self.publish()
                    self.invalid = True
                return await original_event(evt)

            observer.hooks.set(service, "_send_user_audio", send_audio)
            observer.hooks.set(service, "send_client_event", send_event)

    def publish(self, final=False):
        pending = []
        completed = set()
        for span, ids in self.associations:
            events = [{"item_id": item, **self.items.get(item, {})} for item in ids]
            ready = all("audio_start_ms" in event and "audio_end_ms" in event for event in events)
            if ready and self.observer.capture_user_audio and not self.invalid:
                ready = all(ms_to_samples(event["audio_end_ms"]) <= self.sent_samples for event in events)
            if not ready and not final:
                pending.append((span, ids))
                continue
            span.log(metadata={"openai.input_audio_segments": events})
            if self.observer.capture_user_audio and not self.invalid:
                for event in events:
                    if "audio_start_ms" not in event or "audio_end_ms" not in event:
                        continue
                    start, end = ms_to_samples(event["audio_start_ms"]), ms_to_samples(event["audio_end_ms"])
                    ranges = []
                    for a, b, interval in self.sent_ranges:
                        left, right = max(start, a), min(end, b)
                        if right > left:
                            ranges.append([interval["start"] + left - a, interval["start"] + right - a])
                    if sum(b - a for a, b in ranges) == end - start:
                        self.observer.alignment.add(span, ranges, 0)
                    else:
                        self.omitted += 1
            completed.update(ids)
        for item in completed.difference(item for _, ids in pending for item in ids):
            self.items.pop(item, None)
        self.associations = pending

    def finish(self):
        self.publish(final=True)
        self.observer.root.log(
            metadata={
                "braintrust.realtime.alignment_invalidated": self.invalid,
                "braintrust.realtime.associations_omitted": self.omitted,
            }
        )
        self.sent_ranges.clear()
        self.items.clear()
        self.user_turns.clear()
