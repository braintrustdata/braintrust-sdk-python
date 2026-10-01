"""Local capture boundary tests, without simulating model responses."""

import unittest
from types import SimpleNamespace

from pipecat.frames.frames import InputAudioRawFrame  # pylint: disable=import-error

from .instrumentation import NativeObserver
from .realtime import RealtimeCapture
from .test_instrumentation import Span


class RealtimeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        # Socket/queue stand-ins exercise only local send/recording behavior.
        # Native response shape is validated by the separate real-provider calls.
        async def noop(*args, **kwargs):
            pass

        service = SimpleNamespace(
            _websocket=None,
            push_frame=noop,
            _connect=noop,
            _send_tool_result=noop,
            _handle_evt_conversation_item_added=noop,
            _handle_evt_audio_delta=noop,
            _handle_evt_audio_transcript_delta=noop,
            _handle_evt_response_done=noop,
            _handle_evt_speech_started=noop,
            _handle_evt_speech_stopped=noop,
        )

        async def send_event(evt):
            if service._websocket:
                await service._websocket.send("serialized payload")

        async def send_audio(frame):
            await service.send_client_event(SimpleNamespace(type="input_audio_buffer.append"))

        service.send_client_event, service._send_user_audio = send_event, send_audio
        aggregator = SimpleNamespace(
            _handle_transcription=noop, _push_aggregation=noop, reset=noop, add_event_handler=lambda *args: None
        )
        return service, aggregator

    async def test_only_successful_socket_writes_advance_input_clock(self):
        observer = NativeObserver(Span(), retain_audio=True)
        service, aggregator = self.service()
        capture = RealtimeCapture(observer, service, aggregator)
        frame = InputAudioRawFrame(b"\0\0" * 480, 24000, 1)
        await service._send_user_audio(frame)
        self.assertEqual(capture.sent_samples, 0)

        async def send(message):
            pass

        service._websocket = SimpleNamespace(send=send)
        await service._connect()
        await service._send_user_audio(frame)
        self.assertEqual(capture.sent_samples, 480)
        self.assertEqual(observer.call_recording.bytes, 960)
        await service.send_client_event(SimpleNamespace(type="input_audio_buffer.clear"))
        self.assertTrue(capture.invalid)

    async def test_disabled_does_not_wrap_connect_or_audio_send(self):
        observer = NativeObserver(Span(), retain_audio=False)
        service, aggregator = self.service()
        connect, send = service._connect, service._send_user_audio
        capture = RealtimeCapture(observer, service, aggregator)
        self.assertIs(service._connect, connect)
        self.assertIs(service._send_user_audio, send)
        await service._send_user_audio(InputAudioRawFrame(b"\0\0" * 480, 24000, 1))
        self.assertEqual(capture.sent_ranges, [])
        self.assertEqual(observer.call_recording.bytes, 0)
