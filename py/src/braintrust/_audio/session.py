import asyncio
import logging
import time
from collections.abc import Callable

from braintrust._audio.config import AudioRecordingOptions
from braintrust.env import BraintrustEnv
from braintrust.logger import Attachment

from .middleware import encoding_stack
from .recorder import AudioRecorder
from .types import AudioResult, RecordingResult, SegmentMeta, thaw
from .worker import run_native


class AudioSessionRecorder:
    """Schema registries and recorder-local direct attachment export.

    ``allowed`` must include capture opt-ins and effective framework privacy. It
    is checked again before upload. Trace customizers cannot veto audio uploads.
    """

    def __init__(
        self,
        span,
        *,
        sources: tuple[str, ...] = (),
        options: AudioRecordingOptions | None = None,
        allowed: Callable[[], bool] = lambda: True,
    ):
        self.span = span
        self.wall_ns, self.monotonic_ns = time.time_ns(), time.monotonic_ns()
        self.allowed = allowed
        self.options = options or AudioRecordingOptions()
        self._participants: dict[object, str] = {}
        self._speakers: dict[object, str] = {}
        self._terminal: set[int] = set()
        self._inflight: dict[int, SegmentMeta] = {}
        self._closed = False
        self._sources = {source: {} for source in sources}
        self.recorder = (
            AudioRecorder(
                sources=sources,
                options=self.options,
                middleware=encoding_stack(self.options.encoding, channels=len(sources)),
                on_segment=self._export,
                on_omitted=self.omitted,
            )
            if sources
            else None
        )
        self.span.log(metadata=self.session_metadata())

    def session_metadata(self):
        return {
            "audio.schema.version": "3.0.0",
            "audio.timeline": {"origin_unix_ms": self.wall_ns // 1_000_000},
            "audio.sources": dict(self._sources),
        }

    def source(self, source_id, member=None):
        if source_id not in self._sources:
            self._sources[source_id] = dict(member or {})
            self.span.log(metadata={"audio.sources": {source_id: self._sources[source_id]}})
        return source_id

    def elapsed_ns(self):
        return time.monotonic_ns() - self.monotonic_ns

    def from_wall(self, seconds: float):
        return max(0, round(seconds * 1_000_000_000) - self.wall_ns)

    def participant(self, native_key, member):
        return self._identity(self._participants, native_key, member, "audio.participants", "participant")

    def speaker(self, source_id, native_key, member):
        return self._identity(self._speakers, (source_id, native_key), member, "audio.speakers", "speaker")

    def _identity(self, registry, key, member, field, prefix):
        if not self.allowed():
            return None
        if key not in registry:
            identifier = f"{prefix}-{len(registry)}"
            registry[key] = identifier
            self.span.log(metadata={field: {identifier: member}})
        return registry[key]

    def record(self, source_id, pcm, **format_and_time):
        if self.recorder is None or not self.allowed():
            if self.recorder is not None:
                self.recorder.discard("privacy_restricted")
            return False
        return self.recorder.record(source_id, pcm, **format_and_time)

    def advance(self, through_ns):
        if self.recorder is not None:
            self.recorder.advance(through_ns)

    def unavailable(self, start_ns, end_ns, reason):
        if self.recorder is not None:
            self.recorder.unavailable(start_ns, end_ns, reason)

    def omitted(self, meta: SegmentMeta, reason: str):
        if meta.index in self._terminal:
            return
        self._terminal.add(meta.index)
        self._inflight.pop(meta.index, None)
        self.span.log(
            metadata={
                "audio.segments": {
                    f"part-{meta.index}": {
                        "state": "omitted",
                        **meta.placement(),
                        "reason": reason,
                    }
                }
            }
        )

    async def _export(self, result: AudioResult):
        meta = result.meta
        self._inflight[meta.index] = meta
        identifier = f"part-{meta.index}"
        if self._closed or not self.allowed():
            self.omitted(meta, "privacy_restricted" if not self._closed else "shutdown_timeout")
            return
        self.span.log(metadata={"audio.segments": {identifier: {"state": "pending", **meta.placement()}}})
        extension = {"audio/ogg": "ogg", "audio/wav": "wav", "audio/pcm": "pcm"}[meta.mime_type]
        attachment = Attachment(data=result.data, filename=f"{identifier}.{extension}", content_type=meta.mime_type)
        attachment._upload_timeout_seconds = BraintrustEnv.HTTP_TIMEOUT.get(60.0)

        def upload():
            # Recheck after waiting for a native worker slot.
            if self._closed or not self.allowed():
                return None
            return attachment.upload()

        try:
            status = await run_native(upload)
        except Exception:
            self.omitted(meta, "upload_failed")
            raise
        if meta.index in self._terminal:
            return
        if status is None or not self.allowed():
            self.omitted(meta, "privacy_restricted")
        elif status.get("upload_status") != "done":
            self.omitted(meta, "upload_failed")
            raise RuntimeError("Audio upload failed")
        else:
            self._terminal.add(meta.index)
            self._inflight.pop(meta.index, None)
            self.span.log(
                input={"audio": {identifier: attachment.reference}},
                metadata={
                    "audio.segments": {
                        identifier: {
                            "state": "ready",
                            **meta.placement(),
                            "attachment": {"ref": f"/input/audio/{identifier}"},
                            "mime_type": meta.mime_type,
                            "encoding": meta.encoding,
                            "sample_rate_hz": meta.sample_rate_hz,
                            "channels": {
                                str(i): {"sources": list(channel.sources)} for i, channel in enumerate(meta.channels)
                            },
                            "attributes": thaw(meta.attributes),
                        }
                    }
                },
            )

    async def finish(self) -> RecordingResult:
        result = RecordingResult(0)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.options.shutdown_timeout_seconds
        try:
            if self.recorder is not None:
                result = await asyncio.wait_for(self.recorder.finish(), self.options.shutdown_timeout_seconds)
        except (asyncio.TimeoutError, asyncio.CancelledError) as error:
            self._closed = True
            for meta in tuple(self._inflight.values()):
                self.omitted(meta, "shutdown_timeout")
            if self.recorder is not None:
                await self.recorder.abort("shutdown_timeout")
            result = RecordingResult(len(self._terminal), "shutdown_timeout")
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            self._closed = True
            if result.reason:
                self.span.log(metadata={"contrib.braintrust.recording_stop_reason": result.reason})
            # Terminal updates are queued before flushing. No exporter changes.
            flushing = loop.run_in_executor(None, self.span.flush)
            flushing.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
            try:
                await asyncio.wait_for(asyncio.shield(flushing), max(0, deadline - loop.time()))
            except asyncio.TimeoutError:
                result = RecordingResult(result.segments, result.reason or "logger_flush_timeout")
            except Exception:
                logging.getLogger("braintrust").warning("Audio terminal metadata flush failed")
        return result
