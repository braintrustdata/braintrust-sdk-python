"""Execute a sealed segment without deciding recording policy or publishing spans."""

from .attachments import UploadFailed, prepare_recording, upload_recording
from .segment import AudioSegment
from .worker import encode_in_worker


class SegmentExporter:
    def __init__(self, audio_format: str = "ogg"):
        self.audio_format = audio_format

    async def export(self, segment: AudioSegment) -> None:
        def encode():
            try:
                return prepare_recording(segment.segment_id, segment.source.encode, self.audio_format)
            finally:
                # The worker owns PCM until native encoding has finished.
                segment.clear()

        try:
            encoded = await encode_in_worker(encode)
            segment.encoded = encoded
            await upload_recording(encoded)
            segment.ready(encoded)
        except Exception as error:  # noqa: BLE001 - recording cannot stop speech
            segment.omit("upload_failed" if isinstance(error, UploadFailed) else type(error).__name__)
