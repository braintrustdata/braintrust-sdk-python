"""Bound encoded recordings while queued or retained by the exporter."""

import weakref

from braintrust.logger import Attachment

from .budget import ByteBudget
from .worker import RecordingBusy


encoded_budget = ByteBudget(32 * 1024 * 1024)


class RecordingAttachment(Attachment):
    def __init__(self, *, data, filename, content_type):
        self._retained_bytes = 0
        if not encoded_budget.reserve(len(data)):
            raise RecordingBusy("encoded_audio_capacity")
        self._retained_bytes = len(data)
        super().__init__(data=data, filename=filename, content_type=content_type)

    def _init_uploader(self):
        # The base uploader closes over its owner. Avoid keeping uploaded bytes
        # and their budget reservation alive until cyclic GC happens to run.
        return Attachment._init_uploader(weakref.proxy(self))

    def __del__(self):
        encoded_budget.release(getattr(self, "_retained_bytes", 0))
        self._retained_bytes = 0


def prepare_recording(name, function, *args):
    encoded = function(*args)
    if encoded is not None:
        encoded["attachment"] = RecordingAttachment(
            data=encoded["data"],
            filename=f"{name}.{encoded['extension']}",
            content_type=encoded["mime_type"],
        )
    return encoded
