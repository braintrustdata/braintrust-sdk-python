"""Prepare encoded recordings for the standard attachment exporter."""

from braintrust.logger import Attachment


def prepare_recording(name, function, *args):
    encoded = function(*args)
    if encoded is not None:
        encoded["attachment"] = Attachment(
            data=encoded["data"],
            filename=f"{name}.{encoded['extension']}",
            content_type=encoded["mime_type"],
        )
    return encoded
